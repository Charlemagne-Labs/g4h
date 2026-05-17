"""Targeted DOM/header fetch for the live demo (Playwright + Chromium).

Runs server-side only — never in the model's training/eval path. Produces a
list of indicator strings to APPEND to whatever `src.extract.extract_indicators`
returned for the URL, then the combined list is fed to the model.

Design constraints:
  - Realistic Chrome request fingerprint (User-Agent + Accept headers + viewport
    + locale) to avoid the "blocked by bot detection" outcome on real sites.
  - Hard timeout per request (default 10s). Network failures, blocked targets,
    captchas, and shapeshifting JS all bucket to "no enrichment indicators
    added" — never an exception out of this module.
  - Specific error types get mapped to specific training-data indicator names
    (security:ssl_error, network:dns_error, network:timeout,
    network:connection_refused) so the model can use them.
  - DOM-content-loaded waiting strategy, NOT `networkidle` — networkidle hangs
    for ad-heavy phishing pages forever.
  - We DO fetch the page, which means we DO execute the target's JS. Don't
    point this at anything you wouldn't open in a sandboxed browser tab.

Indicators contributed (additive, only those whose check fires):
  - Errors: security:ssl_error, network:dns_error, network:timeout,
    network:connection_refused, http:error, fetch:error
  - Headers: security:hsts_present, security:csp_present,
    security:clickjack_protection_present
  - Outcome: security:https_on_trusted_domain, security:redirect_to_different_domain
  - DOM: seo:canonical_self_domain, content:no_literal_form_tag,
    content:login_form_official_domain, content:form_submits_externally,
    content:credential_form_suspicious, content:brand_impersonation,
    content:assets_trusted_cdns_majority
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_EXTRA_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
}
_VIEWPORT = {"width": 1366, "height": 768}
_LOCALE = "en-US"
_DEFAULT_TIMEOUT_MS = 10_000

# Major CDN domains — used for the "trusted assets" heuristic on a target page.
_TRUSTED_CDN_HOSTS = (
    "cloudflare.com", "cloudfront.net", "akamai", "fastly.net", "jsdelivr.net",
    "unpkg.com", "cdnjs.cloudflare.com", "googleapis.com", "gstatic.com",
    "googleusercontent.com", "azureedge.net", "bootstrapcdn.com",
)

# Tiny brand list — kept in sync with src/extract.py. Used for impersonation
# (page mentions a brand whose canonical domain doesn't match the site host).
_OFFICIAL_BRANDS = {
    "google": "google.com", "paypal": "paypal.com",
    "microsoft": "microsoft.com", "apple": "apple.com",
    "amazon": "amazon.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "linkedin": "linkedin.com",
    "netflix": "netflix.com", "github": "github.com",
    "coinbase": "coinbase.com", "wellsfargo": "wellsfargo.com",
}


# Chromium net::ERR_* → indicator mapping
_ERROR_INDICATOR_MAP = (
    ("ERR_SSL_PROTOCOL_ERROR",     "security:ssl_error:{}"),
    ("ERR_CERT_",                  "security:ssl_error:{}"),
    ("SSL_ERROR",                  "security:ssl_error:{}"),
    ("ERR_NAME_NOT_RESOLVED",      "network:dns_error:{}"),
    ("ERR_NAME_RESOLUTION_FAILED", "network:dns_error:{}"),
    ("ERR_CONNECTION_TIMED_OUT",   "network:timeout:{}"),
    ("ERR_TIMED_OUT",              "network:timeout:{}"),
    ("Timeout",                    "network:timeout:{}"),
    ("ERR_CONNECTION_REFUSED",     "network:connection_refused:{}"),
    ("ERR_CONNECTION_RESET",       "network:connection_refused:{}"),
    ("ERR_TOO_MANY_REDIRECTS",     "http:error:{\"reason\":\"redirect_loop\"}"),
)


def _error_to_indicators(error_message: str) -> list[str]:
    """Map a Playwright error message to specific training-data indicators."""
    indicators: list[str] = []
    msg = error_message or ""
    for needle, indicator in _ERROR_INDICATOR_MAP:
        if needle in msg:
            indicators.append(indicator)
            break
    # Always also emit a generic fetch:error so the model has the high-level
    # "page didn't load" signal.
    indicators.append(_ind("fetch:error", reason=msg[:80]))
    return indicators


@dataclass
class FetchResult:
    final_url: str | None
    status: int | None
    response_headers: dict[str, str]
    title: str | None
    indicators: list[str]
    error: str | None = None


def _ind(category_name: str, **kwargs) -> str:
    return f"{category_name}:{json.dumps(kwargs, separators=(',', ':'))}"


def _safe_host(url: str | None) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except Exception:
        return ""


def _registrable_domain(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


async def fetch_enrich(url: str, *, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> FetchResult:
    """Navigate to `url`, extract targeted DOM/header signals, return indicators.

    Fail-soft: any error during navigation, timeout, or extraction produces a
    `FetchResult` with `error` set and specific error indicators added (e.g.
    security:ssl_error, network:dns_error). Never raises.
    """
    try:
        from playwright.async_api import async_playwright, Error as PWError, TimeoutError as PWTimeout
    except ImportError as e:
        return FetchResult(
            final_url=None, status=None, response_headers={}, title=None,
            indicators=[_ind("fetch:error", reason="playwright_not_installed")],
            error=f"playwright not installed: {e}",
        )

    input_host = _safe_host(url)
    input_reg = _registrable_domain(input_host) if input_host else ""

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--no-sandbox", "--disable-blink-features=AutomationControlled",
            ])
            context = await browser.new_context(
                user_agent=_USER_AGENT,
                viewport=_VIEWPORT,
                locale=_LOCALE,
                extra_http_headers=_EXTRA_HEADERS,
                ignore_https_errors=True,
            )
            page = await context.new_page()
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except (PWTimeout, PWError) as e:
                await browser.close()
                return FetchResult(
                    final_url=None, status=None, response_headers={}, title=None,
                    indicators=_error_to_indicators(str(e)),
                    error=str(e)[:200],
                )

            if response is None:
                await browser.close()
                return FetchResult(
                    final_url=None, status=None, response_headers={}, title=None,
                    indicators=[_ind("fetch:error", reason="no_response")],
                    error="navigation returned no response",
                )

            final_url = response.url
            status = response.status
            headers_lower = {k.lower(): v for k, v in response.headers.items()}
            indicators: list[str] = []

            # HTTP-level error status
            if status >= 400:
                indicators.append(_ind("http:error", status=status))

            # --- Header-based security indicators ---
            if "strict-transport-security" in headers_lower:
                indicators.append("security:hsts_present:{}")
            csp = headers_lower.get("content-security-policy", "")
            if csp:
                indicators.append("security:csp_present:{}")
            if "x-frame-options" in headers_lower or "frame-ancestors" in csp.lower():
                indicators.append("security:clickjack_protection_present:{}")

            # HTTPS on a domain whose response succeeded
            if final_url.startswith("https://") and 200 <= status < 400:
                final_reg = _registrable_domain(_safe_host(final_url))
                if final_reg == input_reg:
                    indicators.append("security:https_on_trusted_domain:{}")

            # Redirect-to-different-domain signal
            final_host = _safe_host(final_url)
            if input_host and final_host and final_host != input_host:
                if _registrable_domain(final_host) != input_reg:
                    indicators.append(_ind(
                        "security:redirect_to_different_domain",
                        from_host=input_host, to_host=final_host,
                    ))

            # --- DOM analysis ---
            try:
                title = await page.title()
            except Exception:
                title = None

            try:
                form_count = await page.locator("form").count()
            except Exception:
                form_count = 0

            try:
                has_password = await page.locator('input[type="password"]').count() > 0
            except Exception:
                has_password = False

            form_actions: list[str] = []
            canonical_href: str = ""
            script_srcs: list[str] = []
            try:
                doc_info = await page.evaluate("""
                    () => ({
                        actions: Array.from(document.querySelectorAll('form'))
                                     .map(f => f.action || ''),
                        canonical: (document.querySelector('link[rel=canonical]') || {}).href || '',
                        scripts: Array.from(document.querySelectorAll('script[src]'))
                                      .map(s => s.src).slice(0, 30),
                    })
                """)
                if isinstance(doc_info, dict):
                    a = doc_info.get("actions") or []
                    if isinstance(a, list):
                        form_actions = [x for x in a if isinstance(x, str) and x]
                    canonical_href = doc_info.get("canonical") or ""
                    s = doc_info.get("scripts") or []
                    if isinstance(s, list):
                        script_srcs = [x for x in s if isinstance(x, str)]
            except Exception:
                pass

            # No literal <form> tag at all
            if form_count == 0:
                indicators.append("content:no_literal_form_tag:{}")

            # Login form, on-domain vs off-domain action analysis
            if has_password and form_actions:
                offsite = [
                    a for a in form_actions
                    if _safe_host(a) and _registrable_domain(_safe_host(a)) != input_reg
                ]
                if offsite:
                    indicators.append(_ind(
                        "content:form_submits_externally",
                        action_host=_safe_host(offsite[0]),
                    ))
                    indicators.append("content:credential_form_suspicious:{}")
                else:
                    indicators.append(_ind(
                        "content:login_form_official_domain",
                        action_domain=input_reg,
                    ))

            # Canonical link on the same domain → benign signal
            if canonical_href:
                canon_reg = _registrable_domain(_safe_host(canonical_href))
                if canon_reg == input_reg:
                    indicators.append(_ind(
                        "seo:canonical_self_domain",
                        canonical_domain=canon_reg,
                    ))

            # Majority of script CDNs on trusted hosts → benign
            if script_srcs:
                trusted_count = 0
                total_external = 0
                for src in script_srcs:
                    h = _safe_host(src)
                    if not h or _registrable_domain(h) == input_reg:
                        continue
                    total_external += 1
                    if any(t in h for t in _TRUSTED_CDN_HOSTS):
                        trusted_count += 1
                if total_external >= 3:
                    ratio = trusted_count / total_external
                    if ratio >= 0.7:
                        indicators.append(_ind(
                            "content:assets_trusted_cdns_majority",
                            ratio=round(ratio, 2), total=total_external,
                        ))

            # Brand impersonation: page title mentions a known brand whose
            # canonical domain doesn't match the host.
            if title and input_reg:
                title_lower = title.lower()
                for brand, brand_domain in _OFFICIAL_BRANDS.items():
                    if brand in title_lower and input_reg != brand_domain:
                        indicators.append(_ind(
                            "content:brand_impersonation",
                            brand=brand, on_domain=input_reg,
                        ))
                        break

            await browser.close()
            return FetchResult(
                final_url=final_url, status=status,
                response_headers={k: v for k, v in headers_lower.items() if k in {
                    "content-security-policy", "strict-transport-security",
                    "x-frame-options", "content-type", "server",
                }},
                title=(title or "")[:200] if title else None,
                indicators=indicators,
            )
    except Exception as e:
        return FetchResult(
            final_url=None, status=None, response_headers={}, title=None,
            indicators=_error_to_indicators(str(e)),
            error=str(e)[:200],
        )


def fetch_enrich_sync(url: str, **kw) -> FetchResult:
    return asyncio.run(fetch_enrich(url, **kw))
