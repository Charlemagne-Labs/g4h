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
  - No JavaScript evaluation other than reading the rendered DOM via
    `page.locator(...)`. We don't run user code from the target page.
  - DOM-content-loaded waiting strategy, NOT `networkidle` — networkidle hangs
    for ad-heavy phishing pages forever.
  - We DO fetch the page, which means we DO execute the target's JS. Don't
    point this at anything you wouldn't open in a sandboxed browser tab.

Indicators contributed (additive, only those whose check fires):
  - security:hsts_present, security:csp_present, security:clickjack_protection_present
  - security:https_on_trusted_domain        (TLS scheme + 2xx final response)
  - security:redirect_to_different_domain   (final URL host != input host)
  - content:login_form_official_domain      (page has password input + form action stays on-domain)
  - content:form_submits_externally         (page has form whose action goes off-domain)
  - content:credential_form_suspicious      (login form + suspicious heuristics)
  - fetch:error                             (request failed; the model has seen this signal in training)
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Realistic Chrome 131 fingerprint. Keep this in lockstep with whatever the
# real Chrome stable release is doing — older fingerprints get bot-flagged.
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

# How long to wait for the navigation + DOM content. Hard cap — phishing
# pages routinely take forever or hang. ~10s is generous; production should
# probably tighten.
_DEFAULT_TIMEOUT_MS = 10_000


@dataclass
class FetchResult:
    """What a successful fetch produced. Used both for the indicator list and
    for the response metadata we surface in the webapp UI."""
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
    """Crude registrable-domain extractor: last two labels. Good enough for
    same-domain checks (paypal.com vs paypal.evil.com); doesn't handle
    co.uk-style TLDs. Acceptable for the live demo."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


async def fetch_enrich(url: str, *, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> FetchResult:
    """Navigate to `url`, extract targeted DOM/header signals, return indicators.

    Fail-soft: any error during navigation, timeout, or extraction produces a
    `FetchResult` with `error` set and a single `fetch:error:{...}` indicator
    appended. Never raises.
    """
    # Lazy import: playwright pulls a 300 MB Chromium dependency. Don't make
    # `pip install -e .` users on Mac install it just to import.
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
                ignore_https_errors=True,  # we want to see the page even if cert is bad
            )
            page = await context.new_page()
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except (PWTimeout, PWError) as e:
                await browser.close()
                return FetchResult(
                    final_url=None, status=None, response_headers={}, title=None,
                    indicators=[_ind("fetch:error", reason=type(e).__name__)],
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

            # --- Header-based security indicators ---
            if "strict-transport-security" in headers_lower:
                indicators.append("security:hsts_present:{}")
            if "content-security-policy" in headers_lower:
                indicators.append("security:csp_present:{}")
            if "x-frame-options" in headers_lower or "frame-ancestors" in headers_lower.get("content-security-policy", "").lower():
                indicators.append("security:clickjack_protection_present:{}")

            # HTTPS on a domain whose response succeeded — proxy for "looks fine"
            if final_url.startswith("https://") and 200 <= status < 400:
                final_reg = _registrable_domain(_safe_host(final_url))
                if final_reg == input_reg:
                    indicators.append("security:https_on_trusted_domain:{}")

            # --- Redirect signal ---
            final_host = _safe_host(final_url)
            if input_host and final_host and final_host != input_host:
                if _registrable_domain(final_host) != input_reg:
                    indicators.append(_ind(
                        "security:redirect_to_different_domain",
                        from_host=input_host, to_host=final_host,
                    ))

            # --- Targeted DOM analysis ---
            try:
                title = await page.title()
            except Exception:
                title = None

            try:
                has_password = await page.locator('input[type="password"]').count() > 0
            except Exception:
                has_password = False

            form_actions: list[str] = []
            try:
                actions = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('form'))"
                    ".map(f => f.action || '')"
                )
                if isinstance(actions, list):
                    form_actions = [a for a in actions if isinstance(a, str) and a]
            except Exception:
                pass

            # Login form on the same domain → benign signal; off-domain → red flag
            if has_password and form_actions:
                offsite_actions = [
                    a for a in form_actions
                    if _registrable_domain(_safe_host(a)) != input_reg
                    and _safe_host(a)  # ignore relative URLs
                ]
                if offsite_actions:
                    indicators.append(_ind(
                        "content:form_submits_externally",
                        action_host=_safe_host(offsite_actions[0]),
                    ))
                    indicators.append("content:credential_form_suspicious:{}")
                else:
                    indicators.append(_ind(
                        "content:login_form_official_domain",
                        action_domain=input_reg,
                    ))

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
        # Anything else (Playwright install missing, Chromium crash, etc.)
        # returns a fetch:error and we still let the URL-only path produce a verdict.
        return FetchResult(
            final_url=None, status=None, response_headers={}, title=None,
            indicators=[_ind("fetch:error", reason=type(e).__name__)],
            error=str(e)[:200],
        )


def fetch_enrich_sync(url: str, **kw) -> FetchResult:
    """Synchronous wrapper for environments without an event loop."""
    return asyncio.run(fetch_enrich(url, **kw))
