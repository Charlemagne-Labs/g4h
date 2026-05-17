"""URL-only feature extraction for the live demo.

Produces indicator strings in the format the trained model expects:
    `category:name:{"key":"value", ...}` separated by spaces.

This is a CLEAN-ROOM minimal extractor — implemented from scratch based on
the indicator names observable in the training data, NOT a port of any
production extractor. Only URL-string signals; no HTTP fetches, no DNS,
no WHOIS. The DOM-fetch enrichment lives separately in `server/fetch.py`
and runs ONLY in the webapp's cloud-side request path.

Coverage of the 58 indicators present in training data:
  - 12 URL-only indicators implemented here (security:no_https,
    url:ip_hostname, url:suspicious_tld, url:excessive_hyphens,
    url:confusable_hostname, domain:official_brand_domain,
    domain:trusted_whitelist_hit, domain:brand_impersonation_subdomain,
    domain:brand_lookalike, hosting:free_or_dynamic_platform,
    intent:phishing_phrase, intent:credential_capture)
  - The remaining ~46 indicators require HTTP/DOM access; `server/fetch.py`
    contributes additional indicators from a targeted Playwright fetch.

Output:
    extract_indicators(url) -> list[str]   # ordered list of indicator strings
    extract_text(url) -> str               # space-joined, matches model input format
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

# Lists below are small, public, and replicated from any phishing-detection
# 101 reference. They are NOT taken from any internal production data.

_SUSPICIOUS_TLDS = {
    ".ml", ".cf", ".tk", ".ga", ".gq",
    ".top", ".xyz", ".pw", ".live", ".rest",
}

# Free / dynamic hosting platforms commonly hosting phishing pages.
_FREE_HOSTING = {
    "netlify.app", "vercel.app", "pages.dev", "herokuapp.com",
    "blogspot.com", "wuaze.com", "000webhostapp.com", "github.io",
    "weebly.com", "wixsite.com",
}

# Tiny brand-impersonation reference. Each entry maps a brand label to
# its canonical apex domain.
_OFFICIAL_BRANDS = {
    "google": "google.com", "paypal": "paypal.com",
    "microsoft": "microsoft.com", "apple": "apple.com",
    "amazon": "amazon.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "linkedin": "linkedin.com",
    "netflix": "netflix.com", "github": "github.com",
    "coinbase": "coinbase.com", "wellsfargo": "wellsfargo.com",
}

_PHISHING_PHRASES = (
    "verify", "suspend", "restrict", "expire", "urgent",
    "confirm", "secure-update", "account-update", "verify-account",
)

_CREDENTIAL_KEYWORDS = ("login", "signin", "auth", "sign-in", "log-in")

_IP_HOSTNAME_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _ind(category_name: str, **kwargs) -> str:
    """Format one indicator as `category:name:{compact-json}`."""
    return f"{category_name}:{json.dumps(kwargs, separators=(',', ':'))}"


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


_HOST_TOKEN_RE = re.compile(r"[.\-_]")


def _is_brand_lookalike(host: str) -> tuple[bool, str | None]:
    """One-edit Levenshtein against the brand list. Splits the hostname on
    `.`, `-`, and `_` so multi-token labels like `paypa1-secure-verify` get
    inspected one token at a time. Skips exact matches.
    """
    tokens = [t for t in _HOST_TOKEN_RE.split(host) if t and t != "com"]
    for token in tokens:
        for brand in _OFFICIAL_BRANDS:
            if token == brand:
                return False, None
            if abs(len(token) - len(brand)) <= 1 and _edit_distance(token, brand) == 1:
                return True, brand
    return False, None


def extract_indicators(url: str) -> list[str]:
    """Run all URL-only checks against `url` and return matched indicators.

    Returns an empty-but-not-None list of `["meta:no_indicators:{}"]` when
    nothing fires — the model was trained with that sentinel for clean URLs.
    """
    if not url or not isinstance(url, str):
        return ['meta:no_indicators:{}']
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip(".")
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    scheme = parsed.scheme

    out: list[str] = []

    if _IP_HOSTNAME_RE.match(host):
        out.append(_ind("url:ip_hostname", hostname=host))

    if scheme != "https":
        out.append(_ind("security:no_https", scheme=scheme))

    if "." in host:
        tld = "." + host.rsplit(".", 1)[-1]
        if tld in _SUSPICIOUS_TLDS:
            out.append(_ind("url:suspicious_tld", tld=tld))

    if host.count("-") >= 3:
        out.append(_ind("url:excessive_hyphens", count=host.count("-")))

    if "xn--" in host:
        out.append(_ind("url:confusable_hostname", reason="punycode"))

    # Brand presence checks
    official_brand = None
    for brand, domain in _OFFICIAL_BRANDS.items():
        if host == domain or host.endswith("." + domain):
            official_brand = brand
            out.append(_ind("domain:official_brand_domain", brand=brand))
            out.append("domain:trusted_whitelist_hit:{}")
            break

    if official_brand is None:
        # Brand-name appearing as a substring in any hostname token
        # (e.g. paypal.evil.com, secure-paypal-update.xyz, paypal-login.tk)
        host_tokens = [t for t in _HOST_TOKEN_RE.split(host) if t]
        for brand in _OFFICIAL_BRANDS:
            if any(brand in tok for tok in host_tokens):
                out.append(_ind("domain:brand_impersonation_subdomain", brand=brand))
                break

        # Typo lookalike across tokens (paypa1, googl, microsft, etc.)
        is_la, brand = _is_brand_lookalike(host)
        if is_la:
            out.append(_ind("domain:brand_lookalike", brand=brand))

    # Free hosting platforms
    for fh in _FREE_HOSTING:
        if host.endswith(fh):
            out.append(_ind("hosting:free_or_dynamic_platform", platform=fh))
            break

    # Phishing-intent phrases in path or query
    text_for_intent = path + " " + query
    for phrase in _PHISHING_PHRASES:
        if phrase in text_for_intent:
            out.append(_ind("intent:phishing_phrase", phrase=phrase))
            break

    # Credential capture signal
    if any(kw in path for kw in _CREDENTIAL_KEYWORDS):
        out.append("intent:credential_capture:{}")

    if not out:
        out.append("meta:no_indicators:{}")

    return out


def extract_text(url: str) -> str:
    """Return the space-joined indicator string the model expects as input."""
    return " ".join(extract_indicators(url))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.extract <url>", file=sys.stderr)
        raise SystemExit(2)
    print(extract_text(sys.argv[1]))
