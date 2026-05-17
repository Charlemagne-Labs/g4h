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
import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs

# Lists below are small, public, and informed by samples from OpenPhish
# (a public phishing-URL feed). They are NOT taken from any internal
# production data.

# TLDs commonly abused in phishing campaigns (sampled from current OpenPhish
# feed: .cyou, .shop, .vip, .link, .top, .ml, etc. all appeared 9+ times in
# a 300-URL sample). We include generic gTLDs like .digital/.site/.online
# which are cheap and popular for fast-spun phishing domains.
_SUSPICIOUS_TLDS = {
    ".ml", ".cf", ".tk", ".ga", ".gq",
    ".top", ".xyz", ".pw", ".live", ".rest",
    ".cyou", ".shop", ".vip", ".link",
    ".digital", ".site", ".online", ".click", ".work", ".life",
}

# Free / dynamic hosting platforms commonly hosting phishing pages.
# github.io is included despite many legitimate uses — it appeared 27 times
# in our 300-URL OpenPhish sample as a phishing host. The model has both
# allow and block training examples for github.io, so this is a soft signal.
_FREE_HOSTING = {
    "netlify.app", "vercel.app", "pages.dev", "herokuapp.com",
    "blogspot.com", "wuaze.com", "000webhostapp.com",
    "weebly.com", "weeblysite.com", "wixsite.com",
    "github.io", "framer.app", "workers.dev", "iceiy.com",
    "alwaysdata.net", "r2.dev", "sevalla.page", "dweb.link",
    "compute-1.amazonaws.com", "compute.amazonaws.com",
    "azurewebsites.net", "cloudfront.net",
}

# Brand-impersonation reference. Expanded from the OpenPhish sample where
# roblox/outlook/chase/att/bancolombia/imtoken/bet365 etc. were all
# represented.
_OFFICIAL_BRANDS = {
    # Big tech / consumer
    "google": "google.com", "paypal": "paypal.com",
    "microsoft": "microsoft.com", "apple": "apple.com",
    "amazon": "amazon.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "linkedin": "linkedin.com",
    "netflix": "netflix.com", "github": "github.com",
    "outlook": "outlook.com", "roblox": "roblox.com",
    "ebay": "ebay.com", "att": "att.com",
    # Financial
    "coinbase": "coinbase.com", "wellsfargo": "wellsfargo.com",
    "chase": "chase.com", "fidelity": "fidelity.com",
    "bancolombia": "bancolombia.com", "santander": "santander.com",
    "itau": "itau.com.br", "bradesco": "bradesco.com.br",
    # Crypto / wallets
    "binance": "binance.com", "imtoken": "token.im",
    "metamask": "metamask.io", "trustwallet": "trustwallet.com",
    # Other
    "bet365": "bet365.com",
}

_PHISHING_PHRASES = (
    # English
    "verify", "suspend", "restrict", "expire", "urgent",
    "confirm", "secure-update", "account-update", "verify-account",
    "unlock", "limited", "reactivate", "expired", "reward",
    "claim", "winner", "prize", "gift-card",
    # Spanish / Portuguese (common in LATAM-targeted phishing)
    "verificar", "actualizar", "atualizar", "acceder", "acesso",
    "cadastro", "seguranca", "seguridad", "confirmar", "ingresar",
)

_CREDENTIAL_KEYWORDS = ("login", "signin", "auth", "sign-in", "log-in")

# Well-known URL shorteners that often hide phishing destinations.
_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "shorturl.at", "cutt.ly",
}

# Cross-channel mention keywords (telegram/whatsapp/skype/discord/crypto
# wallet refs in URLs are training-data indicators).
_CROSS_CHANNEL = {
    "telegram": "telegram", "t.me": "telegram",
    "whatsapp": "whatsapp", "wa.me": "whatsapp",
    "discord": "discord", "discord.gg": "discord",
    "skype": "skype",
    "metamask": "crypto", "wallet": "crypto",
}

# Sensitive file extensions or paths that signal payload-style attacks.
_SENSITIVE_PATH_BITS = (".exe", ".scr", ".zip", ".rar", "/admin", "/wp-admin")

# Query parameter names that hint at credential capture or open redirects.
_REDIRECT_PARAMS = {"redirect", "url", "next", "return", "returnurl", "continue", "redir"}
_CREDENTIAL_PARAMS = {"key", "pin", "otp", "token", "session"}

_IP_HOSTNAME_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_PCT_ENCODED_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


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


def _shannon_entropy(s: str) -> float:
    """Bits-per-character entropy. High values suggest a random-looking
    (DGA-like) domain string."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


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

    # --- New URL-only signals (Phase 4 expansion) ---

    # Long URL (commonly used to hide the real destination)
    if len(url) > 100:
        out.append(_ind("url:long_url", length=len(url)))

    # Known URL shorteners
    if any(host == sh or host.endswith("." + sh) for sh in _URL_SHORTENERS):
        out.append(_ind("url:url_shortener", host=host))

    # Excessive %-encoded characters in path (>5 → likely obfuscation)
    enc_count = len(_PCT_ENCODED_RE.findall(parsed.path or ""))
    if enc_count >= 5:
        out.append(_ind("path:excessive_encoding", count=enc_count))

    # Non-ASCII in path (mixed scripts / homograph attempts)
    if _NON_ASCII_RE.search(parsed.path or ""):
        out.append("path:mixed_scripts:{}")

    # Sensitive file extension / admin paths
    for bit in _SENSITIVE_PATH_BITS:
        if bit in path:
            out.append(_ind("path:sensitive_keyword_file", marker=bit))
            break

    # Redirect chain hints in query string
    query_params = parse_qs(query) if query else {}
    for q in query_params:
        if q.lower() in _REDIRECT_PARAMS:
            out.append(_ind("query:external_redirect", param=q))
            break

    # Credential-looking query params (key=, pin=, otp=)
    for q in query_params:
        ql = q.lower()
        if ql in _CREDENTIAL_PARAMS:
            out.append(_ind(f"context_field:{ql}", param=q))
            break  # one is enough to signal

    # Cross-channel mentions (telegram, whatsapp, etc. embedded in URL)
    full_url_text = (host + " " + path + " " + query).lower()
    for needle, channel in _CROSS_CHANNEL.items():
        if needle in full_url_text:
            out.append(_ind(f"cross_channel:{channel}_mention", token=needle))
            break

    # High-entropy registered domain (random-looking, DGA-ish).
    # Threshold 3.3 catches all-unique consonant strings (entropy ~3.5+)
    # without over-firing on normal English domains (entropy ~2.5-3.0).
    if "." in host:
        registered = host.rsplit(".", 2)[0] if host.count(".") >= 2 else host.split(".")[0]
        if len(registered) >= 7:
            e = _shannon_entropy(registered)
            if e > 3.3:
                out.append(_ind("domain:high_entropy", entropy=round(e, 2)))

    # Keyword stuffing: phishing keywords in the hostname.
    # Two tiers:
    #   HIGH_SIGNAL — LATAM financial/registration words. Strong enough to
    #     fire on a single hit (legit sites rarely have "cadastro" or
    #     "bancolombia" in non-bank hostnames).
    #   LOW_SIGNAL — generic English words also common on legit sites.
    #     Requires 2+ hits to fire.
    host_lower = host.lower()
    high_signal = ("cadastro", "banco", "conta", "cuenta",
                   "seguranca", "seguridad", "bancolombia", "santander",
                   "bradesco")
    low_signal = ("secure", "login", "verify", "account", "bank", "update",
                  "support", "auth", "signin", "confirm", "official",
                  "acesso", "cliente", "acceso", "ingresar")
    high_hits = sum(1 for kw in high_signal if kw in host_lower)
    low_hits = sum(1 for kw in low_signal if kw in host_lower)
    if high_hits >= 1 or (high_hits + low_hits) >= 2:
        out.append(_ind("domain:keyword_stuffing", count=high_hits + low_hits))

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
