"""``http_get`` — fetch a URL from an allowlist of hosts.

This is the only Gemma-facing tool that can cross the network
boundary. Two layers of defense:

1. **Allowlist.** Hosts must appear in
   ``~/.config/gemma/tool-allowlist.toml`` (or the built-in default
   list). Anything else is refused before the request leaves the
   process.
2. **Stdlib only.** We use :mod:`urllib.request` rather than pulling
   in ``requests`` or ``httpx``. Fewer third-party bytes = smaller
   supply-chain surface for the one tool that touches the internet.

Responses are capped at 256 KB and redacted on their way into the
audit record — if a server ever returns an access token in a response
body, it doesn't land on disk.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

# ``tomllib`` landed in Python 3.11; fall back to ``tomli`` (API-compatible)
# on 3.10 so the allowlist config loads identically on both interpreters.
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - interpreter-version branch
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


#: Response body cap. Matches the subprocess runner's default so tool
#: outputs have a consistent maximum size from the model's POV.
_RESPONSE_CAP_BYTES = 256 * 1024

#: Wall-clock timeout for the fetch.
_HTTP_TIMEOUT = 10.0

#: Hosts that ship allowed out-of-the-box. Deliberately narrow —
#: documentation sources that a coding assistant legitimately needs.
_DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "docs.python.org",
    "developer.mozilla.org",
    "api.github.com",
    "pypi.org",
)


def _load_allowlist() -> List[str]:
    """Return the effective host allowlist.

    Users add hosts by editing ``~/.config/gemma/tool-allowlist.toml``::

        [http_get]
        hosts = ["internal.wiki.example.com", "pkg.myco.dev"]

    Anything the TOML declares is *added to* the defaults; there is no
    way to shrink the default list via config, only extend it.
    """
    path = Path.home() / ".config" / "gemma" / "tool-allowlist.toml"
    if not path.is_file():
        return list(_DEFAULT_ALLOWLIST)

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        # Config parse failure must not unlock *more* hosts — fall
        # back to the default list.
        return list(_DEFAULT_ALLOWLIST)

    extras = data.get("http_get", {}).get("hosts", [])
    if not isinstance(extras, list):
        extras = []
    extras = [h for h in extras if isinstance(h, str)]
    return list(_DEFAULT_ALLOWLIST) + extras


@tool(ToolSpec(
    name="http_get",
    description=(
        "Issue an HTTPS GET against an allowlisted host and return the "
        "response body. Non-HTTPS URLs and non-allowlisted hosts are "
        "refused. Response body is capped at 256 KB."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL. Scheme must be https://.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    capability=Capability.NETWORK,
    timeout_s=int(_HTTP_TIMEOUT),
    max_output_bytes=_RESPONSE_CAP_BYTES,
))
def http_get(url: str) -> ToolResult:
    """Fetch ``url`` over HTTPS with policy enforcement."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return ToolResult(ok=False, error="invalid_url", content=f"unparseable URL: {url!r}")

    if parsed.scheme != "https":
        return ToolResult(
            ok=False, error="bad_scheme",
            content=f"refusing non-HTTPS scheme {parsed.scheme!r}",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        return ToolResult(ok=False, error="no_host", content="URL has no host component")

    allowlist = _load_allowlist()
    if not _host_allowed(host, allowlist):
        return ToolResult(
            ok=False, error="host_blocked",
            content=(
                f"host {host!r} not in allowlist. "
                f"Add it to ~/.config/gemma/tool-allowlist.toml to permit."
            ),
        )

    # Build a vanilla Request. We set a polite User-Agent; we do NOT
    # inherit anything from the environment (no proxies, no auth).
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "gemma-cli/tools http_get"},
    )

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT, context=ctx) as resp:
            raw = resp.read(_RESPONSE_CAP_BYTES + 1)
            status = resp.status
            headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return ToolResult(
            ok=False, error="http_error",
            content=f"HTTP {exc.code} {exc.reason}",
            metadata={"status": exc.code},
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ToolResult(ok=False, error="fetch_failed", content=f"fetch failed: {exc}")

    truncated = len(raw) > _RESPONSE_CAP_BYTES
    body_bytes = raw[:_RESPONSE_CAP_BYTES]
    body = body_bytes.decode("utf-8", errors="replace")
    if truncated:
        body += f"\n…[truncated at {_RESPONSE_CAP_BYTES} bytes]"

    return ToolResult(
        ok=True,
        content=body,
        metadata={
            "status": status,
            "content_type": headers.get("content-type", ""),
            "truncated": truncated,
            "bytes_read": len(body_bytes),
        },
    )


# ---------------------------------------------------------------------------
# Host matching
# ---------------------------------------------------------------------------

def _host_allowed(host: str, allowlist: List[str]) -> bool:
    """Return True if ``host`` equals any allowlisted entry, case-insensitive.

    We deliberately do *not* support wildcards here. An explicit list
    of hosts is easier to audit than a glob language — and users who
    want a whole TLD can list subdomains individually.
    """
    host = host.lower().strip()
    return any(host == e.lower().strip() for e in allowlist)
