"""
Shared slowapi Limiter instance. Lives in its own module (rather than
main.py) so auth_routes.py can import it too without a main<->auth_routes
circular import.

Client-IP keying: in the public ngrok setup (docs/DEPLOYMENT_PUBLIC.md) every
request reaches uvicorn from the local Vite proxy, so the direct peer is
always 127.0.0.1 and, keyed on that alone, every visitor would share one
bucket. So when -- and only when -- the direct peer is 127.0.0.1, the
X-Forwarded-For chain is used instead. It's read right to left, skipping
loopback hops (the Vite proxy / ngrok agent), and the first non-loopback
address is the one ngrok itself saw. Anything further left was supplied by
the client and is never trusted. A request from any other peer ignores the
header entirely, so it can't be spoofed from outside this machine.
"""

import ipaddress

from slowapi import Limiter
from starlette.requests import Request

TRUSTED_PROXY_PEERS = {"127.0.0.1"}


def _is_loopback(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return (mapped or addr).is_loopback


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "127.0.0.1"
    if peer not in TRUSTED_PROXY_PEERS:
        return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    for hop in reversed([h.strip() for h in forwarded.split(",")]):
        if not hop or _is_loopback(hop):
            continue
        try:
            ipaddress.ip_address(hop)
        except ValueError:
            # Malformed entry: stop rather than skip past it to a
            # client-supplied value further left.
            break
        return hop
    return peer


limiter = Limiter(key_func=client_ip)
