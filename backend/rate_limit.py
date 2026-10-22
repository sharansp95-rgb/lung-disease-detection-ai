"""
Shared slowapi Limiter instance. Lives in its own module (rather than
main.py) so auth_routes.py can import it too without a main<->auth_routes
circular import.

Client-IP keying: behind a reverse proxy the direct peer is the proxy, so
keyed on that alone every visitor would share one bucket. So when -- and
only when -- the direct peer is a trusted proxy (config.TRUSTED_PROXY_IPS),
the X-Forwarded-For chain is used instead. It's read right to left,
skipping loopback hops and hops that are themselves trusted proxies, and
the first remaining address is the one the outermost trusted proxy saw.
Anything further left was supplied by the client and is never trusted. A
request from any other peer ignores the header entirely, so it can't be
spoofed by connecting directly.

  - Local ngrok setup (docs/DEPLOYMENT_PUBLIC.md): the default
    TRUSTED_PROXY_IPS=127.0.0.1 trusts only the local Vite proxy.
  - EC2 deploy (docker-compose.yml): TRUSTED_PROXY_IPS is the Caddy
    container's fixed address. Caddy doesn't trust incoming X-Forwarded-For
    from the internet, so it sends exactly one hop: the visitor's IP.
"""

import ipaddress
import logging

from slowapi import Limiter
from starlette.requests import Request

import config

logger = logging.getLogger("ct_xray_api.rate_limit")


def _parse_networks(values: list[str]):
    networks = []
    for value in values:
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            logger.warning(f"Ignoring invalid TRUSTED_PROXY_IPS entry: {value!r}")
    return networks


TRUSTED_PROXY_NETWORKS = _parse_networks(config.TRUSTED_PROXY_IPS)


def _parse_ip(value: str):
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return None
    return getattr(addr, "ipv4_mapped", None) or addr


def _in_trusted_networks(addr) -> bool:
    return any(addr.version == net.version and addr in net for net in TRUSTED_PROXY_NETWORKS)


def is_trusted_proxy(peer: str | None) -> bool:
    addr = _parse_ip(peer) if peer else None
    return addr is not None and _in_trusted_networks(addr)


_logged_forwarded_example = False


def client_ip(request: Request) -> str:
    global _logged_forwarded_example
    peer = request.client.host if request.client else "127.0.0.1"
    if not is_trusted_proxy(peer):
        return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    result = peer
    for hop in reversed([h.strip() for h in forwarded.split(",")]):
        if not hop:
            continue
        addr = _parse_ip(hop)
        if addr is None:
            # Malformed entry: stop rather than skip past it to a
            # client-supplied value further left.
            break
        if addr.is_loopback or _in_trusted_networks(addr):
            continue
        result = hop
        break

    if forwarded and not _logged_forwarded_example:
        # Once per process, so a new deployment's proxy chain can be checked
        # against TRUSTED_PROXY_IPS from the logs.
        _logged_forwarded_example = True
        logger.info(f"Rate-limit key from proxy chain: peer={peer} x-forwarded-for={forwarded!r} -> {result}")
    return result


limiter = Limiter(key_func=client_ip)
