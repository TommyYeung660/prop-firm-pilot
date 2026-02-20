"""
MatchTrader API connectivity test — multi-endpoint Cloudflare bypass diagnostics.

Tests multiple login endpoints and header configurations to find a working
approach that bypasses Cloudflare's JavaScript challenge.

Run with:
    uv run python scripts/test_api_connectivity.py

Requires .env with:
    MATCHTRADER_API_URL, MATCHTRADER_USERNAME, MATCHTRADER_PASSWORD,
    MATCHTRADER_BROKER_ID, MATCHTRADER_ACCOUNT_ID
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Browser-like headers ────────────────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/json",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _make_headers(base_url: str) -> dict[str, str]:
    """Build full browser-like headers with Referer/Origin set to base_url."""
    headers = {**BROWSER_HEADERS}
    headers["Origin"] = base_url
    headers["Referer"] = f"{base_url}/"
    return headers


# ── Endpoint test functions ─────────────────────────────────────────────────


async def _test_endpoint_httpx(
    base_url: str,
    path: str,
    payload: dict,
    label: str,
    headers: dict[str, str],
) -> dict | None:
    """Try a login endpoint with httpx. Returns parsed JSON on success, None on failure."""
    import httpx

    url = f"{base_url}{path}"
    logger.info("  [{label}] POST {url}", label=label, url=url)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            resp = await client.post(url, json=payload, headers=headers)

            # Check for Cloudflare block
            body_preview = resp.text[:500]
            is_cloudflare = (
                "Just a moment" in body_preview
                or "cf-browser-verification" in body_preview
                or "Checking if the site connection is secure" in body_preview
                or resp.status_code == 403
                and "cloudflare" in resp.text.lower()
            )

            if is_cloudflare:
                logger.warning(
                    "  [{label}] ❌ CLOUDFLARE BLOCKED (status={status})",
                    label=label,
                    status=resp.status_code,
                )
                return None

            if resp.status_code >= 400:
                logger.warning(
                    "  [{label}] ❌ HTTP {status}: {body}",
                    label=label,
                    status=resp.status_code,
                    body=body_preview,
                )
                # Still return data if it's JSON — error messages can be informative
                try:
                    return resp.json()
                except Exception:
                    return None

            data = resp.json()
            logger.info(
                "  [{label}] ✅ SUCCESS (status={status}, keys={keys})",
                label=label,
                status=resp.status_code,
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return data

    except httpx.ConnectError as e:
        logger.error("  [{label}] ❌ CONNECTION ERROR: {err}", label=label, err=e)
        return None
    except Exception as e:
        logger.error("  [{label}] ❌ ERROR: {err}", label=label, err=e)
        return None


async def _test_endpoint_curl_cffi(
    base_url: str,
    path: str,
    payload: dict,
    label: str,
) -> dict | None:
    """Try a login endpoint with curl_cffi (browser TLS fingerprint). Returns JSON or None."""
    try:
        from curl_cffi.requests import AsyncSession
    except ImportError:
        logger.warning("  [{label}] ⚠ curl_cffi not installed — skipping", label=label)
        return None

    url = f"{base_url}{path}"
    logger.info("  [{label}] POST {url} (curl_cffi/chrome)", label=label, url=url)

    try:
        async with AsyncSession(impersonate="chrome") as session:
            resp = await session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            body_preview = resp.text[:500]
            is_cloudflare = (
                "Just a moment" in body_preview or "cf-browser-verification" in body_preview
            )

            if is_cloudflare:
                logger.warning(
                    "  [{label}] ❌ CLOUDFLARE BLOCKED even with curl_cffi (status={status})",
                    label=label,
                    status=resp.status_code,
                )
                return None

            if resp.status_code >= 400:
                logger.warning(
                    "  [{label}] ❌ HTTP {status}: {body}",
                    label=label,
                    status=resp.status_code,
                    body=body_preview,
                )
                try:
                    return resp.json()
                except Exception:
                    return None

            data = resp.json()
            logger.info(
                "  [{label}] ✅ SUCCESS (status={status}, keys={keys})",
                label=label,
                status=resp.status_code,
                keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return data

    except Exception as e:
        logger.error("  [{label}] ❌ ERROR: {err}", label=label, err=e)
        return None


# ── Extraction helpers ──────────────────────────────────────────────────────


def _extract_trading_api_domain(data: dict) -> str | None:
    """Extract tradingApiDomain from a login response (various formats)."""
    # Official format: tradingAccounts[].offer.system.tradingApiDomain
    for key in ("tradingAccounts", "accounts"):
        accounts = data.get(key, [])
        for acct in accounts:
            domain = acct.get("offer", {}).get("system", {}).get("tradingApiDomain")
            if domain:
                return domain
    return None


def _extract_token_and_uuid(data: dict, account_id: str | None) -> tuple[str, str] | None:
    """Extract (tradingApiToken, systemUUID) from a login response."""
    for key in ("tradingAccounts", "accounts"):
        accounts = data.get(key, [])
        if not accounts:
            continue

        # Select account
        if account_id:
            acct = next(
                (a for a in accounts if str(a.get("tradingAccountId", a.get("id"))) == account_id),
                None,
            )
        else:
            acct = accounts[0]

        if not acct:
            continue

        # Token: try per-account first, then top-level
        token = (
            acct.get("tradingApiToken")
            or (acct.get("tradingAccountToken", {}) or {}).get("token")
            or data.get("tradingApiToken")
        )
        # UUID: try nested offer.system.uuid first, then flat
        uuid = (
            acct.get("offer", {}).get("system", {}).get("uuid")
            or acct.get("systemUUID")
            or acct.get("id")
        )

        if token and uuid:
            return (token, uuid)

    return None


# ── Main test runner ────────────────────────────────────────────────────────


async def main() -> None:
    """Run multi-endpoint connectivity diagnostics."""
    load_dotenv()

    base_url = os.getenv("MATCHTRADER_API_URL", "").rstrip("/")
    email = os.getenv("MATCHTRADER_USERNAME", "")
    password = os.getenv("MATCHTRADER_PASSWORD", "")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = os.getenv("MATCHTRADER_ACCOUNT_ID")

    if not all([base_url, email, password]):
        logger.error("Missing required env vars (MATCHTRADER_API_URL, USERNAME, PASSWORD).")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("MatchTrader API — Multi-Endpoint Connectivity Diagnostics")
    logger.info("=" * 70)
    logger.info("Base URL:   {}", base_url)
    logger.info("Email:      {}", email)
    logger.info("Broker ID:  {}", broker_id)
    logger.info("Account ID: {}", account_id or "(auto-select)")
    logger.info("=" * 70)

    # Payloads for each endpoint format
    co_login_payload = {"email": email, "password": password, "brokerId": broker_id}
    mtr_login_payload = {"email": email, "password": password, "brokerId": broker_id}
    edge_login_payload = {"email": email, "password": password, "brokerId": broker_id}

    endpoints = [
        ("/manager/mtr-login", mtr_login_payload, "mtr-login"),
        ("/manager/co-login", co_login_payload, "co-login"),
        ("/mtr-core-edge/login", edge_login_payload, "edge-login"),
    ]

    headers = _make_headers(base_url)
    successful_data: dict | None = None
    successful_label: str = ""

    # ── Phase 1: Test all endpoints with httpx + full browser headers ───
    logger.info("")
    logger.info("── Phase 1: httpx with full browser headers ──")
    for path, payload, label in endpoints:
        data = await _test_endpoint_httpx(base_url, path, payload, f"httpx/{label}", headers)
        if data and not successful_data:
            accounts = data.get("tradingAccounts", data.get("accounts", []))
            if accounts:
                successful_data = data
                successful_label = f"httpx/{label}"
        logger.info("")

    # ── Phase 2: Test with curl_cffi (browser TLS fingerprint) ──────────
    if not successful_data:
        logger.info("── Phase 2: curl_cffi (browser TLS impersonation) ──")
        for path, payload, label in endpoints:
            data = await _test_endpoint_curl_cffi(base_url, path, payload, f"curl_cffi/{label}")
            if data and not successful_data:
                accounts = data.get("tradingAccounts", data.get("accounts", []))
                if accounts:
                    successful_data = data
                    successful_label = f"curl_cffi/{label}"
            logger.info("")

    # ── Phase 3: Analyze results ────────────────────────────────────────
    logger.info("=" * 70)
    if not successful_data:
        logger.error("ALL ENDPOINTS BLOCKED — no login succeeded")
        logger.info("")
        logger.info("Recommended next steps:")
        logger.info("  1. Install curl_cffi: uv add curl_cffi")
        logger.info("  2. Use Playwright browser to extract cf_clearance cookie")
        logger.info("  3. Check if E8 has a separate API domain (ask support)")
        sys.exit(1)

    logger.info("✅ LOGIN SUCCEEDED via: {}", successful_label)
    logger.info("")

    # Show response structure
    logger.info("Response top-level keys: {}", list(successful_data.keys()))

    # Extract tradingApiDomain
    alt_domain = _extract_trading_api_domain(successful_data)
    if alt_domain:
        logger.info("🔑 tradingApiDomain found: {}", alt_domain)
        logger.info("   This may be a Cloudflare-free API domain!")
    else:
        logger.info("ℹ No tradingApiDomain in response")

    # Extract token + UUID
    token_info = _extract_token_and_uuid(successful_data, account_id)
    if not token_info:
        logger.error("Could not extract token/UUID from response. Raw response:")
        logger.info("{}", json.dumps(successful_data, indent=2, default=str)[:3000])
        sys.exit(1)

    trading_token, system_uuid = token_info
    logger.info("Trading token (first 20): {}...", trading_token[:20])
    logger.info("System UUID: {}", system_uuid)

    # ── Phase 4: Test balance via curl_cffi ─────────────────────────────
    logger.info("")
    logger.info("── Phase 4: Test balance via curl_cffi ──")

    from curl_cffi.requests import AsyncSession as CurlSession

    balance_url = f"{base_url}/mtr-api/{system_uuid}/balance"
    auth_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Auth-Trading-Api": trading_token,
        "Auth-trading-api": trading_token,
    }

    logger.info("  GET {}", balance_url)
    try:
        async with CurlSession(impersonate="chrome", timeout=30) as curl_session:
            resp = await curl_session.get(balance_url, headers=auth_headers)
            if resp.status_code >= 400:
                logger.warning("  ❌ HTTP {}: {}", resp.status_code, resp.text[:500])
            else:
                balance_data = resp.json()
                logger.info("  ✅ BALANCE: {}", json.dumps(balance_data, indent=2))
    except Exception as e:
        logger.error("  ❌ ERROR: {}", e)

    # ── Phase 5: End-to-end via MatchTraderClient ───────────────────────
    logger.info("")
    logger.info("── Phase 5: End-to-end via MatchTraderClient (curl_cffi transport) ──")

    from src.execution.matchtrader_client import MatchTraderClient

    try:
        async with MatchTraderClient(
            base_url=base_url,
            email=email,
            password=password,
            broker_id=broker_id,
            account_id=account_id,
        ) as client:
            # Login
            logger.info("  [1/3] Logging in via MatchTraderClient...")
            tokens = await client.login()
            logger.info("  ✅ LOGIN OK — UUID: {}", tokens.system_uuid)

            # Balance
            logger.info("  [2/3] Getting balance...")
            bal = await client.get_balance()
            logger.info("  ✅ BALANCE: ${:.2f} (equity=${:.2f})", bal.balance, bal.equity)

            # Positions
            logger.info("  [3/3] Getting open positions...")
            positions = await client.get_open_positions()
            logger.info("  ✅ POSITIONS: {} open", len(positions))
            for pos in positions:
                logger.info(
                    "    {} {} {} @ {:.5f} | PnL: ${:.2f}",
                    pos.position_id,
                    pos.side,
                    pos.symbol,
                    pos.open_price,
                    pos.profit,
                )

        logger.info("")
        logger.info("🎉 END-TO-END TEST PASSED — MatchTrader API fully operational!")
    except Exception as e:
        logger.error("  ❌ MatchTraderClient FAILED: {}", e)
        import traceback

        traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("  Working endpoint: {}", successful_label)
    if alt_domain:
        logger.info("  Alt API domain: {}", alt_domain)
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
