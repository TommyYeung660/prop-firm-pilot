"""
Tests for src/execution/matchtrader_client.py.

Tests cover:
- Login flow extracts token and UUID from /manager/co-login response
- _ensure_auth() auto-refreshes if TOKEN_REFRESH_SECONDS passed
- RateLimiter functionality (daily limits, reset, counting)
- Context manager lifecycle
- API request error handling (401, 429, retries)
- Trading operations (open/close/modify positions)
- Account queries (balance, positions, instruments)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.broker_models import (
    BrokerBalanceInfo,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)
from src.execution.matchtrader_client import (
    AuthTokens,
    BalanceInfo,
    InstrumentInfo,
    MatchTraderAuthError,
    MatchTraderClient,
    MatchTraderError,
    MatchTraderRateLimitError,
    OrderResult,
    PositionInfo,
    QuoteInfo,
    RateLimiter,
)

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def client_config() -> dict:
    """Basic client configuration for testing."""
    return {
        "base_url": "https://mtr.e8markets.com",
        "email": "test@example.com",
        "password": "testpass",
        "broker_id": "2",
        "account_id": "950552",
    }


@pytest.fixture
def mock_response() -> AsyncMock:
    """Mock curl_cffi response object."""
    response = AsyncMock()
    response.status_code = 200
    response.text = "{}"
    response.json = MagicMock(return_value={})
    return response


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock AsyncSession.request for HTTP calls."""
    session = AsyncMock()
    session.request = AsyncMock()
    session.close = AsyncMock()
    return session


# ── RateLimiter Tests ───────────────────────────────────────────────────────


class TestRateLimiter:
    """Test daily rate limiting logic."""

    def test_initial_state(self) -> None:
        limiter = RateLimiter(daily_limit=2000)
        assert limiter.count == 0
        assert limiter.remaining == 2000
        assert limiter.can_proceed() is True

    def test_record_increments_count(self) -> None:
        limiter = RateLimiter(daily_limit=100)
        limiter.record()
        assert limiter.count == 1
        assert limiter.remaining == 99

    def test_multiple_records(self) -> None:
        limiter = RateLimiter(daily_limit=100)
        for _ in range(50):
            limiter.record()
        assert limiter.count == 50
        assert limiter.remaining == 50

    def test_can_proceed_with_reserve(self) -> None:
        limiter = RateLimiter(daily_limit=100)
        # Reserve default 50
        for _ in range(49):
            limiter.record(call_type="write")
        assert limiter.can_proceed() is True
        assert limiter.can_proceed(reserve=50) is True

    def test_can_proceed_at_reserve_boundary(self) -> None:
        limiter = RateLimiter(daily_limit=100)
        for _ in range(50):
            limiter.record(call_type="write")
        # 50 used, 50 remaining, reserve=50 -> exactly at boundary
        assert limiter.can_proceed(reserve=50) is False

    def test_can_proceed_custom_reserve(self) -> None:
        limiter = RateLimiter(daily_limit=100)
        for _ in range(90):
            limiter.record(call_type="write")
        assert limiter.can_proceed(reserve=5) is True
        assert limiter.can_proceed(reserve=15) is False

    def test_remaining_never_negative(self) -> None:
        limiter = RateLimiter(daily_limit=10)
        for _ in range(20):
            limiter.record()
        assert limiter.count == 20
        assert limiter.remaining >= 0  # Should be clamped to 0


# ── Client Lifecycle ─────────────────────────────────────────────────────────


class TestClientLifecycle:
    """Test context manager and initialization."""

    def test_initialization(self, client_config: dict) -> None:
        client = MatchTraderClient(**client_config)
        assert client._base_url == "https://mtr.e8markets.com"
        assert client._email == "test@example.com"
        assert client._account_id == "950552"
        assert client._tokens is None
        assert client.is_authenticated is False

    async def test_context_manager_enters(self, client_config: dict) -> None:
        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                assert client._session is not None
                mock_session_cls.assert_called_once()

    async def test_context_manager_exits(self, client_config: dict) -> None:
        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            client = MatchTraderClient(**client_config)
            await client.__aenter__()
            await client.__aexit__(None, None, None)
            mock_session.close.assert_awaited_once()

    async def test_system_uuid_requires_auth(self, client_config: dict) -> None:
        client = MatchTraderClient(**client_config)
        with pytest.raises(RuntimeError, match="Not authenticated"):
            _ = client.system_uuid


# ── Login Flow Tests ───────────────────────────────────────────────────────


class TestLoginFlow:
    """Test login authentication flow."""

    async def test_login_success_extracts_token_and_uuid(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Verify login extracts tradingApiToken and system UUID from response."""
        # Mock successful login response
        mock_response.json = MagicMock(
            return_value={
                "token": "refresh_token_value",
                "accounts": [
                    {
                        "tradingAccountId": "950552",
                        "tradingApiToken": "dummy_jwt_token",
                        "offer": {
                            "system": {
                                "uuid": "sys-123-abc",
                            }
                        },
                    }
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                tokens = await client.login()

                assert tokens.trading_api_token == "dummy_jwt_token"
                assert tokens.refresh_token == "refresh_token_value"
                assert tokens.system_uuid == "sys-123-abc"
                assert client.is_authenticated is True
                assert client.system_uuid == "sys-123-abc"

                # Verify the request was made to correct endpoint
                mock_session.request.assert_awaited_once_with(
                    "POST",
                    "https://mtr.e8markets.com/manager/co-login",
                    json={
                        "email": "test@example.com",
                        "password": "testpass",
                        "brokerId": "2",
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )

    async def test_login_with_legacy_response_format(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Test fallback to legacy systemUUID format."""
        mock_response.json = MagicMock(
            return_value={
                "token": "refresh_token_value",
                "accounts": [
                    {
                        "tradingAccountId": "950552",
                        "tradingApiToken": "dummy_jwt",
                        "systemUUID": "legacy-uuid-456",
                    }
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                tokens = await client.login()
                assert tokens.system_uuid == "legacy-uuid-456"

    async def test_login_selects_account_by_id(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Test account_id parameter selects correct account."""
        mock_response.json = MagicMock(
            return_value={
                "token": "refresh_token",
                "accounts": [
                    {
                        "tradingAccountId": "111111",
                        "tradingApiToken": "token_for_111",
                        "offer": {"system": {"uuid": "uuid-111"}},
                    },
                    {
                        "tradingAccountId": "950552",
                        "tradingApiToken": "token_for_950552",
                        "offer": {"system": {"uuid": "uuid-950552"}},
                    },
                    {
                        "tradingAccountId": "222222",
                        "tradingApiToken": "token_for_222",
                        "offer": {"system": {"uuid": "uuid-222"}},
                    },
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            config = client_config.copy()
            config["account_id"] = "950552"
            async with MatchTraderClient(**config) as client:
                tokens = await client.login()
                assert tokens.trading_api_token == "token_for_950552"
                assert tokens.system_uuid == "uuid-950552"

    async def test_login_uses_first_account_when_no_id(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Test first account is used when account_id is None."""
        mock_response.json = MagicMock(
            return_value={
                "token": "refresh_token",
                "accounts": [
                    {
                        "tradingAccountId": "999999",
                        "tradingApiToken": "first_token",
                        "offer": {"system": {"uuid": "first_uuid"}},
                    },
                    {
                        "tradingAccountId": "888888",
                        "tradingApiToken": "second_token",
                        "offer": {"system": {"uuid": "second_uuid"}},
                    },
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            config = client_config.copy()
            config["account_id"] = None
            async with MatchTraderClient(**config) as client:
                tokens = await client.login()
                assert tokens.trading_api_token == "first_token"
                assert tokens.system_uuid == "first_uuid"

    async def test_login_fails_no_accounts(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(return_value={"token": "rt", "accounts": []})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="No trading accounts found"):
                    await client.login()

    async def test_login_fails_account_not_found(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value={
                "token": "rt",
                "accounts": [
                    {
                        "tradingAccountId": "111111",
                        "tradingApiToken": "t1",
                        "offer": {"system": {"uuid": "u1"}},
                    }
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="Account 950552 not found"):
                    await client.login()

    async def test_login_fails_missing_system_uuid(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value={
                "token": "rt",
                "accounts": [{"tradingAccountId": "950552", "tradingApiToken": "jwt"}],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="Could not extract systemUUID"):
                    await client.login()

    async def test_login_fails_missing_token(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value={
                "accounts": [
                    {
                        "tradingAccountId": "950552",
                        "offer": {"system": {"uuid": "uuid"}},
                    }
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="Could not extract tradingApiToken"):
                    await client.login()


# ── Token Refresh Tests ─────────────────────────────────────────────────────


class TestTokenRefresh:
    """Test token refresh and auto-refresh logic."""

    async def test_ensure_auth_skips_when_fresh(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Test _ensure_auth does nothing when token is fresh."""
        # First login
        mock_response.json = MagicMock(
            return_value={
                "token": "refresh_token",
                "accounts": [
                    {
                        "tradingAccountId": "950552",
                        "tradingApiToken": "initial_jwt",
                        "offer": {"system": {"uuid": "sys-123"}},
                    }
                ],
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                await client.login()

                # Reset mock to track new requests
                mock_session.request.reset_mock()

                # _ensure_auth should not call refresh when token is fresh
                await client._ensure_auth()

                # No new HTTP request should have been made
                mock_session.request.assert_not_awaited()

    async def test_ensure_auth_refreshes_when_expired(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Test _ensure_auth refreshes token when TOKEN_REFRESH_SECONDS passed."""
        # Login response
        login_response_data = {
            "token": "refresh_token",
            "accounts": [
                {
                    "tradingAccountId": "950552",
                    "tradingApiToken": "initial_jwt",
                    "offer": {"system": {"uuid": "sys-123"}},
                }
            ],
        }

        # Refresh response
        refresh_response_data = {
            "token": "new_refresh_token",
            "accounts": [
                {
                    "tradingAccountId": "950552",
                    "tradingApiToken": "new_jwt_token",
                    "offer": {"system": {"uuid": "sys-123"}},
                }
            ],
        }

        call_count = 0

        def mock_json_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return login_response_data
            else:
                return refresh_response_data

        mock_response.json = MagicMock(side_effect=mock_json_side_effect)

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                # Use a short refresh time for testing
                client.TOKEN_REFRESH_SECONDS = 0.1

                await client.login()

                # Wait for token to expire
                import time

                time.sleep(0.15)

                # Reset mock to track new requests
                mock_session.request.reset_mock()

                # This should trigger refresh
                await client._ensure_auth()

                # Verify refresh endpoint was called
                calls = mock_session.request.call_args_list
                assert len(calls) == 1
                assert calls[0][0][0] == "POST"
                assert "/manager/co-login" in calls[0][0][1]

                # Verify token was updated
                assert client._tokens.trading_api_token == "new_jwt_token"
                assert client._tokens.refresh_token == "new_refresh_token"

    async def test_ensure_auth_raises_when_not_authenticated(self, client_config: dict) -> None:
        """Test _ensure_auth raises error before login."""
        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="Not authenticated"):
                    await client._ensure_auth()

    async def test_refresh_token_requires_existing_auth(self, client_config: dict) -> None:
        """Test refresh_token raises error when not authenticated."""
        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                with pytest.raises(RuntimeError, match="Cannot refresh"):
                    await client.refresh_token()

    async def test_refresh_login_uses_debug_log_level(self, client_config: dict) -> None:
        """Token refresh should log at DEBUG, not INFO (BUG #10)."""
        login_resp = {
            "token": "refresh_token",
            "accounts": [
                {
                    "tradingAccountId": "950552",
                    "offer": {"system": {"uuid": "sys-uuid"}},
                    "tradingApiToken": "jwt-token",
                }
            ],
        }

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            resp = AsyncMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value=login_resp)
            mock_session.request = AsyncMock(return_value=resp)
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                # Initial login should use INFO
                with patch("src.execution.matchtrader_client.logger") as mock_logger:
                    await client.login()
                    mock_logger.info.assert_any_call(
                        "MatchTrader: logging in as {}", "test@example.com"
                    )

                # Refresh uses login(_quiet=True) which should use DEBUG
                with patch("src.execution.matchtrader_client.logger") as mock_logger:
                    await client.login(_quiet=True)
                    mock_logger.debug.assert_any_call(
                        "MatchTrader: logging in as {}", "test@example.com"
                    )
                    # Ensure INFO was NOT called for the login message
                    for call in mock_logger.info.call_args_list:
                        assert "logging in as" not in str(call)


# ── Error Handling Tests ────────────────────────────────────────────────────


class TestErrorHandling:
    """Test HTTP error handling."""

    async def test_401_raises_auth_error(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                # Set up mock authentication
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                with pytest.raises(MatchTraderAuthError):
                    await client._raw_request("GET", "/test", authenticated=True)

    async def test_429_raises_rate_limit_error(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                with pytest.raises(MatchTraderRateLimitError):
                    await client._raw_request("GET", "/test", authenticated=True)

    async def test_500_raises_general_error(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                with pytest.raises(MatchTraderError, match="API error 500"):
                    await client._raw_request("GET", "/test", authenticated=True)


# ── Rate Limiting Integration Tests ─────────────────────────────────────────


class TestRateLimitingIntegration:
    """Test rate limiting in API requests."""

    async def test_raw_request_increments_rate_limiter(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(return_value={})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                initial_count = client._rate_limiter.count
                await client._raw_request("GET", "/test", authenticated=True)

                assert client._rate_limiter.count == initial_count + 1

    async def test_api_request_checks_rate_limit(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(return_value={})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                # Exhaust rate limit (minus reserve)
                for _ in range(1949):
                    client._rate_limiter.record(call_type="write")

                # 1949 used, 51 remaining (above reserve of 50)
                assert client._rate_limiter.can_proceed() is True

                # Make one more to hit reserve boundary
                client._rate_limiter.record(call_type="write")

                # 1950 used, 50 remaining (exactly at reserve - can't proceed)
                assert client._rate_limiter.can_proceed() is False

    async def test_rate_limit_exhausted_raises_error(self, client_config: dict) -> None:
        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(daily_request_limit=100, **client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                # Exhaust all requests
                for _ in range(100):
                    client._rate_limiter.record(call_type="write")

                with pytest.raises(MatchTraderRateLimitError, match="budget exhausted"):
                    await client._api_request("POST", "/mtr-api/sys-123/position/open")

    async def test_read_request_not_blocked_when_write_budget_exhausted(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        """Read/query endpoints should remain available even when write budget is exhausted."""
        mock_response.json = MagicMock(return_value={"ok": True})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(daily_request_limit=100, **client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                # Exhaust write budget
                for _ in range(100):
                    client._rate_limiter.record(call_type="write")

                # Should still allow read endpoint
                response = await client._api_request("GET", "/mtr-api/sys-123/balance")
                assert response is not None


# ── Trading Operations Tests ───────────────────────────────────────────────


class TestTradingOperations:
    """Test position trading operations."""

    async def test_get_balance_returns_balance_info(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value={
                "balance": 5000.0,
                "equity": 5100.0,
                "margin": 100.0,
                "freeMargin": 4900.0,
                "currency": "USD",
            }
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                balance = await client.get_balance()

                assert isinstance(balance, BalanceInfo)
                assert isinstance(balance, BrokerBalanceInfo)
                assert balance.balance == 5000.0
                assert balance.equity == 5100.0
                assert balance.margin == 100.0
                assert balance.free_margin == 4900.0

    async def test_get_open_positions_returns_list(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value=[
                {
                    "positionId": "pos-001",
                    "symbol": "EURUSD.",
                    "side": "BUY",
                    "volume": 0.1,
                    "openPrice": 1.0850,
                    "currentPrice": 1.0900,
                    "profit": 50.0,
                },
                {
                    "positionId": "pos-002",
                    "symbol": "GBPUSD.",
                    "side": "SELL",
                    "volume": 0.05,
                    "openPrice": 1.2700,
                    "currentPrice": 1.2650,
                    "profit": 25.0,
                },
            ]
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()
                # Set recent auth time to prevent auto-refresh
                client._last_auth_time = 9999999999.0

                positions = await client.get_open_positions()

                assert len(positions) == 2
                assert isinstance(positions[0], BrokerPositionInfo)
                assert positions[0].position_id == "pos-001"
                assert positions[0].symbol == "EURUSD."
                assert positions[0].side == "BUY"
                assert positions[0].profit == 50.0
                assert positions[1].position_id == "pos-002"
                assert positions[1].symbol == "GBPUSD."
                assert positions[1].side == "SELL"

    async def test_open_position_returns_order_result(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(return_value={"orderId": "new-order-123"})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                result = await client.open_position(
                    symbol="EURUSD.", side="BUY", volume=0.1, sl=1.080, tp=1.090
                )

                assert result.success is True
                assert isinstance(result, BrokerOrderResult)
                assert result.position_id == "new-order-123"
                assert "opened successfully" in result.message

                # Verify request body
                call_args = mock_session.request.call_args
                body = call_args[1]["json"]
                assert body["instrument"] == "EURUSD."
                assert body["orderSide"] == "BUY"
                assert body["volume"] == 0.1
                assert body["slPrice"] == 1.080
                assert body["tpPrice"] == 1.090

    async def test_open_position_handles_error(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.status_code = 400
        mock_response.text = "Insufficient margin"

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()
                # Set recent auth time to prevent auto-refresh
                client._last_auth_time = 9999999999.0

                result = await client.open_position(symbol="EURUSD.", side="BUY", volume=10.0)

                assert result.success is False
                assert "API error" in result.message

    async def test_close_position_returns_success(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(return_value={"result": "closed"})

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                result = await client.close_position(
                    position_id="pos-001", symbol="EURUSD.", side="BUY", volume=0.1
                )

                assert result.success is True
                assert result.position_id == "pos-001"

    async def test_modify_position_returns_success(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        modify_ok = {"status": "OK", "errorMessage": "", "result": "modified"}
        mock_response.json = MagicMock(return_value=modify_ok)

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                result = await client.modify_position(
                    position_id="pos-001",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.10,
                    sl=1.080,
                    tp=1.090,
                )

                assert result.success is True
                assert result.position_id == "pos-001"

    async def test_modify_position_only_sl(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        modify_ok = {"status": "OK", "errorMessage": "", "result": "modified"}
        mock_response.json = MagicMock(return_value=modify_ok)

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()

                result = await client.modify_position(
                    position_id="pos-001",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.10,
                    sl=1.080,
                )

                assert result.success is True

                # Verify only slPrice was sent
                call_args = mock_session.request.call_args
                body = call_args[1]["json"]
                assert "slPrice" in body
                assert body["slPrice"] == 1.080
                assert "tpPrice" not in body

    async def test_verify_sl_tp_accepts_broker_precision_normalization(
        self, client_config: dict
    ) -> None:
        """Rounded broker prices should verify successfully at declared precision."""
        client = MatchTraderClient(**client_config)
        client.get_open_positions = AsyncMock(
            return_value=[
                MagicMock(
                    position_id="pos-001",
                    sl_price=0.57961,
                    tp_price=0.56644,
                )
            ]
        )

        verified = await client.verify_sl_tp(
            position_id="pos-001",
            expected_sl=0.5796104426501542,
            expected_tp=0.5664437344099075,
            price_precision=5,
        )

        assert verified is True


# ── Instrument Info Tests ──────────────────────────────────────────────────


class TestInstrumentInfo:
    """Test instrument information queries."""

    async def test_get_effective_instruments(
        self, client_config: dict, mock_response: AsyncMock
    ) -> None:
        mock_response.json = MagicMock(
            return_value=[
                {
                    "symbol": "EURUSD.",
                    "alias": "EURUSD",
                    "description": "Euro vs US Dollar",
                    "type": "CASH",
                    "volumeMin": 0.01,
                    "volumeMax": 50.0,
                    "volumeStep": 0.01,
                    "contractSize": 100000,
                    "pricePrecision": 5,
                    "sessionOpen": True,
                },
                {
                    "symbol": "GBPUSD.",
                    "alias": "GBPUSD",
                    "description": "British Pound vs US Dollar",
                    "type": "CASH",
                    "volumeMin": 0.01,
                    "volumeMax": 50.0,
                    "volumeStep": 0.01,
                    "contractSize": 100000,
                    "pricePrecision": 5,
                    "sessionOpen": True,
                },
            ]
        )

        with patch("src.execution.matchtrader_client.AsyncSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            async with MatchTraderClient(**client_config) as client:
                client._tokens = AuthTokens(
                    trading_api_token="jwt",
                    refresh_token="rt",
                    system_uuid="sys-123",
                )
                client._last_auth_time = __import__("time").monotonic()
                # Set recent auth time to prevent auto-refresh
                client._last_auth_time = 9999999999.0

                instruments = await client.get_effective_instruments()

                assert len(instruments) == 2
                assert isinstance(instruments[0], BrokerInstrumentInfo)
                assert instruments[0].symbol == "EURUSD."
                assert instruments[0].volume_min == 0.01
                assert instruments[0].contract_size == 100000
                assert instruments[1].symbol == "GBPUSD."
                assert instruments[1].price_precision == 5


# ── Broker Model Compatibility Tests ───────────────────────────────────────


class TestBrokerModelCompatibility:
    """Ensure MatchTrader client/model exports satisfy broker-neutral shapes."""

    def test_matchtrader_model_aliases_to_broker_models(self) -> None:
        assert BalanceInfo is BrokerBalanceInfo
        assert PositionInfo is BrokerPositionInfo
        assert OrderResult is BrokerOrderResult
        assert QuoteInfo is BrokerQuoteInfo
        assert InstrumentInfo is BrokerInstrumentInfo


# ── QuoteInfo Model Tests ───────────────────────────────────────────────────


class TestQuoteInfo:
    """Tests for QuoteInfo model parsing from MatchTrader Market Watch API."""

    def test_parse_from_api_response(self) -> None:
        """Should parse quote data from typical API response dict."""
        quote = QuoteInfo(
            symbol="EURUSD.",
            bid=float("1.10873"),
            ask=float("1.10877"),
            high=float("1.10901"),
            low=float("1.10775"),
            timestampMs=1726200032067,
        )
        assert quote.symbol == "EURUSD."
        assert quote.bid == 1.10873
        assert quote.ask == 1.10877
        assert quote.high == 1.10901
        assert quote.low == 1.10775
        assert quote.timestamp_ms == 1726200032067

    def test_default_values(self) -> None:
        """Should have sensible defaults for all fields."""
        quote = QuoteInfo()
        assert quote.symbol == ""
        assert quote.bid == 0.0
        assert quote.ask == 0.0
        assert quote.timestamp_ms == 0

    def test_populate_by_name(self) -> None:
        """Should accept both alias and field name."""
        # Using alias
        q1 = QuoteInfo(timestampMs=12345)
        assert q1.timestamp_ms == 12345

        # Using field name
        q2 = QuoteInfo(timestamp_ms=67890)
        assert q2.timestamp_ms == 67890
