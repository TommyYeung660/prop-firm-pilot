import asyncio
from unittest.mock import AsyncMock, patch
from src.execution.matchtrader_client import MatchTraderClient

async def main():
    with patch('src.execution.matchtrader_client.AsyncSession') as mock_session:
        mock_instance = AsyncMock()
        mock_session.return_value = mock_instance
        
        # mock login response
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "token": "tok1",
            "systemUuid": "sys1",
            "refreshToken": "ref1"
        }
        mock_instance.request.return_value = mock_resp
        
        async with MatchTraderClient("http://test", "a@b.com", "pass") as client:
            await client.login()
            print("Login:", client._tokens)
            
            # mock get balance
            mock_balance_resp = AsyncMock()
            mock_balance_resp.status_code = 200
            mock_balance_resp.json.return_value = {"balance": 5000, "equity": 5050, "margin": 100, "freeMargin": 4900, "marginLevel": 100}
            mock_instance.request.return_value = mock_balance_resp
            
            balance = await client.get_balance()
            print("Balance:", balance)

if __name__ == '__main__':
    asyncio.run(main())
