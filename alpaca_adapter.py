"""
Alpaca Market Data Adapter
Real-time WebSocket integration with Alpaca Markets

Supports:
- US Stocks (IEX, SIP feeds)
- Cryptocurrencies
- Real-time quotes, trades, bars
- Paper trading and live trading
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
from loguru import logger

from .base_adapter import (
    BaseMarketDataAdapter,
    Ticker,
    OrderBook,
    Trade,
    OHLCV,
    OrderSide
)


class AlpacaAdapter(BaseMarketDataAdapter):
    """
    Alpaca WebSocket Market Data Adapter
    
    Documentation: https://alpaca.markets/docs/api-references/market-data-api/
    """
    
    # WebSocket endpoints
    WS_STOCKS_URL = "wss://stream.data.alpaca.markets/v2/iex"  # IEX feed (free)
    WS_STOCKS_SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"  # SIP feed (paid)
    WS_CRYPTO_URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
    
    WS_PAPER_URL = "wss://paper-api.alpaca.markets/stream"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        asset_type: str = 'stocks',  # 'stocks' or 'crypto'
        feed: str = 'iex',  # 'iex' or 'sip' for stocks
        paper_trading: bool = True,
        enable_audit: bool = True
    ):
        super().__init__(
            exchange_name="ALPACA",
            api_key=api_key,
            api_secret=api_secret,
            testnet=paper_trading,
            enable_audit=enable_audit
        )
        
        self.asset_type = asset_type
        self.feed = feed
        
        # Select WebSocket URL based on asset type and feed
        if asset_type == 'crypto':
            self.ws_url = self.WS_CRYPTO_URL
        elif asset_type == 'stocks':
            self.ws_url = self.WS_STOCKS_SIP_URL if feed == 'sip' else self.WS_STOCKS_URL
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}")
        
        self.authenticated = False
        self.subscriptions: Dict[str, List[str]] = {
            'quotes': [],
            'trades': [],
            'bars': []
        }
    
    async def connect(self) -> bool:
        """Establish WebSocket connection and authenticate"""
        try:
            logger.info(f"Connecting to Alpaca {self.asset_type} WebSocket: {self.ws_url}")
            
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            
            await self.websocket.send(json.dumps(auth_msg))
            
            # Wait for auth response
            response = json.loads(await self.websocket.recv())
            
            if response[0].get('T') == 'success' and response[0].get('msg') == 'authenticated':
                self.authenticated = True
                logger.info("✓ Authenticated with Alpaca")
                
                # Start message loop
                asyncio.create_task(self._message_loop())
                
                # Re-subscribe to streams if reconnecting
                if any(self.subscriptions.values()):
                    await self._resubscribe()
                
                return True
            else:
                logger.error(f"Authentication failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        self.authenticated = False
        logger.info("Disconnected from Alpaca")
    
    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """Subscribe to quote (ticker) stream"""
        await self._subscribe('quotes', symbols)
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20) -> None:
        """
        Alpaca doesn't provide full order book via WebSocket
        Using best bid/ask from quotes instead
        """
        logger.warning("Alpaca doesn't provide order book depth. Using quotes for BBO.")
        await self.subscribe_ticker(symbols)
    
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to trade stream"""
        await self._subscribe('trades', symbols)
    
    async def subscribe_ohlcv(self, symbols: List[str], timeframe: str = '1Min') -> None:
        """
        Subscribe to bar (OHLCV) stream
        
        Timeframes: 1Min, 5Min, 15Min, 1Hour, 1Day
        """
        await self._subscribe('bars', symbols)
    
    async def _subscribe(self, stream_type: str, symbols: List[str]) -> None:
        """Send subscription message"""
        if not self.authenticated:
            logger.warning("Not authenticated. Connect first.")
            return
        
        # Add to subscription tracking
        for symbol in symbols:
            if symbol not in self.subscriptions[stream_type]:
                self.subscriptions[stream_type].append(symbol)
        
        # Send subscription message
        subscribe_msg = {
            "action": "subscribe",
            stream_type: symbols
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {stream_type}: {symbols}")
    
    async def _resubscribe(self) -> None:
        """Resubscribe to all streams after reconnection"""
        for stream_type, symbols in self.subscriptions.items():
            if symbols:
                await self._subscribe(stream_type, symbols)
    
    async def _message_loop(self) -> None:
        """Main message processing loop"""
        try:
            async for message in self.websocket:
                messages = json.loads(message)
                
                # Messages come as array
                for msg in messages:
                    await self._handle_message(msg)
                    self._update_health()
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Alpaca WebSocket connection closed")
            self.connected = False
            await self.reconnect()
        except Exception as e:
            self._handle_error(e)
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Route messages based on type"""
        msg_type = message.get('T', '')
        
        try:
            if msg_type == 'q':  # Quote
                await self._handle_quote(message)
            elif msg_type == 't':  # Trade
                await self._handle_trade_msg(message)
            elif msg_type == 'b':  # Bar (OHLCV)
                await self._handle_bar(message)
            elif msg_type == 'success':
                logger.debug(f"Success: {message.get('msg')}")
            elif msg_type == 'error':
                logger.error(f"Error from Alpaca: {message.get('msg')}")
            elif msg_type == 'subscription':
                logger.info(f"Subscription confirmed: {message}")
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._handle_error(e)
    
    async def _handle_quote(self, data: Dict[str, Any]) -> None:
        """
        Parse quote message and emit as ticker
        
        Quote format:
        {
            "T": "q",
            "S": "AAPL",
            "bx": "Q",  # Bid exchange
            "bp": 150.12,  # Bid price
            "bs": 100,  # Bid size
            "ax": "Q",  # Ask exchange
            "ap": 150.15,  # Ask price
            "as": 100,  # Ask size
            "t": "2024-02-09T10:30:00.123456Z"
        }
        """
        # Calculate mid price as "last"
        bid = float(data.get('bp', 0))
        ask = float(data.get('ap', 0))
        last = (bid + ask) / 2 if bid and ask else 0
        
        ticker = Ticker(
            symbol=data['S'],
            exchange=self.exchange_name,
            timestamp=self._parse_timestamp(data['t']),
            bid=bid,
            ask=ask,
            last=last,
            volume_24h=0,  # Not available in real-time quote
            high_24h=0,
            low_24h=0,
            open_24h=0,
            change_24h=0,
            change_24h_pct=0,
            latency_ms=self._calculate_latency()
        )
        
        await self._emit_ticker(ticker)
    
    async def _handle_trade_msg(self, data: Dict[str, Any]) -> None:
        """
        Parse trade message
        
        Trade format:
        {
            "T": "t",
            "S": "AAPL",
            "i": 123456,  # Trade ID
            "x": "Q",  # Exchange
            "p": 150.13,  # Price
            "s": 100,  # Size
            "t": "2024-02-09T10:30:00.123456Z",
            "c": ["@", "I"]  # Conditions
        }
        """
        trade = Trade(
            symbol=data['S'],
            exchange=self.exchange_name,
            timestamp=self._parse_timestamp(data['t']),
            trade_id=str(data['i']),
            price=float(data['p']),
            quantity=float(data['s']),
            side=OrderSide.BUY,  # Alpaca doesn't specify side
            audit_id=f"{self.exchange_name}_{data['S']}_{data['i']}"
        )
        
        await self._emit_trade(trade)
    
    async def _handle_bar(self, data: Dict[str, Any]) -> None:
        """
        Parse bar (OHLCV) message
        
        Bar format:
        {
            "T": "b",
            "S": "AAPL",
            "o": 150.10,  # Open
            "h": 150.20,  # High
            "l": 150.00,  # Low
            "c": 150.15,  # Close
            "v": 10000,  # Volume
            "t": "2024-02-09T10:30:00Z",
            "n": 100,  # Number of trades
            "vw": 150.12  # VWAP
        }
        """
        ohlcv = OHLCV(
            symbol=data['S'],
            exchange=self.exchange_name,
            timestamp=self._parse_timestamp(data['t']),
            timeframe='1Min',  # Alpaca sends 1-minute bars
            open=float(data['o']),
            high=float(data['h']),
            low=float(data['l']),
            close=float(data['c']),
            volume=float(data['v']),
            trades_count=data.get('n'),
            vwap=float(data.get('vw', 0)),
            audit_id=f"{self.exchange_name}_{data['S']}_{data['t']}"
        )
        
        await self._emit_ohlcv(ohlcv)
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Alpaca timestamp format"""
        # Remove 'Z' and parse
        timestamp_str = timestamp_str.rstrip('Z')
        try:
            return datetime.fromisoformat(timestamp_str)
        except:
            # Fallback to current time if parsing fails
            return datetime.utcnow()


# Example usage
async def main():
    """Example: Connect to Alpaca and stream market data"""
    
    # Initialize adapter (replace with your API keys)
    adapter = AlpacaAdapter(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_SECRET_KEY",
        asset_type='stocks',
        feed='iex',
        paper_trading=True
    )
    
    # Define callbacks
    def on_ticker_update(ticker: Ticker):
        print(f"QUOTE: {ticker.symbol} | Bid: ${ticker.bid:.2f} | Ask: ${ticker.ask:.2f}")
    
    async def on_trade_update(trade: Trade):
        print(f"TRADE: {trade.symbol} | {trade.quantity} @ ${trade.price:.2f}")
    
    async def on_bar_update(ohlcv: OHLCV):
        print(f"BAR: {ohlcv.symbol} | O: ${ohlcv.open:.2f} | C: ${ohlcv.close:.2f} | V: {ohlcv.volume}")
    
    # Register callbacks
    adapter.on_ticker(on_ticker_update)
    adapter.on_trade(on_trade_update)
    adapter.on_ohlcv(on_bar_update)
    
    # Subscribe to streams
    symbols = ['AAPL', 'TSLA', 'SPY']
    await adapter.subscribe_ticker(symbols)
    await adapter.subscribe_trades(symbols)
    await adapter.subscribe_ohlcv(symbols)
    
    # Connect
    if await adapter.connect():
        print("✓ Connected to Alpaca")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            status = adapter.get_status()
            print(f"Status: Connected={status.connected}, Messages={adapter.message_count}")
    else:
        print("✗ Failed to connect")


if __name__ == "__main__":
    asyncio.run(main())
