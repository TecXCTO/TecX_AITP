"""
Binance Market Data Adapter
Real-time WebSocket integration with Binance exchange

Features:
- WebSocket streaming for tickers, order books, trades
- Automatic reconnection with exponential backoff
- Data normalization to platform standards
- Latency monitoring
- SOC-2 compliant audit logging
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


class BinanceAdapter(BaseMarketDataAdapter):
    """
    Binance WebSocket Market Data Adapter
    
    Documentation: https://binance-docs.github.io/apidocs/spot/en/
    """
    
    # WebSocket endpoints
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    WS_TESTNET_URL = "wss://testnet.binance.vision/ws"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        enable_audit: bool = True
    ):
        super().__init__(
            exchange_name="BINANCE",
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            enable_audit=enable_audit
        )
        
        self.ws_url = self.WS_TESTNET_URL if testnet else self.WS_BASE_URL
        self.subscriptions: List[str] = []
        self.stream_names: List[str] = []
        
        # Binance-specific state
        self.orderbook_snapshots: Dict[str, OrderBook] = {}
        self.last_update_ids: Dict[str, int] = {}
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Binance
        
        Returns:
            bool: Connection success status
        """
        try:
            # Build combined stream URL
            if not self.stream_names:
                logger.warning("No streams to subscribe. Call subscribe_* methods first.")
                return False
            
            # Binance combined stream format: /stream?streams=stream1/stream2/stream3
            streams = "/".join(self.stream_names)
            url = f"{self.ws_url}/{streams}"
            
            logger.info(f"Connecting to Binance WebSocket: {url}")
            
            self.websocket = await websockets.connect(url)
            self.connected = True
            
            # Start message handler
            asyncio.create_task(self._message_loop())
            
            logger.info("✓ Connected to Binance WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        logger.info("Disconnected from Binance")
    
    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """
        Subscribe to 24hr ticker statistics
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        for symbol in symbols:
            stream = f"{symbol.lower()}@ticker"
            if stream not in self.stream_names:
                self.stream_names.append(stream)
                logger.info(f"Subscribed to ticker: {symbol}")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20) -> None:
        """
        Subscribe to order book depth updates
        
        Args:
            symbols: List of trading pairs
            depth: Order book depth (5, 10, 20)
        """
        valid_depths = [5, 10, 20]
        if depth not in valid_depths:
            depth = 20
            logger.warning(f"Invalid depth, using {depth}")
        
        for symbol in symbols:
            # Depth snapshot stream
            stream = f"{symbol.lower()}@depth{depth}"
            if stream not in self.stream_names:
                self.stream_names.append(stream)
                logger.info(f"Subscribed to orderbook: {symbol} (depth={depth})")
    
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time trade stream
        
        Args:
            symbols: List of trading pairs
        """
        for symbol in symbols:
            stream = f"{symbol.lower()}@trade"
            if stream not in self.stream_names:
                self.stream_names.append(stream)
                logger.info(f"Subscribed to trades: {symbol}")
    
    async def subscribe_ohlcv(self, symbols: List[str], timeframe: str = '1m') -> None:
        """
        Subscribe to candlestick/kline updates
        
        Args:
            symbols: List of trading pairs
            timeframe: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        """
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if timeframe not in valid_timeframes:
            timeframe = '1m'
            logger.warning(f"Invalid timeframe, using {timeframe}")
        
        for symbol in symbols:
            stream = f"{symbol.lower()}@kline_{timeframe}"
            if stream not in self.stream_names:
                self.stream_names.append(stream)
                logger.info(f"Subscribed to OHLCV: {symbol} ({timeframe})")
    
    async def _message_loop(self) -> None:
        """Main message processing loop"""
        try:
            async for message in self.websocket:
                await self._handle_message(json.loads(message))
                self._update_health()
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed")
            self.connected = False
            await self.reconnect()
        except Exception as e:
            self._handle_error(e)
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Parse and route incoming WebSocket messages
        
        Message format: {"stream": "btcusdt@ticker", "data": {...}}
        """
        if 'stream' not in message or 'data' not in message:
            return
        
        stream = message['stream']
        data = message['data']
        event_type = data.get('e', '')
        
        try:
            # Route to appropriate handler
            if event_type == '24hrTicker':
                await self._handle_ticker(data)
            elif event_type == 'depthUpdate' or 'depth' in stream:
                await self._handle_orderbook(data, stream)
            elif event_type == 'trade':
                await self._handle_trade(data)
            elif event_type == 'kline':
                await self._handle_kline(data)
            else:
                logger.debug(f"Unknown message type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._handle_error(e)
    
    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """Parse and emit ticker update"""
        ticker = Ticker(
            symbol=data['s'],
            exchange=self.exchange_name,
            timestamp=datetime.fromtimestamp(data['E'] / 1000),
            bid=float(data['b']),
            ask=float(data['a']),
            last=float(data['c']),
            volume_24h=float(data['v']),
            high_24h=float(data['h']),
            low_24h=float(data['l']),
            open_24h=float(data['o']),
            change_24h=float(data['p']),
            change_24h_pct=float(data['P']),
            latency_ms=self._calculate_latency()
        )
        
        await self._emit_ticker(ticker)
    
    async def _handle_orderbook(self, data: Dict[str, Any], stream: str) -> None:
        """Parse and emit order book update"""
        # Extract symbol from stream name
        symbol = stream.split('@')[0].upper()
        
        # Parse bids and asks
        bids = [(float(price), float(qty)) for price, qty in data.get('bids', [])]
        asks = [(float(price), float(qty)) for price, qty in data.get('asks', [])]
        
        # Handle both snapshot and update messages
        timestamp_key = 'E' if 'E' in data else 'lastUpdateId'
        timestamp = datetime.fromtimestamp(data.get('E', 0) / 1000) if 'E' in data else datetime.utcnow()
        
        orderbook = OrderBook(
            symbol=symbol,
            exchange=self.exchange_name,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=data.get('lastUpdateId', data.get('u')),
            audit_id=f"{self.exchange_name}_{symbol}_{data.get('lastUpdateId', 0)}"
        )
        
        # Validate order book integrity
        if self._validate_orderbook(orderbook):
            await self._emit_orderbook(orderbook)
        else:
            logger.warning(f"Invalid orderbook for {symbol}, skipping")
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Parse and emit trade update"""
        trade = Trade(
            symbol=data['s'],
            exchange=self.exchange_name,
            timestamp=datetime.fromtimestamp(data['T'] / 1000),
            trade_id=str(data['t']),
            price=float(data['p']),
            quantity=float(data['q']),
            side=OrderSide.BUY if not data['m'] else OrderSide.SELL,  # m = is buyer maker
            is_buyer_maker=data['m'],
            audit_id=f"{self.exchange_name}_{data['s']}_{data['t']}"
        )
        
        await self._emit_trade(trade)
    
    async def _handle_kline(self, data: Dict[str, Any]) -> None:
        """Parse and emit OHLCV/candlestick update"""
        k = data['k']  # Kline data nested in 'k'
        
        # Only emit on candle close
        if not k['x']:  # x = is candle closed
            return
        
        ohlcv = OHLCV(
            symbol=data['s'],
            exchange=self.exchange_name,
            timestamp=datetime.fromtimestamp(k['t'] / 1000),
            timeframe=k['i'],
            open=float(k['o']),
            high=float(k['h']),
            low=float(k['l']),
            close=float(k['c']),
            volume=float(k['v']),
            trades_count=k['n'],
            audit_id=f"{self.exchange_name}_{data['s']}_{k['t']}"
        )
        
        await self._emit_ohlcv(ohlcv)
    
    def _validate_orderbook(self, orderbook: OrderBook) -> bool:
        """
        Validate order book data quality
        
        Checks:
        - Bids and asks not empty
        - Prices are sequential
        - No negative prices or quantities
        """
        if not orderbook.bids or not orderbook.asks:
            return False
        
        # Check bid prices are descending
        for i in range(len(orderbook.bids) - 1):
            if orderbook.bids[i][0] <= orderbook.bids[i + 1][0]:
                return False
        
        # Check ask prices are ascending
        for i in range(len(orderbook.asks) - 1):
            if orderbook.asks[i][0] >= orderbook.asks[i + 1][0]:
                return False
        
        # Check spread is positive
        best_bid = orderbook.bids[0][0]
        best_ask = orderbook.asks[0][0]
        if best_bid >= best_ask:
            return False
        
        return True


# Example usage
async def main():
    """Example: Connect to Binance and stream market data"""
    
    # Initialize adapter
    adapter = BinanceAdapter(testnet=False)
    
    # Define callbacks
    def on_ticker_update(ticker: Ticker):
        print(f"TICKER: {ticker.symbol} | Last: ${ticker.last:.2f} | 24h Change: {ticker.change_24h_pct:.2f}%")
    
    async def on_trade_update(trade: Trade):
        print(f"TRADE: {trade.symbol} | {trade.side.value} {trade.quantity} @ ${trade.price}")
    
    def on_orderbook_update(orderbook: OrderBook):
        best_bid = orderbook.bids[0] if orderbook.bids else (0, 0)
        best_ask = orderbook.asks[0] if orderbook.asks else (0, 0)
        spread = best_ask[0] - best_bid[0]
        print(f"BOOK: {orderbook.symbol} | Bid: ${best_bid[0]:.2f} | Ask: ${best_ask[0]:.2f} | Spread: ${spread:.2f}")
    
    # Register callbacks
    adapter.on_ticker(on_ticker_update)
    adapter.on_trade(on_trade_update)
    adapter.on_orderbook(on_orderbook_update)
    
    # Subscribe to streams
    symbols = ['BTCUSDT', 'ETHUSDT']
    await adapter.subscribe_ticker(symbols)
    await adapter.subscribe_trades(symbols)
    await adapter.subscribe_orderbook(symbols, depth=10)
    
    # Connect
    if await adapter.connect():
        print("✓ Connected to Binance")
        
        # Monitor health
        while True:
            await asyncio.sleep(10)
            status = adapter.get_status()
            print(f"Status: Connected={status.connected}, Messages={adapter.message_count}, Latency={status.latency_ms:.2f}ms")
    else:
        print("✗ Failed to connect")


if __name__ == "__main__":
    asyncio.run(main())
