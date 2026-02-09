"""
Base Market Data Adapter Interface
Defines the contract for all exchange adapters

SOC-2 Compliant | Regulatory-by-Design
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
from loguru import logger


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


@dataclass
class Ticker:
    """Normalized ticker data"""
    symbol: str
    exchange: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    open_24h: float
    change_24h: float
    change_24h_pct: float
    
    # Compliance metadata
    data_quality_score: float = 1.0  # 0-1 score
    latency_ms: Optional[float] = None
    audit_id: Optional[str] = None


@dataclass
class OrderBook:
    """Normalized order book data"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[tuple[float, float]]  # [(price, quantity), ...]
    asks: List[tuple[float, float]]
    sequence: Optional[int] = None  # For detecting gaps
    
    # Compliance
    checksum: Optional[str] = None
    audit_id: Optional[str] = None


@dataclass
class Trade:
    """Normalized trade data"""
    symbol: str
    exchange: str
    timestamp: datetime
    trade_id: str
    price: float
    quantity: float
    side: OrderSide
    
    # Compliance
    is_buyer_maker: bool = False
    audit_id: Optional[str] = None


@dataclass
class OHLCV:
    """OHLCV candlestick data"""
    symbol: str
    exchange: str
    timestamp: datetime
    timeframe: str  # '1m', '5m', '1h', etc.
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Additional metrics
    trades_count: Optional[int] = None
    vwap: Optional[float] = None  # Volume-weighted average price
    audit_id: Optional[str] = None


@dataclass
class ConnectionStatus:
    """WebSocket connection status"""
    exchange: str
    connected: bool
    timestamp: datetime
    reconnect_count: int = 0
    last_message_time: Optional[datetime] = None
    latency_ms: Optional[float] = None


class BaseMarketDataAdapter(ABC):
    """
    Abstract base class for all market data adapters
    
    Compliance Features:
    - Audit logging for all data ingestion
    - Data quality validation
    - Latency monitoring
    - Connection health tracking
    - Error handling with circuit breakers
    """
    
    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        enable_audit: bool = True
    ):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.enable_audit = enable_audit
        
        # Connection state
        self.connected = False
        self.websocket = None
        self.reconnect_count = 0
        
        # Callbacks
        self.ticker_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.ohlcv_callbacks: List[Callable] = []
        
        # Monitoring
        self.last_message_time: Optional[datetime] = None
        self.message_count = 0
        self.error_count = 0
        
        # Circuit breaker
        self.max_errors = 10
        self.circuit_open = False
        
        logger.info(f"Initializing {exchange_name} adapter (testnet={testnet})")
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to exchange
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully"""
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """Subscribe to ticker updates for given symbols"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20) -> None:
        """Subscribe to order book updates"""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to trade stream"""
        pass
    
    @abstractmethod
    async def subscribe_ohlcv(self, symbols: List[str], timeframe: str = '1m') -> None:
        """Subscribe to OHLCV/candlestick updates"""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message"""
        pass
    
    # ==================== Callback Management ====================
    
    def on_ticker(self, callback: Callable[[Ticker], None]) -> None:
        """Register ticker update callback"""
        self.ticker_callbacks.append(callback)
        logger.debug(f"Registered ticker callback: {callback.__name__}")
    
    def on_orderbook(self, callback: Callable[[OrderBook], None]) -> None:
        """Register order book update callback"""
        self.orderbook_callbacks.append(callback)
        logger.debug(f"Registered orderbook callback: {callback.__name__}")
    
    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register trade update callback"""
        self.trade_callbacks.append(callback)
        logger.debug(f"Registered trade callback: {callback.__name__}")
    
    def on_ohlcv(self, callback: Callable[[OHLCV], None]) -> None:
        """Register OHLCV update callback"""
        self.ohlcv_callbacks.append(callback)
        logger.debug(f"Registered OHLCV callback: {callback.__name__}")
    
    # ==================== Event Dispatching ====================
    
    async def _emit_ticker(self, ticker: Ticker) -> None:
        """Emit ticker update to all registered callbacks"""
        if self.enable_audit:
            self._audit_log("TICKER", ticker)
        
        for callback in self.ticker_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ticker)
                else:
                    callback(ticker)
            except Exception as e:
                logger.error(f"Error in ticker callback: {e}")
    
    async def _emit_orderbook(self, orderbook: OrderBook) -> None:
        """Emit order book update to all registered callbacks"""
        if self.enable_audit:
            self._audit_log("ORDERBOOK", orderbook)
        
        for callback in self.orderbook_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(orderbook)
                else:
                    callback(orderbook)
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
    
    async def _emit_trade(self, trade: Trade) -> None:
        """Emit trade update to all registered callbacks"""
        if self.enable_audit:
            self._audit_log("TRADE", trade)
        
        for callback in self.trade_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trade)
                else:
                    callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    async def _emit_ohlcv(self, ohlcv: OHLCV) -> None:
        """Emit OHLCV update to all registered callbacks"""
        if self.enable_audit:
            self._audit_log("OHLCV", ohlcv)
        
        for callback in self.ohlcv_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ohlcv)
                else:
                    callback(ohlcv)
            except Exception as e:
                logger.error(f"Error in OHLCV callback: {e}")
    
    # ==================== Health & Monitoring ====================
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return ConnectionStatus(
            exchange=self.exchange_name,
            connected=self.connected,
            timestamp=datetime.utcnow(),
            reconnect_count=self.reconnect_count,
            last_message_time=self.last_message_time,
            latency_ms=self._calculate_latency()
        )
    
    def _calculate_latency(self) -> Optional[float]:
        """Calculate average message latency"""
        if self.last_message_time:
            return (datetime.utcnow() - self.last_message_time).total_seconds() * 1000
        return None
    
    def _update_health(self) -> None:
        """Update connection health metrics"""
        self.last_message_time = datetime.utcnow()
        self.message_count += 1
        
        # Reset error count if receiving messages
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle errors with circuit breaker pattern"""
        self.error_count += 1
        logger.error(f"{self.exchange_name} error ({self.error_count}/{self.max_errors}): {error}")
        
        if self.error_count >= self.max_errors:
            self.circuit_open = True
            logger.critical(f"Circuit breaker opened for {self.exchange_name}")
            asyncio.create_task(self._reset_circuit_breaker())
    
    async def _reset_circuit_breaker(self, delay: int = 60) -> None:
        """Reset circuit breaker after delay"""
        await asyncio.sleep(delay)
        self.circuit_open = False
        self.error_count = 0
        logger.info(f"Circuit breaker reset for {self.exchange_name}")
    
    # ==================== Compliance & Audit ====================
    
    def _audit_log(self, event_type: str, data: Any) -> None:
        """
        Log data ingestion events for compliance
        
        SOC-2 Requirement: Maintain audit trail of all data access
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "exchange": self.exchange_name,
            "event_type": event_type,
            "data_summary": self._summarize_data(data),
            "message_count": self.message_count
        }
        
        # In production, write to audit database
        logger.debug(f"AUDIT: {audit_entry}")
    
    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Create summary of data for audit purposes"""
        if isinstance(data, Ticker):
            return {"symbol": data.symbol, "price": data.last, "timestamp": data.timestamp.isoformat()}
        elif isinstance(data, Trade):
            return {"symbol": data.symbol, "price": data.price, "quantity": data.quantity}
        elif isinstance(data, OrderBook):
            return {"symbol": data.symbol, "bid_depth": len(data.bids), "ask_depth": len(data.asks)}
        elif isinstance(data, OHLCV):
            return {"symbol": data.symbol, "timeframe": data.timeframe, "close": data.close}
        return {}
    
    # ==================== Utility Methods ====================
    
    async def health_check(self) -> bool:
        """Perform health check"""
        if not self.connected:
            return False
        
        if self.circuit_open:
            return False
        
        # Check if we've received messages recently (within last 30 seconds)
        if self.last_message_time:
            time_since_last_message = (datetime.utcnow() - self.last_message_time).total_seconds()
            if time_since_last_message > 30:
                logger.warning(f"{self.exchange_name}: No messages for {time_since_last_message}s")
                return False
        
        return True
    
    async def reconnect(self, max_attempts: int = 5) -> bool:
        """Attempt to reconnect with exponential backoff"""
        for attempt in range(max_attempts):
            try:
                await self.disconnect()
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                if await self.connect():
                    self.reconnect_count += 1
                    logger.info(f"{self.exchange_name} reconnected (attempt {attempt + 1})")
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error(f"{self.exchange_name} failed to reconnect after {max_attempts} attempts")
        return False
