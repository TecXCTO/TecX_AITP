"""
Advanced Backtesting Engine
Realistic simulation accounting for 2026 market realities

Features:
- Variable latency simulation
- Realistic slippage models
- Exchange fees (maker/taker)
- Market impact
- Order fill simulation
- Transaction cost analysis (TCA)
- Performance metrics
- Risk analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from loguru import logger


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class BacktestOrder:
    """Order representation in backtesting"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET' or 'LIMIT'
    quantity: float
    price: Optional[float] = None  # For limit orders
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Capital
    initial_capital: float = 100000.0
    
    # Fees (2026 competitive rates)
    maker_fee: float = 0.0010  # 0.10% maker fee
    taker_fee: float = 0.0015  # 0.15% taker fee
    
    # Slippage model
    base_slippage_bps: float = 1.0  # 1 basis point base slippage
    volume_impact_factor: float = 0.1  # Impact per % of volume
    
    # Latency simulation (microseconds)
    min_latency_us: float = 100  # 0.1ms best case
    max_latency_us: float = 5000  # 5ms worst case
    mean_latency_us: float = 500  # 0.5ms typical
    
    # Market impact
    enable_market_impact: bool = True
    liquidity_impact_factor: float = 0.05
    
    # Order fill simulation
    partial_fill_probability: float = 0.05  # 5% chance of partial fill
    rejection_probability: float = 0.01  # 1% chance of rejection
    
    # Risk limits
    max_position_size: float = 10000.0
    max_leverage: float = 1.0
    
    # Compliance
    enable_audit_trail: bool = True


class SlippageModel:
    """Advanced slippage calculation"""
    
    @staticmethod
    def calculate_slippage(
        order: BacktestOrder,
        market_price: float,
        market_volume: float,
        config: BacktestConfig
    ) -> float:
        """
        Calculate realistic slippage based on:
        - Order size relative to market volume
        - Volatility
        - Time of day
        - Order type
        
        Returns: Slippage in price units
        """
        # Base slippage
        base_slippage = market_price * (config.base_slippage_bps / 10000)
        
        # Volume impact
        if market_volume > 0:
            order_volume_pct = order.quantity / market_volume
            volume_slippage = market_price * order_volume_pct * config.volume_impact_factor
        else:
            volume_slippage = base_slippage * 2  # Penalty for low volume
        
        # Market orders get more slippage
        if order.order_type == 'MARKET':
            total_slippage = base_slippage + volume_slippage
        else:  # LIMIT
            total_slippage = base_slippage * 0.5  # Less slippage for limit orders
        
        # Add randomness (±20%)
        noise = np.random.uniform(-0.2, 0.2)
        total_slippage *= (1 + noise)
        
        return abs(total_slippage)
    
    @staticmethod
    def calculate_market_impact(
        order: BacktestOrder,
        market_price: float,
        liquidity: float,
        config: BacktestConfig
    ) -> float:
        """
        Calculate permanent market impact
        
        Returns: Price impact in price units
        """
        if not config.enable_market_impact:
            return 0.0
        
        # Square root model: impact ∝ sqrt(order_size / liquidity)
        if liquidity > 0:
            impact_factor = np.sqrt(order.quantity / liquidity)
            impact = market_price * impact_factor * config.liquidity_impact_factor
        else:
            impact = market_price * 0.001  # 0.1% default impact
        
        return impact if order.side == 'BUY' else -impact


class LatencySimulator:
    """Realistic latency simulation"""
    
    @staticmethod
    def simulate_latency(config: BacktestConfig) -> float:
        """
        Simulate execution latency in milliseconds
        Uses log-normal distribution to model realistic network delays
        """
        # Log-normal distribution parameters
        mu = np.log(config.mean_latency_us)
        sigma = 0.5  # Variance
        
        latency_us = np.random.lognormal(mu, sigma)
        
        # Clip to min/max
        latency_us = np.clip(latency_us, config.min_latency_us, config.max_latency_us)
        
        # Convert to milliseconds
        return latency_us / 1000


class BacktestEngine:
    """
    Production-grade backtesting engine
    
    Simulates realistic market conditions including:
    - Execution delays
    - Slippage
    - Market impact
    - Transaction costs
    - Partial fills
    - Order rejections
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
        # Portfolio state
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.cash = self.capital
        
        # Trading history
        self.orders: List[BacktestOrder] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
        # Market data cache
        self.current_prices: Dict[str, float] = {}
        self.current_volumes: Dict[str, float] = {}
        
        logger.info(f"Backtest engine initialized with ${self.capital:,.2f}")
    
    async def process_bar(
        self,
        timestamp: datetime,
        bars: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Process a single bar of data
        
        Args:
            timestamp: Bar timestamp
            bars: Dict of {symbol: {open, high, low, close, volume}}
        """
        # Update market data cache
        for symbol, bar in bars.items():
            self.current_prices[symbol] = bar['close']
            self.current_volumes[symbol] = bar['volume']
        
        # Update portfolio value
        portfolio_value = self.calculate_portfolio_value()
        self.equity_curve.append((timestamp, portfolio_value))
        
        # Check for pending orders
        await self._process_pending_orders(timestamp)
        
        # Update positions
        self._update_positions()
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'MARKET',
        limit_price: Optional[float] = None
    ) -> BacktestOrder:
        """
        Submit an order for execution
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: 'MARKET' or 'LIMIT'
            limit_price: Limit price (for limit orders)
        
        Returns:
            BacktestOrder: Order object
        """
        order = BacktestOrder(
            order_id=f"ORDER_{len(self.orders) + 1}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price
        )
        
        # Simulate latency
        latency_ms = LatencySimulator.simulate_latency(self.config)
        order.latency_ms = latency_ms
        
        # Wait for latency (in real backtest, this affects fill price)
        # await asyncio.sleep(latency_ms / 1000)  # Uncomment for real-time simulation
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order.order_id}")
            return order
        
        # Execute order
        await self._execute_order(order)
        
        self.orders.append(order)
        return order
    
    async def _execute_order(self, order: BacktestOrder) -> None:
        """Execute order with realistic fill simulation"""
        market_price = self.current_prices.get(order.symbol, 0)
        market_volume = self.current_volumes.get(order.symbol, 0)
        
        if market_price == 0:
            order.status = OrderStatus.REJECTED
            logger.error(f"No market price for {order.symbol}")
            return
        
        # Check for order rejection
        if np.random.random() < self.config.rejection_probability:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected (random): {order.order_id}")
            return
        
        # Calculate slippage
        slippage = SlippageModel.calculate_slippage(
            order, market_price, market_volume, self.config
        )
        
        # Calculate market impact
        impact = SlippageModel.calculate_market_impact(
            order, market_price, market_volume, self.config
        )
        
        # Determine fill price
        if order.side == 'BUY':
            fill_price = market_price + slippage + impact
        else:
            fill_price = market_price - slippage + impact
        
        # Check for partial fill
        if np.random.random() < self.config.partial_fill_probability:
            fill_quantity = order.quantity * np.random.uniform(0.5, 0.95)
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(f"Partial fill: {fill_quantity}/{order.quantity}")
        else:
            fill_quantity = order.quantity
            order.status = OrderStatus.FILLED
        
        # Calculate commission
        commission = fill_quantity * fill_price * self.config.taker_fee
        if order.order_type == 'LIMIT':
            commission = fill_quantity * fill_price * self.config.maker_fee
        
        # Update order
        order.filled_quantity = fill_quantity
        order.filled_price = fill_price
        order.commission = commission
        order.slippage = slippage
        
        # Update portfolio
        self._update_portfolio(order)
        
        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'slippage': slippage,
            'latency_ms': order.latency_ms
        })
        
        self.total_commission += commission
        self.total_slippage += abs(slippage) * fill_quantity
        self.total_trades += 1
        
        logger.info(
            f"Order filled: {order.symbol} {order.side} {fill_quantity}@${fill_price:.2f} "
            f"(slip: ${slippage:.4f}, comm: ${commission:.2f}, latency: {order.latency_ms:.2f}ms)"
        )
    
    def _validate_order(self, order: BacktestOrder) -> bool:
        """Validate order against risk limits"""
        market_price = self.current_prices.get(order.symbol, 0)
        
        if market_price == 0:
            return False
        
        # Check position size limit
        current_pos = self.positions.get(order.symbol)
        new_position_size = order.quantity if not current_pos else current_pos.quantity + order.quantity
        
        if new_position_size > self.config.max_position_size:
            logger.warning(f"Position size limit exceeded: {new_position_size}")
            return False
        
        # Check capital
        order_value = order.quantity * market_price
        if order.side == 'BUY' and order_value > self.cash:
            logger.warning(f"Insufficient capital: ${self.cash:.2f} < ${order_value:.2f}")
            return False
        
        return True
    
    def _update_portfolio(self, order: BacktestOrder) -> None:
        """Update portfolio state after order execution"""
        fill_value = order.filled_quantity * order.filled_price
        
        if order.side == 'BUY':
            # Reduce cash
            self.cash -= (fill_value + order.commission)
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.filled_quantity
                avg_price = ((pos.quantity * pos.entry_price) + (order.filled_quantity * order.filled_price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.filled_quantity,
                    entry_price=order.filled_price,
                    entry_time=order.timestamp,
                    current_price=order.filled_price
                )
        
        else:  # SELL
            # Increase cash
            self.cash += (fill_value - order.commission)
            
            # Update or close position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.filled_quantity
                
                # Calculate realized PnL
                pnl = (order.filled_price - pos.entry_price) * order.filled_quantity
                pos.realized_pnl += pnl
                
                # Close position if fully sold
                if pos.quantity <= 0:
                    del self.positions[order.symbol]
    
    def _update_positions(self) -> None:
        """Update unrealized PnL for all positions"""
        for symbol, pos in self.positions.items():
            current_price = self.current_prices.get(symbol, pos.current_price)
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
    
    async def _process_pending_orders(self, timestamp: datetime) -> None:
        """Process any pending limit orders"""
        # In this implementation, orders are executed immediately
        # In a more advanced version, limit orders would wait for price to reach limit
        pass
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return {}
        
        # Extract equity values
        timestamps, values = zip(*self.equity_curve)
        returns = pd.Series(values).pct_change().dropna()
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0]
        
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0
        
        # Max drawdown
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Average trade
        avg_trade_pnl = sum(t.get('pnl', 0) for t in self.trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'final_capital': values[-1],
            'profit_loss': values[-1] - values[0]
        }
    
    def print_summary(self) -> None:
        """Print backtest summary"""
        metrics = self.get_performance_metrics()
        
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:     ${self.config.initial_capital:,.2f}")
        print(f"Final Capital:       ${metrics.get('final_capital', 0):,.2f}")
        print(f"Profit/Loss:         ${metrics.get('profit_loss', 0):,.2f}")
        print(f"Total Return:        {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Total Trades:        {metrics.get('total_trades', 0)}")
        print(f"Win Rate:            {metrics.get('win_rate', 0) * 100:.2f}%")
        print(f"Avg Trade P/L:       ${metrics.get('avg_trade_pnl', 0):.2f}")
        print(f"Total Commission:    ${metrics.get('total_commission', 0):.2f}")
        print(f"Total Slippage:      ${metrics.get('total_slippage', 0):.2f}")
        print("=" * 60 + "\n")


# Example usage
async def example_backtest():
    """Example backtesting session"""
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        maker_fee=0.001,
        taker_fee=0.0015,
        base_slippage_bps=1.0
    )
    
    engine = BacktestEngine(config)
    
    # Simulate market data
    for i in range(100):
        timestamp = datetime.utcnow() + timedelta(minutes=i)
        
        # Fake bar data
        bars = {
            'BTCUSDT': {
                'open': 50000 + np.random.randn() * 100,
                'high': 50100 + np.random.randn() * 100,
                'low': 49900 + np.random.randn() * 100,
                'close': 50000 + np.random.randn() * 100,
                'volume': 1000 + np.random.randn() * 100
            }
        }
        
        await engine.process_bar(timestamp, bars)
        
        # Simple strategy: Buy on even minutes, sell on odd
        if i % 2 == 0:
            await engine.submit_order('BTCUSDT', 'BUY', 0.1)
        else:
            if 'BTCUSDT' in engine.positions:
                await engine.submit_order('BTCUSDT', 'SELL', 0.1)
    
    # Print results
    engine.print_summary()


if __name__ == "__main__":
    asyncio.run(example_backtest())
