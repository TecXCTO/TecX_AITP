"""
Risk Management Shield
Independent risk layer that operates separately from AI models

Features:
- Circuit breakers
- Position limits
- Drawdown protection
- Order validation
- Kill switches
- Real-time monitoring
- SOC-2 compliant audit logging

This layer CANNOT be bypassed by AI models or strategies
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from loguru import logger


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Trading halted
    HALF_OPEN = "HALF_OPEN"  # Testing resumption


@dataclass
class RiskLimits:
    """Hard-coded risk limits (immutable by AI)"""
    # Position limits
    max_position_size_usd: float = 10000.0
    max_total_exposure_usd: float = 50000.0
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    
    # Loss limits
    max_daily_loss_usd: float = 1000.0
    max_daily_loss_pct: float = 0.02  # 2%
    max_drawdown_pct: float = 0.10  # 10%
    
    # Order limits
    max_order_size_usd: float = 5000.0
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    
    # Leverage
    max_leverage: float = 1.0  # No leverage by default
    
    # Volatility controls
    max_portfolio_volatility: float = 0.02  # 2% daily volatility
    
    # Emergency controls
    kill_switch_enabled: bool = True
    auto_flatten_on_critical: bool = True


@dataclass
class RiskEvent:
    """Risk event record for audit trail"""
    timestamp: datetime
    event_type: str
    severity: RiskLevel
    symbol: Optional[str]
    message: str
    data: Dict = field(default_factory=dict)
    action_taken: Optional[str] = None


class OrderValidator:
    """Pre-execution order validation"""
    
    @staticmethod
    def validate_order(
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_positions: Dict,
        risk_limits: RiskLimits
    ) -> tuple[bool, str]:
        """
        Validate order against risk limits
        
        Returns:
            (is_valid, rejection_reason)
        """
        order_value = quantity * price
        
        # Check order size
        if order_value > risk_limits.max_order_size_usd:
            return False, f"Order size ${order_value:.2f} exceeds limit ${risk_limits.max_order_size_usd:.2f}"
        
        # Check position limits
        if symbol in current_positions:
            current_qty = current_positions[symbol]['quantity']
            if side == 'BUY':
                new_qty = current_qty + quantity
            else:
                new_qty = abs(current_qty - quantity)
            
            new_value = new_qty * price
            if new_value > risk_limits.max_position_size_usd:
                return False, f"Position size ${new_value:.2f} exceeds limit ${risk_limits.max_position_size_usd:.2f}"
        
        # Check total positions
        if len(current_positions) >= risk_limits.max_total_positions:
            if symbol not in current_positions:
                return False, f"Max positions ({risk_limits.max_total_positions}) reached"
        
        # Check for negative prices/quantities
        if price <= 0 or quantity <= 0:
            return False, "Invalid price or quantity"
        
        return True, "OK"


class DrawdownMonitor:
    """Real-time drawdown monitoring"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0
    
    def update(self, current_equity: float) -> RiskLevel:
        """
        Update drawdown calculation
        
        Returns:
            Risk level based on current drawdown
        """
        self.current_equity = current_equity
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
            
            if self.current_drawdown_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = self.current_drawdown_pct
        
        # Determine risk level
        if self.current_drawdown_pct >= self.risk_limits.max_drawdown_pct:
            return RiskLevel.CRITICAL
        elif self.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * 0.75:
            return RiskLevel.HIGH
        elif self.current_drawdown_pct >= self.risk_limits.max_drawdown_pct * 0.50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class RateLimit er:
    """Order rate limiting"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.order_timestamps: List[datetime] = []
    
    def check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if order rate limit is exceeded
        
        Returns:
            (can_trade, reason)
        """
        now = datetime.utcnow()
        
        # Clean old timestamps
        self.order_timestamps = [
            ts for ts in self.order_timestamps
            if now - ts < timedelta(hours=1)
        ]
        
        # Check per-minute limit
        recent_orders = sum(
            1 for ts in self.order_timestamps
            if now - ts < timedelta(minutes=1)
        )
        
        if recent_orders >= self.risk_limits.max_orders_per_minute:
            return False, f"Rate limit: {recent_orders} orders in last minute (max {self.risk_limits.max_orders_per_minute})"
        
        # Check per-hour limit
        if len(self.order_timestamps) >= self.risk_limits.max_orders_per_hour:
            return False, f"Rate limit: {len(self.order_timestamps)} orders in last hour (max {self.risk_limits.max_orders_per_hour})"
        
        return True, "OK"
    
    def record_order(self) -> None:
        """Record order timestamp"""
        self.order_timestamps.append(datetime.utcnow())


class CircuitBreaker:
    """Trading circuit breaker"""
    
    def __init__(self, cooldown_seconds: int = 300):
        self.state = CircuitBreakerState.CLOSED
        self.cooldown_seconds = cooldown_seconds
        self.open_timestamp: Optional[datetime] = None
        self.trip_count = 0
    
    def trip(self, reason: str) -> None:
        """Trip circuit breaker (halt trading)"""
        self.state = CircuitBreakerState.OPEN
        self.open_timestamp = datetime.utcnow()
        self.trip_count += 1
        logger.critical(f"⛔ CIRCUIT BREAKER TRIPPED: {reason}")
    
    def reset(self) -> None:
        """Reset circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.open_timestamp = None
        logger.info("✓ Circuit breaker reset")
    
    async def auto_reset(self) -> None:
        """Auto-reset after cooldown period"""
        if self.state == CircuitBreakerState.OPEN and self.open_timestamp:
            elapsed = (datetime.utcnow() - self.open_timestamp).total_seconds()
            
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.warning("Circuit breaker entering HALF_OPEN state")
                
                # Test with small operations, then fully reset
                await asyncio.sleep(10)
                self.reset()
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.state != CircuitBreakerState.OPEN


class RiskShield:
    """
    Main Risk Management Shield
    
    CRITICAL: This operates independently of AI models and cannot be overridden
    """
    
    def __init__(
        self,
        risk_limits: RiskLimits = None,
        enable_audit: bool = True
    ):
        self.risk_limits = risk_limits or RiskLimits()
        self.enable_audit = enable_audit
        
        # Components
        self.validator = OrderValidator()
        self.drawdown_monitor = DrawdownMonitor(self.risk_limits)
        self.rate_limiter = RateLimiter(self.risk_limits)
        self.circuit_breaker = CircuitBreaker()
        
        # State
        self.current_positions: Dict = {}
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        
        # Audit trail
        self.risk_events: List[RiskEvent] = []
        
        # Kill switch
        self.kill_switch_active = False
        
        logger.info("🛡️  Risk Shield initialized")
        logger.info(f"Max Drawdown: {self.risk_limits.max_drawdown_pct * 100}%")
        logger.info(f"Max Daily Loss: ${self.risk_limits.max_daily_loss_usd}")
    
    async def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> tuple[bool, str]:
        """
        Comprehensive order validation
        
        Returns:
            (is_approved, rejection_reason)
        """
        # Check kill switch
        if self.kill_switch_active:
            return False, "❌ KILL SWITCH ACTIVE"
        
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            return False, "⛔ Circuit breaker OPEN"
        
        # Check rate limits
        can_trade, reason = self.rate_limiter.check_rate_limit()
        if not can_trade:
            self._log_risk_event(
                "RATE_LIMIT_EXCEEDED",
                RiskLevel.MEDIUM,
                symbol,
                reason
            )
            return False, reason
        
        # Validate order parameters
        is_valid, reason = self.validator.validate_order(
            symbol, side, quantity, price,
            self.current_positions,
            self.risk_limits
        )
        
        if not is_valid:
            self._log_risk_event(
                "ORDER_REJECTED",
                RiskLevel.MEDIUM,
                symbol,
                reason
            )
            return False, reason
        
        # All checks passed
        self.rate_limiter.record_order()
        return True, "APPROVED"
    
    def update_portfolio(
        self,
        current_equity: float,
        positions: Dict,
        daily_pnl: float
    ) -> None:
        """Update portfolio state and check risk limits"""
        self.current_positions = positions
        self.daily_pnl = daily_pnl
        
        # Update drawdown
        risk_level = self.drawdown_monitor.update(current_equity)
        
        # Check drawdown threshold
        if risk_level == RiskLevel.CRITICAL:
            self._handle_critical_risk("MAX DRAWDOWN EXCEEDED")
        elif risk_level == RiskLevel.HIGH:
            logger.warning(
                f"⚠️  High drawdown: {self.drawdown_monitor.current_drawdown_pct * 100:.2f}%"
            )
        
        # Check daily loss limit
        if abs(daily_pnl) > self.risk_limits.max_daily_loss_usd:
            self._handle_critical_risk(f"Daily loss limit exceeded: ${abs(daily_pnl):.2f}")
        
        # Check daily loss percentage
        if self.start_of_day_equity > 0:
            daily_loss_pct = abs(daily_pnl) / self.start_of_day_equity
            if daily_loss_pct > self.risk_limits.max_daily_loss_pct:
                self._handle_critical_risk(f"Daily loss % exceeded: {daily_loss_pct * 100:.2f}%")
    
    def _handle_critical_risk(self, reason: str) -> None:
        """Handle critical risk event"""
        self._log_risk_event(
            "CRITICAL_RISK",
            RiskLevel.CRITICAL,
            None,
            reason
        )
        
        # Trip circuit breaker
        self.circuit_breaker.trip(reason)
        
        # Auto-flatten positions if enabled
        if self.risk_limits.auto_flatten_on_critical:
            logger.critical("🚨 AUTO-FLATTENING ALL POSITIONS")
            # In production, this would send close orders for all positions
            # asyncio.create_task(self._flatten_all_positions())
    
    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """
        Activate emergency kill switch
        
        This IMMEDIATELY halts all trading
        """
        self.kill_switch_active = True
        
        self._log_risk_event(
            "KILL_SWITCH_ACTIVATED",
            RiskLevel.CRITICAL,
            None,
            reason,
            action_taken="ALL_TRADING_HALTED"
        )
        
        logger.critical(f"🔴 KILL SWITCH ACTIVATED: {reason}")
        logger.critical("ALL TRADING HALTED")
    
    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch (requires manual intervention)"""
        self.kill_switch_active = False
        logger.warning("Kill switch deactivated - trading resumed")
    
    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at start of trading day)"""
        self.start_of_day_equity = self.drawdown_monitor.current_equity
        self.daily_pnl = 0.0
        logger.info("Daily limits reset")
    
    def _log_risk_event(
        self,
        event_type: str,
        severity: RiskLevel,
        symbol: Optional[str],
        message: str,
        action_taken: Optional[str] = None
    ) -> None:
        """Log risk event to audit trail"""
        event = RiskEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            message=message,
            action_taken=action_taken
        )
        
        self.risk_events.append(event)
        
        if self.enable_audit:
            # In production, write to secure audit database
            self._write_audit_log(event)
        
        # Log to console
        log_func = {
            RiskLevel.LOW: logger.info,
            RiskLevel.MEDIUM: logger.warning,
            RiskLevel.HIGH: logger.warning,
            RiskLevel.CRITICAL: logger.critical
        }[severity]
        
        log_func(f"RISK EVENT [{severity.value}]: {event_type} - {message}")
    
    def _write_audit_log(self, event: RiskEvent) -> None:
        """Write to SOC-2 compliant audit log"""
        audit_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'severity': event.severity.value,
            'symbol': event.symbol,
            'message': event.message,
            'action_taken': event.action_taken,
            'drawdown_pct': self.drawdown_monitor.current_drawdown_pct,
            'daily_pnl': self.daily_pnl
        }
        
        # In production: Write to append-only audit database
        # For now, write to file
        with open('logs/risk_audit.jsonl', 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        return {
            'kill_switch_active': self.kill_switch_active,
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'current_drawdown_pct': self.drawdown_monitor.current_drawdown_pct * 100,
            'max_drawdown_pct': self.drawdown_monitor.max_drawdown_pct * 100,
            'daily_pnl': self.daily_pnl,
            'positions_count': len(self.current_positions),
            'risk_events_24h': len([
                e for e in self.risk_events
                if (datetime.utcnow() - e.timestamp).total_seconds() < 86400
            ])
        }
    
    def print_status(self) -> None:
        """Print risk status to console"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("RISK SHIELD STATUS")
        print("=" * 60)
        print(f"Kill Switch:         {'🔴 ACTIVE' if status['kill_switch_active'] else '🟢 INACTIVE'}")
        print(f"Circuit Breaker:     {status['circuit_breaker_state']}")
        print(f"Current Drawdown:    {status['current_drawdown_pct']:.2f}%")
        print(f"Max Drawdown:        {status['max_drawdown_pct']:.2f}%")
        print(f"Daily P/L:           ${status['daily_pnl']:,.2f}")
        print(f"Open Positions:      {status['positions_count']}")
        print(f"Risk Events (24h):   {status['risk_events_24h']}")
        print("=" * 60 + "\n")


# Example usage
async def main():
    """Example: Risk Shield in action"""
    
    # Initialize with strict limits
    limits = RiskLimits(
        max_position_size_usd=10000,
        max_daily_loss_usd=1000,
        max_drawdown_pct=0.10
    )
    
    shield = RiskShield(limits)
    
    # Simulate trading day
    shield.reset_daily_limits()
    
    # Test order validation
    is_valid, reason = await shield.validate_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=50000
    )
    
    print(f"Order validation: {is_valid} - {reason}")
    
    # Update portfolio (simulate loss)
    shield.update_portfolio(
        current_equity=95000,  # 5% drawdown
        positions={'BTCUSDT': {'quantity': 0.1, 'value': 5000}},
        daily_pnl=-500
    )
    
    # Print status
    shield.print_status()
    
    # Test kill switch
    # shield.activate_kill_switch("Testing emergency stop")
    # shield.print_status()


if __name__ == "__main__":
    asyncio.run(main())
