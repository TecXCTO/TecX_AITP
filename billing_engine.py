"""
Metering & Billing Engine
Track resource usage and generate invoices for multi-tenant SaaS

Features:
- Resource usage tracking (CPU, memory, API calls, storage)
- Usage aggregation per tenant
- Billing calculations
- Invoice generation
- Stripe integration
- Usage-based pricing
- Overage handling

SOC-2 Compliant | Audit Trail
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal
import asyncio
from loguru import logger
from collections import defaultdict
import json


class ResourceType(Enum):
    """Resource types for metering"""
    API_CALLS = "api_calls"
    CPU_SECONDS = "cpu_seconds"
    MEMORY_GB_HOURS = "memory_gb_hours"
    STORAGE_GB = "storage_gb"
    TRADES = "trades"
    STRATEGIES = "strategies"
    BACKTEST_HOURS = "backtest_hours"
    AI_INFERENCE = "ai_inference"


@dataclass
class UsageRecord:
    """Single usage record"""
    tenant_id: str
    resource_type: ResourceType
    quantity: Decimal
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    
    # Audit
    recorded_by: str = "system"
    record_id: str = ""


@dataclass
class UsageAggregation:
    """Aggregated usage for a period"""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    resource_usage: Dict[ResourceType, Decimal]
    total_cost: Decimal = Decimal('0.00')


@dataclass
class PricingTier:
    """Pricing configuration"""
    # Base subscription
    base_price_monthly: Decimal = Decimal('49.00')
    
    # Resource prices (per unit)
    prices: Dict[ResourceType, Decimal] = field(default_factory=lambda: {
        ResourceType.API_CALLS: Decimal('0.0001'),  # $0.0001 per API call
        ResourceType.CPU_SECONDS: Decimal('0.00005'),  # $0.05 per 1000 CPU seconds
        ResourceType.MEMORY_GB_HOURS: Decimal('0.01'),  # $0.01 per GB-hour
        ResourceType.STORAGE_GB: Decimal('0.10'),  # $0.10 per GB per month
        ResourceType.TRADES: Decimal('0.01'),  # $0.01 per trade
        ResourceType.STRATEGIES: Decimal('5.00'),  # $5 per strategy per month
        ResourceType.BACKTEST_HOURS: Decimal('1.00'),  # $1 per backtest hour
        ResourceType.AI_INFERENCE: Decimal('0.001'),  # $0.001 per inference
    })
    
    # Included quotas (free tier)
    included_quota: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.API_CALLS: 10000,
        ResourceType.CPU_SECONDS: 3600,
        ResourceType.MEMORY_GB_HOURS: 100,
        ResourceType.STORAGE_GB: 1,
        ResourceType.TRADES: 100,
        ResourceType.STRATEGIES: 1,
        ResourceType.BACKTEST_HOURS: 1,
        ResourceType.AI_INFERENCE: 1000,
    })


class UsageCollector:
    """
    Collects resource usage metrics
    
    Integrates with:
    - API Gateway (request counting)
    - Kubernetes metrics (CPU/memory)
    - Database (storage)
    - Trading engine (trades)
    """
    
    def __init__(self):
        self.usage_buffer: Dict[str, List[UsageRecord]] = defaultdict(list)
        self.flush_interval = 60  # seconds
        self.buffer_size = 1000  # records
    
    async def record_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        quantity: Decimal,
        metadata: Dict = None
    ):
        """
        Record resource usage
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource consumed
            quantity: Amount consumed
            metadata: Additional metadata
        """
        record = UsageRecord(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            record_id=f"{tenant_id}_{int(datetime.utcnow().timestamp() * 1000)}"
        )
        
        # Add to buffer
        self.usage_buffer[tenant_id].append(record)
        
        # Flush if buffer is full
        if len(self.usage_buffer[tenant_id]) >= self.buffer_size:
            await self.flush_usage(tenant_id)
        
        logger.debug(f"Recorded: {tenant_id} used {quantity} {resource_type.value}")
    
    async def flush_usage(self, tenant_id: str = None):
        """
        Flush usage buffer to database
        
        Args:
            tenant_id: Specific tenant to flush, or None for all
        """
        tenants_to_flush = [tenant_id] if tenant_id else list(self.usage_buffer.keys())
        
        for tid in tenants_to_flush:
            if tid in self.usage_buffer and self.usage_buffer[tid]:
                records = self.usage_buffer[tid]
                
                # Save to database
                await self._save_usage_records(records)
                
                # Clear buffer
                self.usage_buffer[tid] = []
                
                logger.info(f"Flushed {len(records)} usage records for {tid}")
    
    async def _save_usage_records(self, records: List[UsageRecord]):
        """Save usage records to database"""
        # In production: INSERT INTO usage_records...
        # For now, just log
        for record in records:
            logger.debug(f"Saved: {record.record_id}")
    
    async def auto_flush_loop(self):
        """Periodically flush usage buffer"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush_usage()


class UsageAggregator:
    """
    Aggregates usage for billing periods
    """
    
    def __init__(self):
        pass
    
    async def aggregate_usage(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> UsageAggregation:
        """
        Aggregate usage for a billing period
        
        Args:
            tenant_id: Tenant identifier
            period_start: Period start datetime
            period_end: Period end datetime
        
        Returns:
            UsageAggregation object
        """
        # Query usage records from database
        records = await self._fetch_usage_records(tenant_id, period_start, period_end)
        
        # Aggregate by resource type
        resource_usage = defaultdict(Decimal)
        
        for record in records:
            resource_usage[record.resource_type] += record.quantity
        
        aggregation = UsageAggregation(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            resource_usage=dict(resource_usage)
        )
        
        logger.info(f"Aggregated usage for {tenant_id}: {len(records)} records")
        return aggregation
    
    async def _fetch_usage_records(
        self,
        tenant_id: str,
        start: datetime,
        end: datetime
    ) -> List[UsageRecord]:
        """Fetch usage records from database"""
        # In production: SELECT * FROM usage_records WHERE...
        # For now, return mock data
        return []


class BillingEngine:
    """
    Calculates bills based on usage and pricing
    """
    
    def __init__(self, pricing: PricingTier = None):
        self.pricing = pricing or PricingTier()
        self.aggregator = UsageAggregator()
    
    async def calculate_bill(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        subscription_plan: str = "professional"
    ) -> Dict:
        """
        Calculate bill for a billing period
        
        Args:
            tenant_id: Tenant identifier
            period_start: Period start
            period_end: Period end
            subscription_plan: Plan name
        
        Returns:
            Bill breakdown
        """
        # Get usage aggregation
        usage = await self.aggregator.aggregate_usage(tenant_id, period_start, period_end)
        
        # Calculate costs
        base_cost = self.pricing.base_price_monthly
        usage_cost = Decimal('0.00')
        overage_cost = Decimal('0.00')
        
        cost_breakdown = {}
        
        for resource_type, quantity in usage.resource_usage.items():
            # Check included quota
            included = self.pricing.included_quota.get(resource_type, 0)
            overage = max(0, quantity - included)
            
            if overage > 0:
                price_per_unit = self.pricing.prices.get(resource_type, Decimal('0'))
                cost = overage * price_per_unit
                overage_cost += cost
                
                cost_breakdown[resource_type.value] = {
                    'quantity': float(quantity),
                    'included': included,
                    'overage': float(overage),
                    'price_per_unit': float(price_per_unit),
                    'cost': float(cost)
                }
        
        total_cost = base_cost + overage_cost
        
        bill = {
            'tenant_id': tenant_id,
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'subscription_plan': subscription_plan,
            'base_cost': float(base_cost),
            'usage_cost': float(overage_cost),
            'total_cost': float(total_cost),
            'breakdown': cost_breakdown,
            'currency': 'USD'
        }
        
        logger.info(f"Calculated bill for {tenant_id}: ${total_cost:.2f}")
        return bill
    
    async def generate_invoice(
        self,
        tenant_id: str,
        bill: Dict
    ) -> str:
        """
        Generate invoice PDF/HTML
        
        Args:
            tenant_id: Tenant identifier
            bill: Bill data
        
        Returns:
            Invoice ID
        """
        invoice_id = f"INV-{tenant_id}-{int(datetime.utcnow().timestamp())}"
        
        # In production:
        # 1. Generate PDF invoice
        # 2. Save to storage (S3)
        # 3. Send via email
        # 4. Save to database
        
        logger.info(f"Generated invoice: {invoice_id} for ${bill['total_cost']:.2f}")
        return invoice_id


class StripeIntegration:
    """
    Stripe payment processing integration
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In production: import stripe library
    
    async def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: str
    ) -> str:
        """
        Create Stripe customer
        
        Returns:
            Stripe customer ID
        """
        # stripe.Customer.create(
        #     email=email,
        #     name=name,
        #     metadata={'tenant_id': tenant_id}
        # )
        
        customer_id = f"cus_{tenant_id}"
        logger.info(f"Created Stripe customer: {customer_id}")
        return customer_id
    
    async def create_subscription(
        self,
        customer_id: str,
        plan_id: str
    ) -> str:
        """
        Create subscription
        
        Returns:
            Stripe subscription ID
        """
        # stripe.Subscription.create(
        #     customer=customer_id,
        #     items=[{'price': plan_id}]
        # )
        
        subscription_id = f"sub_{customer_id}"
        logger.info(f"Created subscription: {subscription_id}")
        return subscription_id
    
    async def charge_usage(
        self,
        customer_id: str,
        amount: Decimal,
        description: str
    ):
        """
        Charge for usage overage
        
        Args:
            customer_id: Stripe customer ID
            amount: Amount to charge (USD)
            description: Charge description
        """
        # stripe.Charge.create(
        #     customer=customer_id,
        #     amount=int(amount * 100),  # Convert to cents
        #     currency='usd',
        #     description=description
        # )
        
        logger.info(f"Charged {customer_id}: ${amount:.2f} - {description}")
    
    async def invoice_customer(
        self,
        customer_id: str,
        bill: Dict
    ):
        """
        Create and send invoice via Stripe
        
        Args:
            customer_id: Stripe customer ID
            bill: Bill data
        """
        # Create invoice items
        # stripe.InvoiceItem.create(
        #     customer=customer_id,
        #     amount=int(bill['base_cost'] * 100),
        #     currency='usd',
        #     description='Monthly subscription'
        # )
        
        # Add usage charges
        # for resource, data in bill['breakdown'].items():
        #     stripe.InvoiceItem.create(
        #         customer=customer_id,
        #         amount=int(data['cost'] * 100),
        #         currency='usd',
        #         description=f"{resource} overage"
        #     )
        
        # Create and finalize invoice
        # invoice = stripe.Invoice.create(customer=customer_id)
        # stripe.Invoice.finalize_invoice(invoice.id)
        
        logger.info(f"Invoiced {customer_id}: ${bill['total_cost']:.2f}")


class MeteringService:
    """
    Main metering service
    
    Orchestrates usage collection, aggregation, and billing
    """
    
    def __init__(self, stripe_api_key: str = None):
        self.collector = UsageCollector()
        self.billing_engine = BillingEngine()
        self.stripe = StripeIntegration(stripe_api_key) if stripe_api_key else None
    
    async def start(self):
        """Start metering service"""
        logger.info("Starting metering service...")
        
        # Start auto-flush loop
        asyncio.create_task(self.collector.auto_flush_loop())
        
        # Start monthly billing loop
        asyncio.create_task(self._monthly_billing_loop())
        
        logger.info("✓ Metering service started")
    
    async def track_api_call(self, tenant_id: str):
        """Track single API call"""
        await self.collector.record_usage(
            tenant_id,
            ResourceType.API_CALLS,
            Decimal('1')
        )
    
    async def track_cpu_usage(self, tenant_id: str, cpu_seconds: float):
        """Track CPU usage"""
        await self.collector.record_usage(
            tenant_id,
            ResourceType.CPU_SECONDS,
            Decimal(str(cpu_seconds))
        )
    
    async def track_memory_usage(self, tenant_id: str, gb_hours: float):
        """Track memory usage"""
        await self.collector.record_usage(
            tenant_id,
            ResourceType.MEMORY_GB_HOURS,
            Decimal(str(gb_hours))
        )
    
    async def track_trade(self, tenant_id: str):
        """Track single trade"""
        await self.collector.record_usage(
            tenant_id,
            ResourceType.TRADES,
            Decimal('1')
        )
    
    async def get_current_usage(
        self,
        tenant_id: str
    ) -> Dict[ResourceType, Decimal]:
        """
        Get current month's usage
        
        Returns:
            Usage by resource type
        """
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage = await self.billing_engine.aggregator.aggregate_usage(
            tenant_id,
            period_start,
            now
        )
        
        return usage.resource_usage
    
    async def generate_monthly_bill(self, tenant_id: str) -> Dict:
        """Generate bill for current month"""
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        bill = await self.billing_engine.calculate_bill(
            tenant_id,
            period_start,
            now
        )
        
        return bill
    
    async def _monthly_billing_loop(self):
        """Generate monthly bills"""
        while True:
            # Wait until first day of month
            now = datetime.utcnow()
            next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
            wait_seconds = (next_month - now).total_seconds()
            
            await asyncio.sleep(wait_seconds)
            
            # Generate bills for all tenants
            logger.info("Generating monthly bills...")
            
            # In production: fetch all active tenants
            # for tenant in tenants:
            #     await self._bill_tenant(tenant)


# Example usage
async def main():
    """Example: Metering service"""
    
    # Initialize service
    service = MeteringService()
    await service.start()
    
    # Simulate usage
    tenant_id = "acme-corp"
    
    # Track API calls
    for _ in range(100):
        await service.track_api_call(tenant_id)
    
    # Track trades
    for _ in range(10):
        await service.track_trade(tenant_id)
    
    # Track CPU usage
    await service.track_cpu_usage(tenant_id, 3600)  # 1 hour
    
    # Get current usage
    usage = await service.get_current_usage(tenant_id)
    print(f"Current usage: {usage}")
    
    # Generate bill
    bill = await service.generate_monthly_bill(tenant_id)
    print(f"\nMonthly bill: ${bill['total_cost']:.2f}")
    print(json.dumps(bill, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
