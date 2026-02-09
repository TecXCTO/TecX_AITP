"""
Tenant Manager - Multi-Tenant Isolation & Provisioning

Features:
- Tenant onboarding and provisioning
- Database schema isolation
- Resource quota management
- Tenant lifecycle management
- Data isolation enforcement

SOC-2 Compliant | Multi-Tenant SaaS
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import secrets
import hashlib
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class SubscriptionPlan(Enum):
    """Subscription plan tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TenantStatus(Enum):
    """Tenant lifecycle status"""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    DELETED = "deleted"


@dataclass
class ResourceQuota:
    """Per-tenant resource limits"""
    # Trading limits
    max_bots: int = 1
    max_positions: int = 10
    max_daily_trades: int = 100
    
    # API limits
    api_calls_per_minute: int = 60
    api_calls_per_day: int = 10000
    
    # Storage limits
    storage_gb: int = 1
    max_strategies: int = 5
    
    # Compute limits
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    
    # Data retention
    data_retention_days: int = 90


@dataclass
class Tenant:
    """Tenant entity"""
    tenant_id: str
    name: str
    slug: str  # URL-safe identifier
    email: str
    plan: SubscriptionPlan
    status: TenantStatus
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration
    schema_name: str = ""  # PostgreSQL schema name
    encryption_key: str = ""  # Per-tenant encryption key
    api_key: str = ""  # Tenant API key
    
    # Quotas
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    
    # Billing
    stripe_customer_id: Optional[str] = None
    subscription_id: Optional[str] = None
    trial_ends_at: Optional[datetime] = None
    
    # Settings
    settings: Dict = field(default_factory=dict)


class TenantManager:
    """
    Manages tenant lifecycle and isolation
    
    Responsibilities:
    - Tenant provisioning
    - Schema creation
    - Resource quota enforcement
    - Tenant deprovisioning
    """
    
    def __init__(self, master_db_url: str):
        """
        Initialize tenant manager
        
        Args:
            master_db_url: Connection string to master database
        """
        self.master_engine = create_engine(master_db_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.master_engine)
        
        logger.info("Tenant Manager initialized")
    
    def create_tenant(
        self,
        name: str,
        email: str,
        plan: SubscriptionPlan = SubscriptionPlan.STARTER,
        trial_days: int = 14
    ) -> Tenant:
        """
        Create and provision a new tenant
        
        Args:
            name: Tenant/company name
            email: Admin email
            plan: Subscription plan
            trial_days: Trial period length
        
        Returns:
            Tenant object
        """
        # Generate unique identifiers
        tenant_id = self._generate_tenant_id()
        slug = self._generate_slug(name)
        schema_name = f"tenant_{slug}"
        
        # Generate security credentials
        api_key = self._generate_api_key()
        encryption_key = self._generate_encryption_key()
        
        # Set quota based on plan
        quota = self._get_plan_quota(plan)
        
        # Calculate trial end date
        trial_ends_at = datetime.utcnow() + timedelta(days=trial_days) if trial_days > 0 else None
        
        # Create tenant object
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            slug=slug,
            email=email,
            plan=plan,
            status=TenantStatus.PENDING,
            schema_name=schema_name,
            encryption_key=encryption_key,
            api_key=api_key,
            quota=quota,
            trial_ends_at=trial_ends_at
        )
        
        try:
            # 1. Save tenant to master database
            self._save_tenant_to_master(tenant)
            
            # 2. Create tenant schema
            self._create_tenant_schema(tenant)
            
            # 3. Initialize tenant tables
            self._initialize_tenant_tables(tenant)
            
            # 4. Set up resource quotas
            self._setup_resource_quotas(tenant)
            
            # 5. Create default resources
            self._create_default_resources(tenant)
            
            # Update status
            tenant.status = TenantStatus.ACTIVE
            self._update_tenant_status(tenant.tenant_id, TenantStatus.ACTIVE)
            
            logger.info(f"✓ Tenant created: {tenant.name} ({tenant.tenant_id})")
            logger.info(f"  Plan: {tenant.plan.value}")
            logger.info(f"  Schema: {tenant.schema_name}")
            logger.info(f"  API Key: {tenant.api_key[:16]}...")
            
            return tenant
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            # Rollback: Clean up any created resources
            self._rollback_tenant_creation(tenant)
            raise
    
    def get_tenant(self, tenant_id: str = None, api_key: str = None) -> Optional[Tenant]:
        """
        Retrieve tenant by ID or API key
        
        Args:
            tenant_id: Tenant ID
            api_key: Tenant API key
        
        Returns:
            Tenant object or None
        """
        with self.Session() as session:
            if tenant_id:
                query = text("""
                    SELECT * FROM tenants WHERE tenant_id = :tenant_id
                """)
                result = session.execute(query, {"tenant_id": tenant_id})
            elif api_key:
                query = text("""
                    SELECT * FROM tenants WHERE api_key = :api_key
                """)
                result = session.execute(query, {"api_key": api_key})
            else:
                return None
            
            row = result.fetchone()
            if row:
                return self._row_to_tenant(row)
            return None
    
    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        plan: Optional[SubscriptionPlan] = None
    ) -> List[Tenant]:
        """List all tenants with optional filters"""
        with self.Session() as session:
            query = "SELECT * FROM tenants WHERE 1=1"
            params = {}
            
            if status:
                query += " AND status = :status"
                params["status"] = status.value
            
            if plan:
                query += " AND plan = :plan"
                params["plan"] = plan.value
            
            result = session.execute(text(query), params)
            return [self._row_to_tenant(row) for row in result]
    
    def update_tenant_plan(self, tenant_id: str, new_plan: SubscriptionPlan) -> None:
        """
        Update tenant subscription plan
        
        Args:
            tenant_id: Tenant ID
            new_plan: New subscription plan
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Update quota
        new_quota = self._get_plan_quota(new_plan)
        
        with self.Session() as session:
            query = text("""
                UPDATE tenants 
                SET plan = :plan,
                    quota = :quota,
                    updated_at = :updated_at
                WHERE tenant_id = :tenant_id
            """)
            
            session.execute(query, {
                "plan": new_plan.value,
                "quota": new_quota.__dict__,
                "updated_at": datetime.utcnow(),
                "tenant_id": tenant_id
            })
            session.commit()
        
        logger.info(f"Updated tenant {tenant_id} to {new_plan.value} plan")
    
    def suspend_tenant(self, tenant_id: str, reason: str = "") -> None:
        """
        Suspend tenant (e.g., non-payment, TOS violation)
        
        Args:
            tenant_id: Tenant ID
            reason: Suspension reason
        """
        self._update_tenant_status(tenant_id, TenantStatus.SUSPENDED)
        
        # Stop all running bots
        # Disable API access
        # Send notification
        
        logger.warning(f"Tenant {tenant_id} suspended: {reason}")
    
    def delete_tenant(self, tenant_id: str, hard_delete: bool = False) -> None:
        """
        Delete tenant (soft or hard delete)
        
        Args:
            tenant_id: Tenant ID
            hard_delete: If True, permanently delete all data
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        if hard_delete:
            # Hard delete: Remove all data permanently
            self._drop_tenant_schema(tenant)
            self._delete_tenant_from_master(tenant_id)
            logger.critical(f"Tenant {tenant_id} HARD DELETED")
        else:
            # Soft delete: Mark as deleted but keep data
            self._update_tenant_status(tenant_id, TenantStatus.DELETED)
            logger.warning(f"Tenant {tenant_id} soft deleted")
    
    # ==================== Private Methods ====================
    
    def _generate_tenant_id(self) -> str:
        """Generate unique tenant ID"""
        return f"tn_{secrets.token_urlsafe(16)}"
    
    def _generate_slug(self, name: str) -> str:
        """Generate URL-safe slug from name"""
        slug = name.lower().replace(" ", "-").replace("_", "-")
        # Remove special characters
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        # Add random suffix to ensure uniqueness
        suffix = secrets.token_hex(4)
        return f"{slug}-{suffix}"
    
    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        return f"sk_{secrets.token_urlsafe(32)}"
    
    def _generate_encryption_key(self) -> str:
        """Generate per-tenant encryption key"""
        return secrets.token_urlsafe(32)
    
    def _get_plan_quota(self, plan: SubscriptionPlan) -> ResourceQuota:
        """Get resource quota for plan"""
        quotas = {
            SubscriptionPlan.FREE: ResourceQuota(
                max_bots=1,
                max_positions=5,
                max_daily_trades=50,
                api_calls_per_day=1000,
                storage_gb=0.5,
                cpu_cores=0.5,
                memory_gb=1.0
            ),
            SubscriptionPlan.STARTER: ResourceQuota(
                max_bots=3,
                max_positions=10,
                max_daily_trades=200,
                api_calls_per_day=10000,
                storage_gb=1,
                cpu_cores=1.0,
                memory_gb=2.0
            ),
            SubscriptionPlan.PROFESSIONAL: ResourceQuota(
                max_bots=10,
                max_positions=50,
                max_daily_trades=1000,
                api_calls_per_day=100000,
                storage_gb=10,
                cpu_cores=2.0,
                memory_gb=4.0
            ),
            SubscriptionPlan.ENTERPRISE: ResourceQuota(
                max_bots=999,
                max_positions=999,
                max_daily_trades=999999,
                api_calls_per_day=999999,
                storage_gb=100,
                cpu_cores=8.0,
                memory_gb=16.0
            )
        }
        return quotas.get(plan, ResourceQuota())
    
    def _save_tenant_to_master(self, tenant: Tenant) -> None:
        """Save tenant record to master database"""
        with self.Session() as session:
            query = text("""
                INSERT INTO tenants (
                    tenant_id, name, slug, email, plan, status,
                    schema_name, encryption_key, api_key, quota,
                    trial_ends_at, created_at, updated_at
                ) VALUES (
                    :tenant_id, :name, :slug, :email, :plan, :status,
                    :schema_name, :encryption_key, :api_key, :quota,
                    :trial_ends_at, :created_at, :updated_at
                )
            """)
            
            session.execute(query, {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "slug": tenant.slug,
                "email": tenant.email,
                "plan": tenant.plan.value,
                "status": tenant.status.value,
                "schema_name": tenant.schema_name,
                "encryption_key": tenant.encryption_key,
                "api_key": tenant.api_key,
                "quota": tenant.quota.__dict__,
                "trial_ends_at": tenant.trial_ends_at,
                "created_at": tenant.created_at,
                "updated_at": tenant.updated_at
            })
            session.commit()
    
    def _create_tenant_schema(self, tenant: Tenant) -> None:
        """Create isolated PostgreSQL schema for tenant"""
        with self.master_engine.connect() as conn:
            # Create schema
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {tenant.schema_name}"))
            conn.commit()
            
            logger.info(f"Created schema: {tenant.schema_name}")
    
    def _initialize_tenant_tables(self, tenant: Tenant) -> None:
        """Create tables in tenant schema"""
        with self.master_engine.connect() as conn:
            # Set search path to tenant schema
            conn.execute(text(f"SET search_path TO {tenant.schema_name}"))
            
            # Create tenant tables
            tables_sql = """
            CREATE TABLE trades (
                trade_id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                bot_id VARCHAR(50),
                pnl DECIMAL(20, 2)
            );
            
            CREATE TABLE positions (
                position_id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL UNIQUE,
                quantity DECIMAL(20, 8) NOT NULL,
                entry_price DECIMAL(20, 8) NOT NULL,
                current_price DECIMAL(20, 8),
                unrealized_pnl DECIMAL(20, 2),
                opened_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            
            CREATE TABLE orders (
                order_id VARCHAR(50) PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8),
                order_type VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                filled_at TIMESTAMP
            );
            
            CREATE TABLE strategies (
                strategy_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                code TEXT,
                parameters JSONB,
                status VARCHAR(20) NOT NULL DEFAULT 'inactive',
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            
            CREATE TABLE bots (
                bot_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                strategy_id VARCHAR(50) REFERENCES strategies(strategy_id),
                status VARCHAR(20) NOT NULL DEFAULT 'stopped',
                config JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX idx_trades_symbol ON trades(symbol);
            CREATE INDEX idx_positions_symbol ON positions(symbol);
            CREATE INDEX idx_orders_status ON orders(status);
            """
            
            conn.execute(text(tables_sql))
            conn.commit()
            
            logger.info(f"Initialized tables in {tenant.schema_name}")
    
    def _setup_resource_quotas(self, tenant: Tenant) -> None:
        """Configure resource limits for tenant"""
        # In production, this would configure:
        # - Kubernetes resource quotas
        # - Rate limiting rules in Redis
        # - Storage quotas
        pass
    
    def _create_default_resources(self, tenant: Tenant) -> None:
        """Create default resources for new tenant"""
        # Create default strategy
        # Set up default risk parameters
        # Initialize demo data (optional)
        pass
    
    def _update_tenant_status(self, tenant_id: str, status: TenantStatus) -> None:
        """Update tenant status"""
        with self.Session() as session:
            query = text("""
                UPDATE tenants 
                SET status = :status, updated_at = :updated_at
                WHERE tenant_id = :tenant_id
            """)
            session.execute(query, {
                "status": status.value,
                "updated_at": datetime.utcnow(),
                "tenant_id": tenant_id
            })
            session.commit()
    
    def _drop_tenant_schema(self, tenant: Tenant) -> None:
        """Drop tenant schema (hard delete)"""
        with self.master_engine.connect() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {tenant.schema_name} CASCADE"))
            conn.commit()
            logger.warning(f"Dropped schema: {tenant.schema_name}")
    
    def _delete_tenant_from_master(self, tenant_id: str) -> None:
        """Delete tenant record from master database"""
        with self.Session() as session:
            query = text("DELETE FROM tenants WHERE tenant_id = :tenant_id")
            session.execute(query, {"tenant_id": tenant_id})
            session.commit()
    
    def _rollback_tenant_creation(self, tenant: Tenant) -> None:
        """Rollback tenant creation on error"""
        try:
            self._drop_tenant_schema(tenant)
            self._delete_tenant_from_master(tenant.tenant_id)
        except:
            pass
    
    def _row_to_tenant(self, row) -> Tenant:
        """Convert database row to Tenant object"""
        # This would parse the row into a Tenant object
        # Simplified for example
        return Tenant(
            tenant_id=row.tenant_id,
            name=row.name,
            slug=row.slug,
            email=row.email,
            plan=SubscriptionPlan(row.plan),
            status=TenantStatus(row.status),
            schema_name=row.schema_name,
            encryption_key=row.encryption_key,
            api_key=row.api_key
        )


# Example usage
if __name__ == "__main__":
    # Initialize tenant manager
    manager = TenantManager("postgresql://user:pass@localhost:5432/trading_master")
    
    # Create new tenant
    tenant = manager.create_tenant(
        name="Acme Trading Corp",
        email="admin@acme.com",
        plan=SubscriptionPlan.PROFESSIONAL,
        trial_days=14
    )
    
    print(f"✓ Tenant created: {tenant.tenant_id}")
    print(f"  API Key: {tenant.api_key}")
    print(f"  Schema: {tenant.schema_name}")
    print(f"  Trial ends: {tenant.trial_ends_at}")
