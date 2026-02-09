"""
Local-Cloud Sync Agent
Lightweight agent for running trading bot locally while syncing to cloud dashboard

Features:
- Local execution (low latency)
- Cloud status reporting
- WebSocket connection to cloud
- Heartbeat monitoring
- Offline capability with later sync
- Resource-efficient

Usage:
    python main.py --tenant-id acme-corp --api-key sk_xxx --cloud-url wss://api.platform.com
"""

import asyncio
import json
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import websockets
from loguru import logger
import sqlite3
import signal
import sys


class AgentStatus(Enum):
    """Sync agent status"""
    STARTING = "starting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SYNCING = "syncing"
    ERROR = "error"


@dataclass
class SyncMessage:
    """Message for cloud sync"""
    type: str  # 'status', 'trade', 'position', 'pnl', 'heartbeat'
    tenant_id: str
    timestamp: datetime
    data: Dict
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    
    def to_json(self) -> str:
        """Convert to JSON for transmission"""
        return json.dumps({
            'type': self.type,
            'tenant_id': self.tenant_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'message_id': self.message_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SyncMessage':
        """Create from JSON"""
        data = json.loads(json_str)
        return cls(
            type=data['type'],
            tenant_id=data['tenant_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            message_id=data['message_id']
        )


class LocalDatabase:
    """Lightweight SQLite database for local state"""
    
    def __init__(self, db_path: str = "local_state.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize local database tables"""
        cursor = self.conn.cursor()
        
        # Pending messages (not yet synced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_messages (
                message_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Local trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Local positions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sync status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_status (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def save_message(self, message: SyncMessage):
        """Save message for later sync"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO pending_messages (message_id, type, data)
            VALUES (?, ?, ?)
        """, (message.message_id, message.type, json.dumps(message.data)))
        self.conn.commit()
    
    def get_pending_messages(self) -> list:
        """Get messages not yet synced"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message_id, type, data FROM pending_messages 
            WHERE synced = FALSE
            ORDER BY created_at ASC
        """)
        return cursor.fetchall()
    
    def mark_synced(self, message_id: str):
        """Mark message as synced"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE pending_messages SET synced = TRUE
            WHERE message_id = ?
        """, (message_id,))
        self.conn.commit()
    
    def save_trade(self, trade: Dict):
        """Save local trade"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO local_trades (trade_id, symbol, side, quantity, price)
            VALUES (?, ?, ?, ?, ?)
        """, (
            trade['trade_id'],
            trade['symbol'],
            trade['side'],
            trade['quantity'],
            trade['price']
        ))
        self.conn.commit()
    
    def update_position(self, symbol: str, position: Dict):
        """Update local position"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO local_positions 
            (symbol, quantity, entry_price, current_price, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?)
        """, (
            symbol,
            position['quantity'],
            position['entry_price'],
            position.get('current_price'),
            position.get('unrealized_pnl')
        ))
        self.conn.commit()
    
    def get_status(self, key: str) -> Optional[str]:
        """Get status value"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM sync_status WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def set_status(self, key: str, value: str):
        """Set status value"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sync_status (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class CloudConnector:
    """WebSocket connection to cloud platform"""
    
    def __init__(
        self,
        cloud_url: str,
        tenant_id: str,
        api_key: str
    ):
        self.cloud_url = cloud_url
        self.tenant_id = tenant_id
        self.api_key = api_key
        self.websocket = None
        self.connected = False
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_delay = 60
        
    async def connect(self) -> bool:
        """Connect to cloud WebSocket"""
        try:
            # Authenticate with API key
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'X-Tenant-ID': self.tenant_id
            }
            
            self.websocket = await websockets.connect(
                self.cloud_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.connected = True
            logger.info(f"✓ Connected to cloud: {self.cloud_url}")
            
            # Send initial handshake
            await self.send_message(SyncMessage(
                type='handshake',
                tenant_id=self.tenant_id,
                timestamp=datetime.utcnow(),
                data={'version': '1.0.0', 'agent': 'local-sync'}
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to cloud: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from cloud"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
        logger.info("Disconnected from cloud")
    
    async def send_message(self, message: SyncMessage):
        """Send message to cloud"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to cloud")
        
        await self.websocket.send(message.to_json())
    
    async def receive_message(self) -> Optional[SyncMessage]:
        """Receive message from cloud"""
        if not self.connected or not self.websocket:
            return None
        
        try:
            message_str = await self.websocket.recv()
            return SyncMessage.from_json(message_str)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            return None
    
    async def reconnect_loop(self):
        """Auto-reconnect with exponential backoff"""
        delay = self.reconnect_delay
        
        while True:
            if not self.connected:
                logger.info(f"Attempting to reconnect in {delay}s...")
                await asyncio.sleep(delay)
                
                if await self.connect():
                    delay = self.reconnect_delay  # Reset delay on success
                else:
                    # Exponential backoff
                    delay = min(delay * 2, self.max_reconnect_delay)
            else:
                await asyncio.sleep(1)


class SyncAgent:
    """
    Main sync agent
    
    Runs trading bot locally and syncs status to cloud dashboard
    """
    
    def __init__(
        self,
        tenant_id: str,
        api_key: str,
        cloud_url: str,
        local_db_path: str = "local_state.db"
    ):
        self.tenant_id = tenant_id
        self.status = AgentStatus.STARTING
        
        # Components
        self.local_db = LocalDatabase(local_db_path)
        self.cloud = CloudConnector(cloud_url, tenant_id, api_key)
        
        # State
        self.heartbeat_interval = 30  # seconds
        self.sync_interval = 5  # seconds
        
        # Callbacks
        self.on_cloud_message: Optional[Callable] = None
        
        # Graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Sync Agent initialized for tenant: {tenant_id}")
    
    async def start(self):
        """Start sync agent"""
        logger.info("Starting sync agent...")
        
        # Connect to cloud
        if await self.cloud.connect():
            self.status = AgentStatus.CONNECTED
        else:
            self.status = AgentStatus.DISCONNECTED
            logger.warning("Starting in offline mode")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._sync_loop()),
            asyncio.create_task(self._message_receiver()),
            asyncio.create_task(self.cloud.reconnect_loop())
        ]
        
        logger.info("✓ Sync agent started")
        
        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled, shutting down...")
    
    async def stop(self):
        """Stop sync agent gracefully"""
        logger.info("Stopping sync agent...")
        
        self.running = False
        
        # Disconnect from cloud
        await self.cloud.disconnect()
        
        # Close database
        self.local_db.close()
        
        logger.info("✓ Sync agent stopped")
    
    async def report_trade(self, trade: Dict):
        """Report trade execution"""
        # Save locally
        self.local_db.save_trade(trade)
        
        # Send to cloud
        message = SyncMessage(
            type='trade',
            tenant_id=self.tenant_id,
            timestamp=datetime.utcnow(),
            data=trade
        )
        
        await self._send_or_queue(message)
    
    async def report_position(self, symbol: str, position: Dict):
        """Report position update"""
        # Save locally
        self.local_db.update_position(symbol, position)
        
        # Send to cloud
        message = SyncMessage(
            type='position',
            tenant_id=self.tenant_id,
            timestamp=datetime.utcnow(),
            data={'symbol': symbol, **position}
        )
        
        await self._send_or_queue(message)
    
    async def report_pnl(self, pnl_data: Dict):
        """Report P/L update"""
        message = SyncMessage(
            type='pnl',
            tenant_id=self.tenant_id,
            timestamp=datetime.utcnow(),
            data=pnl_data
        )
        
        await self._send_or_queue(message)
    
    async def report_status(self, status_data: Dict):
        """Report bot status"""
        message = SyncMessage(
            type='status',
            tenant_id=self.tenant_id,
            timestamp=datetime.utcnow(),
            data=status_data
        )
        
        await self._send_or_queue(message)
    
    async def _send_or_queue(self, message: SyncMessage):
        """Send message or queue if offline"""
        if self.cloud.connected:
            try:
                await self.cloud.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                self.local_db.save_message(message)
        else:
            # Queue for later sync
            self.local_db.save_message(message)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)
            
            if self.cloud.connected:
                try:
                    message = SyncMessage(
                        type='heartbeat',
                        tenant_id=self.tenant_id,
                        timestamp=datetime.utcnow(),
                        data={
                            'status': self.status.value,
                            'uptime': time.time()
                        }
                    )
                    await self.cloud.send_message(message)
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
    
    async def _sync_loop(self):
        """Sync pending messages"""
        while self.running:
            await asyncio.sleep(self.sync_interval)
            
            if self.cloud.connected:
                # Get pending messages
                pending = self.local_db.get_pending_messages()
                
                if pending:
                    self.status = AgentStatus.SYNCING
                    logger.info(f"Syncing {len(pending)} pending messages...")
                    
                    for msg_id, msg_type, msg_data in pending:
                        try:
                            message = SyncMessage(
                                type=msg_type,
                                tenant_id=self.tenant_id,
                                timestamp=datetime.utcnow(),
                                data=json.loads(msg_data),
                                message_id=msg_id
                            )
                            await self.cloud.send_message(message)
                            self.local_db.mark_synced(msg_id)
                        except Exception as e:
                            logger.error(f"Failed to sync message {msg_id}: {e}")
                            break
                    
                    self.status = AgentStatus.CONNECTED
                    logger.info("✓ Sync complete")
    
    async def _message_receiver(self):
        """Receive messages from cloud"""
        while self.running:
            if self.cloud.connected:
                try:
                    message = await self.cloud.receive_message()
                    if message:
                        await self._handle_cloud_message(message)
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
            
            await asyncio.sleep(0.1)
    
    async def _handle_cloud_message(self, message: SyncMessage):
        """Handle message from cloud"""
        logger.debug(f"Received cloud message: {message.type}")
        
        # Execute callback if registered
        if self.on_cloud_message:
            await self.on_cloud_message(message)
        
        # Handle specific message types
        if message.type == 'command':
            # Cloud commands (e.g., stop bot, change settings)
            await self._handle_command(message.data)
    
    async def _handle_command(self, command: Dict):
        """Handle command from cloud"""
        cmd_type = command.get('type')
        
        if cmd_type == 'stop_bot':
            logger.warning("Received STOP BOT command from cloud")
            # Stop local bot
        elif cmd_type == 'update_settings':
            logger.info("Received UPDATE SETTINGS command from cloud")
            # Update local settings
        else:
            logger.warning(f"Unknown command: {cmd_type}")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
        sys.exit(0)


# Example usage
async def main():
    """Example: Run sync agent"""
    
    # Configuration
    TENANT_ID = "acme-corp"
    API_KEY = "sk_test_123456789"
    CLOUD_URL = "wss://api.trading-platform.com/sync"
    
    # Create agent
    agent = SyncAgent(
        tenant_id=TENANT_ID,
        api_key=API_KEY,
        cloud_url=CLOUD_URL
    )
    
    # Define cloud message handler
    async def handle_cloud_message(message: SyncMessage):
        print(f"Cloud says: {message.type} - {message.data}")
    
    agent.on_cloud_message = handle_cloud_message
    
    # Start agent
    await agent.start()
    
    # Simulate trading activity
    while agent.running:
        await asyncio.sleep(10)
        
        # Report fake trade
        await agent.report_trade({
            'trade_id': f'trade_{int(time.time())}',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 50000.0
        })
        
        # Report P/L
        await agent.report_pnl({
            'daily_pnl': 123.45,
            'total_pnl': 567.89
        })


if __name__ == "__main__":
    asyncio.run(main())
