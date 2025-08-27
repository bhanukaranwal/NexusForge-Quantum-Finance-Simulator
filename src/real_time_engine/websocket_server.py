import asyncio
import json
import jwt
import time
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime, timedelta
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, InvalidState
from pydantic import BaseModel, Field
import uuid

from src.core.exceptions import AuthenticationException, RateLimitException
from src.core.logging import logger
from src.core.config import settings


class WebSocketConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8765)
    max_connections: int = Field(default=1000)
    max_message_size: int = Field(default=1024 * 1024)  # 1MB
    ping_interval: int = Field(default=20)  # seconds
    ping_timeout: int = Field(default=20)  # seconds
    rate_limit_messages: int = Field(default=100)  # per minute
    rate_limit_window: int = Field(default=60)  # seconds
    jwt_secret: str = Field(default=settings.jwt_secret_key)
    require_auth: bool = Field(default=True)


class ConnectionInfo(BaseModel):
    connection_id: str
    user_id: Optional[str] = None
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str] = Field(default_factory=set)
    message_count: int = Field(default=0)
    rate_limit_reset: datetime = Field(default_factory=datetime.now)


class WebSocketMessage(BaseModel):
    type: str  # subscribe, unsubscribe, request, response
    channel: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketServer:
    """High-performance WebSocket server for real-time financial data"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.logger = logger.bind(component="WebSocketServer")
        
        # Connection management
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # Channel subscriptions
        self.channel_subscriptions: Dict[str, Set[str]] = {}  # channel -> connection_ids
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'ping': self._handle_ping,
            'request': self._handle_request
        }
        
        # Custom handlers for different channels
        self.channel_handlers: Dict[str, Callable] = {}
        
        # Server instance
        self.server = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'connections_total': 0,
            'connections_current': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'start_time': None
        }
    
    async def start_server(self):
        """Start the WebSocket server"""
        if self.is_running:
            self.logger.warning("WebSocket server already running")
            return
        
        self.logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")
        
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.config.host,
                self.config.port,
                max_size=self.config.max_message_size,
                max_queue=100,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                compression=None  # Disable compression for lower latency
            )
            
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            self.logger.info("WebSocket server started successfully")
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._stats_task())
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping WebSocket server")
        
        self.is_running = False
        
        # Close all connections
        close_tasks = []
        for connection in self.connections.values():
            close_tasks.append(self._close_connection(connection, "Server shutdown"))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("WebSocket server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        # Check connection limit
        if len(self.connections) >= self.config.max_connections:
            await websocket.close(1008, "Server at maximum capacity")
            return
        
        try:
            # Authenticate if required
            if self.config.require_auth:
                user_id = await self._authenticate_connection(websocket)
                if not user_id:
                    await websocket.close(1008, "Authentication failed")
                    return
            else:
                user_id = None
            
            # Register connection
            self.connections[connection_id] = websocket
            self.connection_info[connection_id] = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                connected_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.stats['connections_total'] += 1
            self.stats['connections_current'] += 1
            
            self.logger.info(f"New connection: {connection_id} (user: {user_id})")
            
            # Send welcome message
            await self._send_message(connection_id, {
                'type': 'welcome',
                'connection_id': connection_id,
                'server_time': datetime.now().isoformat(),
                'available_channels': list(self.channel_handlers.keys())
            })
            
            # Handle messages
            await self._handle_messages(connection_id, websocket)
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _authenticate_connection(self, websocket: WebSocketServerProtocol) -> Optional[str]:
        """Authenticate WebSocket connection using JWT"""
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') != 'auth':
                return None
            
            token = auth_data.get('token')
            if not token:
                return None
            
            # Verify JWT token
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=['HS256']
            )
            
            user_id = payload.get('user_id')
            exp = payload.get('exp')
            
            # Check expiration
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                return None
            
            return user_id
            
        except asyncio.TimeoutError:
            self.logger.warning("Authentication timeout")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    async def _handle_messages(self, connection_id: str, websocket: WebSocketServerProtocol):
        """Handle incoming messages from connection"""
        try:
            async for raw_message in websocket:
                try:
                    # Update activity
                    self.connection_info[connection_id].last_activity = datetime.now()
                    
                    # Check rate limit
                    if not self._check_rate_limit(connection_id):
                        await self._send_error(connection_id, "Rate limit exceeded")
                        continue
                    
                    # Parse message
                    try:
                        message_data = json.loads(raw_message)
                        message = WebSocketMessage(**message_data)
                    except Exception as e:
                        await self._send_error(connection_id, f"Invalid message format: {e}")
                        continue
                    
                    # Handle message
                    await self._process_message(connection_id, message)
                    
                    self.stats['messages_received'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")
                    await self._send_error(connection_id, "Internal server error")
                    
        except ConnectionClosed:
            self.logger.info(f"Connection {connection_id} closed by client")
        except Exception as e:
            self.logger.error(f"Message loop error: {e}")
    
    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits"""
        conn_info = self.connection_info.get(connection_id)
        if not conn_info:
            return False
        
        now = datetime.now()
        
        # Reset counter if window expired
        if now >= conn_info.rate_limit_reset:
            conn_info.message_count = 0
            conn_info.rate_limit_reset = now + timedelta(seconds=self.config.rate_limit_window)
        
        # Check limit
        if conn_info.message_count >= self.config.rate_limit_messages:
            return False
        
        conn_info.message_count += 1
        return True
    
    async def _process_message(self, connection_id: str, message: WebSocketMessage):
        """Process incoming message"""
        handler = self.message_handlers.get(message.type)
        
        if handler:
            try:
                await handler(connection_id, message)
            except Exception as e:
                self.logger.error(f"Handler error for {message.type}: {e}")
                await self._send_error(connection_id, f"Handler error: {e}")
        else:
            await self._send_error(connection_id, f"Unknown message type: {message.type}")
    
    async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle channel subscription"""
        channel = message.channel
        if not channel:
            await self._send_error(connection_id, "Channel name required for subscription")
            return
        
        # Add to subscriptions
        if channel not in self.channel_subscriptions:
            self.channel_subscriptions[channel] = set()
        
        self.channel_subscriptions[channel].add(connection_id)
        self.connection_info[connection_id].subscriptions.add(channel)
        
        # Confirm subscription
        await self._send_message(connection_id, {
            'type': 'subscribed',
            'channel': channel,
            'request_id': message.request_id
        })
        
        self.logger.debug(f"Connection {connection_id} subscribed to {channel}")
        
        # Call custom channel handler if exists
        channel_handler = self.channel_handlers.get(channel)
        if channel_handler:
            try:
                await channel_handler('subscribe', connection_id, message.data)
            except Exception as e:
                self.logger.error(f"Channel handler error: {e}")
    
    async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle channel unsubscription"""
        channel = message.channel
        if not channel:
            await self._send_error(connection_id, "Channel name required for unsubscription")
            return
        
        # Remove from subscriptions
        if channel in self.channel_subscriptions:
            self.channel_subscriptions[channel].discard(connection_id)
            if not self.channel_subscriptions[channel]:
                del self.channel_subscriptions[channel]
        
        self.connection_info[connection_id].subscriptions.discard(channel)
        
        # Confirm unsubscription
        await self._send_message(connection_id, {
            'type': 'unsubscribed',
            'channel': channel,
            'request_id': message.request_id
        })
        
        self.logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
    
    async def _handle_ping(self, connection_id: str, message: WebSocketMessage):
        """Handle ping message"""
        await self._send_message(connection_id, {
            'type': 'pong',
            'request_id': message.request_id,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_request(self, connection_id: str, message: WebSocketMessage):
        """Handle custom request"""
        # This can be overridden or extended for custom request handling
        await self._send_error(connection_id, "Request handling not implemented")
    
    async def _send_message(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send message to specific connection"""
        websocket = self.connections.get(connection_id)
        if not websocket:
            return False
        
        try:
            message = json.dumps(data, default=str)
            await websocket.send(message)
            self.stats['messages_sent'] += 1
            return True
            
        except ConnectionClosed:
            await self._cleanup_connection(connection_id)
            return False
        except Exception as e:
            self.logger.error(f"Send message error: {e}")
            self.stats['errors'] += 1
            return False
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        await self._send_message(connection_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
    
    async def broadcast_to_channel(self, channel: str, data: Dict[str, Any]) -> int:
        """Broadcast message to all subscribers of a channel"""
        if channel not in self.channel_subscriptions:
            return 0
        
        subscribers = self.channel_subscriptions[channel].copy()
        message_data = {
            'type': 'data',
            'channel': channel,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all subscribers concurrently
        tasks = []
        for connection_id in subscribers:
            tasks.append(self._send_message(connection_id, message_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful sends
        successful = sum(1 for result in results if result is True)
        
        return successful
    
    async def send_to_user(self, user_id: str, data: Dict[str, Any]) -> int:
        """Send message to all connections of a specific user"""
        user_connections = [
            conn_id for conn_id, info in self.connection_info.items()
            if info.user_id == user_id
        ]
        
        if not user_connections:
            return 0
        
        message_data = {
            'type': 'user_message',
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all user connections
        tasks = []
        for connection_id in user_connections:
            tasks.append(self._send_message(connection_id, message_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for result in results if result is True)
        
        return successful
    
    def register_channel_handler(self, channel: str, handler: Callable):
        """Register custom handler for channel events"""
        self.channel_handlers[channel] = handler
        self.logger.info(f"Registered handler for channel: {channel}")
    
    def get_channel_subscribers(self, channel: str) -> Set[str]:
        """Get connection IDs subscribed to a channel"""
        return self.channel_subscriptions.get(channel, set()).copy()
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get connection IDs for a specific user"""
        return [
            conn_id for conn_id, info in self.connection_info.items()
            if info.user_id == user_id
        ]
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a connection"""
        return self.connection_info.get(connection_id)
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up connection resources"""
        try:
            # Remove from subscriptions
            conn_info = self.connection_info.get(connection_id)
            if conn_info:
                for channel in conn_info.subscriptions:
                    if channel in self.channel_subscriptions:
                        self.channel_subscriptions[channel].discard(connection_id)
                        if not self.channel_subscriptions[channel]:
                            del self.channel_subscriptions[channel]
            
            # Remove connection
            self.connections.pop(connection_id, None)
            self.connection_info.pop(connection_id, None)
            
            self.stats['connections_current'] -= 1
            
            self.logger.info(f"Cleaned up connection: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    async def _close_connection(self, websocket: WebSocketServerProtocol, reason: str):
        """Close WebSocket connection"""
        try:
            await websocket.close(1000, reason)
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
    
    async def _cleanup_task(self):
        """Background task to clean up stale connections"""
        while self.is_running:
            try:
                now = datetime.now()
                stale_connections = []
                
                for conn_id, info in self.connection_info.items():
                    # Check for inactive connections (no activity for 5 minutes)
                    if now - info.last_activity > timedelta(minutes=5):
                        stale_connections.append(conn_id)
                
                # Clean up stale connections
                for conn_id in stale_connections:
                    websocket = self.connections.get(conn_id)
                    if websocket:
                        await self._close_connection(websocket, "Inactive connection")
                    await self._cleanup_connection(conn_id)
                
                if stale_connections:
                    self.logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    async def _stats_task(self):
        """Background task to log statistics"""
        while self.is_running:
            try:
                self.logger.info(
                    f"WebSocket Stats - Current: {self.stats['connections_current']}, "
                    f"Total: {self.stats['connections_total']}, "
                    f"Sent: {self.stats['messages_sent']}, "
                    f"Received: {self.stats['messages_received']}, "
                    f"Errors: {self.stats['errors']}, "
                    f"Channels: {len(self.channel_subscriptions)}"
                )
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Stats task error: {e}")
                await asyncio.sleep(300)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current server statistics"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime = datetime.now() - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()
            stats['messages_per_second'] = {
                'sent': stats['messages_sent'] / max(runtime.total_seconds(), 1),
                'received': stats['messages_received'] / max(runtime.total_seconds(), 1)
            }
        
        stats['channels'] = {
            'total': len(self.channel_subscriptions),
            'subscribers': {
                channel: len(subs) 
                for channel, subs in self.channel_subscriptions.items()
            }
        }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get server health status"""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'connections': len(self.connections),
            'max_connections': self.config.max_connections,
            'cpu_usage': len(self.connections) / self.config.max_connections,
            'uptime_seconds': (
                (datetime.now() - self.stats['start_time']).total_seconds()
                if self.stats['start_time'] else 0
            ),
            'error_rate': (
                self.stats['errors'] / max(self.stats['messages_received'], 1)
                if self.stats['messages_received'] > 0 else 0
            )
        }
