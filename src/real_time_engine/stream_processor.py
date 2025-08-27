import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import redis
from pydantic import BaseModel, Field
import websockets
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import time

from src.core.exceptions import ExternalAPIException
from src.core.logging import logger
from src.core.config import settings


@dataclass
class MarketDataPoint:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    volatility: Optional[float] = None


@dataclass  
class ProcessedSignal:
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold', 'alert'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    features: Dict[str, float]
    metadata: Dict[str, Any]


class StreamConfig(BaseModel):
    kafka_topics: List[str] = Field(default=["market-data", "trades", "news"])
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_group_id: str = Field(default="nqfs-stream-processor")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    buffer_size: int = Field(default=10000)
    processing_interval: float = Field(default=0.1)  # seconds
    window_size: int = Field(default=100)  # data points
    enable_real_time_ml: bool = Field(default=True)
    websocket_port: int = Field(default=8765)
    max_connections: int = Field(default=1000)


class StreamProcessor:
    """High-performance real-time data stream processor"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logger.bind(component="StreamProcessor")
        
        # Data buffers
        self.data_buffer = Queue(maxsize=config.buffer_size)
        self.processed_buffer = Queue(maxsize=config.buffer_size)
        
        # Kafka setup
        self.kafka_consumer = None
        self.kafka_producer = None
        
        # Redis setup  
        self.redis_client = None
        
        # WebSocket connections
        self.websocket_clients = set()
        
        # Processing state
        self.is_running = False
        self.processing_threads = []
        self.stats = {
            'messages_processed': 0,
            'signals_generated': 0,
            'errors': 0,
            'start_time': None,
            'last_update': None
        }
        
        # Callbacks for different data types
        self.callbacks = {
            'market_data': [],
            'trade_signal': [],
            'risk_alert': [],
            'news_event': []
        }
        
        # Real-time models (lightweight for stream processing)
        self.streaming_models = {}
        
    async def initialize(self):
        """Initialize all streaming components"""
        try:
            self.logger.info("Initializing stream processor")
            
            # Initialize Kafka
            await self._init_kafka()
            
            # Initialize Redis
            await self._init_redis()
            
            # Initialize real-time ML models
            if self.config.enable_real_time_ml:
                await self._init_streaming_models()
            
            self.logger.info("Stream processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Stream processor initialization failed: {e}")
            raise ExternalAPIException(f"Stream processor initialization failed: {e}")
    
    async def _init_kafka(self):
        """Initialize Kafka consumer and producer"""
        try:
            self.kafka_consumer = KafkaConsumer(
                *self.config.kafka_topics,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
            )
            
            self.logger.info("Kafka initialized")
            
        except Exception as e:
            self.logger.error(f"Kafka initialization failed: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            
            self.logger.info("Redis initialized")
            
        except Exception as e:
            self.logger.error(f"Redis initialization failed: {e}")
            raise
    
    async def _init_streaming_models(self):
        """Initialize lightweight models for real-time processing"""
        try:
            # Simple moving average model
            self.streaming_models['sma'] = {
                'short_window': 5,
                'long_window': 20,
                'data': {}
            }
            
            # Momentum model
            self.streaming_models['momentum'] = {
                'window': 10,
                'data': {}
            }
            
            # Volatility model
            self.streaming_models['volatility'] = {
                'window': 30,
                'data': {}
            }
            
            # Anomaly detection (simple z-score)
            self.streaming_models['anomaly'] = {
                'window': 100,
                'threshold': 3.0,
                'data': {}
            }
            
            self.logger.info("Streaming models initialized")
            
        except Exception as e:
            self.logger.error(f"Streaming models initialization failed: {e}")
            raise
    
    async def start_processing(self):
        """Start the real-time processing pipeline"""
        if self.is_running:
            self.logger.warning("Stream processor already running")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        self.logger.info("Starting stream processing pipeline")
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._kafka_consumer_thread, daemon=True),
            threading.Thread(target=self._data_processor_thread, daemon=True),
            threading.Thread(target=self._signal_publisher_thread, daemon=True),
            threading.Thread(target=self._stats_updater_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            self.processing_threads.append(thread)
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        self.logger.info("Stream processing pipeline started")
    
    def _kafka_consumer_thread(self):
        """Thread for consuming Kafka messages"""
        while self.is_running:
            try:
                message_pack = self.kafka_consumer.poll(timeout_ms=100)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        try:
                            # Parse message
                            data_point = self._parse_message(message.topic, message.value)
                            
                            if data_point:
                                # Add to processing buffer
                                if not self.data_buffer.full():
                                    self.data_buffer.put(data_point)
                                else:
                                    self.logger.warning("Data buffer full, dropping message")
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing Kafka message: {e}")
                            self.stats['errors'] += 1
                            
            except Exception as e:
                self.logger.error(f"Kafka consumer error: {e}")
                time.sleep(1)  # Brief pause before retry
    
    def _parse_message(self, topic: str, message_data: Dict) -> Optional[MarketDataPoint]:
        """Parse Kafka message into MarketDataPoint"""
        try:
            if topic == "market-data":
                return MarketDataPoint(
                    symbol=message_data.get('symbol', ''),
                    timestamp=datetime.fromisoformat(message_data.get('timestamp', datetime.now().isoformat())),
                    price=float(message_data.get('price', 0)),
                    volume=float(message_data.get('volume', 0)),
                    bid=message_data.get('bid'),
                    ask=message_data.get('ask'),
                    spread=message_data.get('spread'),
                    volatility=message_data.get('volatility')
                )
            
            # Handle other message types (trades, news, etc.)
            return None
            
        except Exception as e:
            self.logger.error(f"Message parsing error: {e}")
            return None
    
    def _data_processor_thread(self):
        """Main data processing thread"""
        while self.is_running:
            try:
                if not self.data_buffer.empty():
                    data_point = self.data_buffer.get()
                    
                    # Process the data point
                    signals = self._process_data_point(data_point)
                    
                    # Store in Redis for quick access
                    self._store_in_redis(data_point)
                    
                    # Add generated signals to output buffer
                    for signal in signals:
                        if not self.processed_buffer.full():
                            self.processed_buffer.put(signal)
                    
                    # Update stats
                    self.stats['messages_processed'] += 1
                    self.stats['signals_generated'] += len(signals)
                    self.stats['last_update'] = datetime.now()
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('market_data', data_point)
                    for signal in signals:
                        await self._trigger_callbacks('trade_signal', signal)
                
                else:
                    time.sleep(self.config.processing_interval)
                    
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                self.stats['errors'] += 1
    
    def _process_data_point(self, data_point: MarketDataPoint) -> List[ProcessedSignal]:
        """Process a single data point and generate signals"""
        signals = []
        
        try:
            symbol = data_point.symbol
            
            # Update streaming models
            self._update_streaming_models(data_point)
            
            # Generate signals from models
            
            # 1. Moving Average Crossover
            sma_signal = self._generate_sma_signal(symbol, data_point)
            if sma_signal:
                signals.append(sma_signal)
            
            # 2. Momentum Signal
            momentum_signal = self._generate_momentum_signal(symbol, data_point)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # 3. Volatility Signal
            vol_signal = self._generate_volatility_signal(symbol, data_point)
            if vol_signal:
                signals.append(vol_signal)
            
            # 4. Anomaly Detection
            anomaly_signal = self._generate_anomaly_signal(symbol, data_point)
            if anomaly_signal:
                signals.append(anomaly_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return []
    
    def _update_streaming_models(self, data_point: MarketDataPoint):
        """Update streaming model data structures"""
        symbol = data_point.symbol
        
        # Initialize symbol data if not exists
        for model_name, model_data in self.streaming_models.items():
            if symbol not in model_data['data']:
                model_data['data'][symbol] = {
                    'prices': [],
                    'volumes': [],
                    'timestamps': []
                }
        
        # Add new data point
        for model_name, model_data in self.streaming_models.items():
            symbol_data = model_data['data'][symbol]
            
            symbol_data['prices'].append(data_point.price)
            symbol_data['volumes'].append(data_point.volume)
            symbol_data['timestamps'].append(data_point.timestamp)
            
            # Maintain window size
            window_size = model_data.get('window', 100)
            if len(symbol_data['prices']) > window_size:
                symbol_data['prices'] = symbol_data['prices'][-window_size:]
                symbol_data['volumes'] = symbol_data['volumes'][-window_size:]
                symbol_data['timestamps'] = symbol_data['timestamps'][-window_size:]
    
    def _generate_sma_signal(self, symbol: str, data_point: MarketDataPoint) -> Optional[ProcessedSignal]:
        """Generate Simple Moving Average crossover signal"""
        try:
            model = self.streaming_models['sma']
            symbol_data = model['data'].get(symbol, {})
            prices = symbol_data.get('prices', [])
            
            if len(prices) < model['long_window']:
                return None
            
            short_sma = np.mean(prices[-model['short_window']:])
            long_sma = np.mean(prices[-model['long_window']:])
            prev_short_sma = np.mean(prices[-model['short_window']-1:-1])
            prev_long_sma = np.mean(prices[-model['long_window']-1:-1])
            
            # Detect crossover
            signal_type = None
            strength = 0.0
            
            if prev_short_sma <= prev_long_sma and short_sma > long_sma:
                signal_type = 'buy'
                strength = min(1.0, (short_sma - long_sma) / long_sma)
            elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
                signal_type = 'sell'
                strength = min(1.0, (long_sma - short_sma) / long_sma)
            
            if signal_type:
                return ProcessedSignal(
                    signal_id=f"sma_{symbol}_{int(data_point.timestamp.timestamp())}",
                    timestamp=data_point.timestamp,
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=abs(strength),
                    confidence=0.7,  # Medium confidence for SMA
                    features={
                        'short_sma': short_sma,
                        'long_sma': long_sma,
                        'current_price': data_point.price
                    },
                    metadata={'model': 'sma_crossover'}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"SMA signal generation error: {e}")
            return None
    
    def _generate_momentum_signal(self, symbol: str, data_point: MarketDataPoint) -> Optional[ProcessedSignal]:
        """Generate momentum-based signal"""
        try:
            model = self.streaming_models['momentum']
            symbol_data = model['data'].get(symbol, {})
            prices = symbol_data.get('prices', [])
            
            if len(prices) < model['window']:
                return None
            
            # Calculate momentum (rate of change)
            current_price = prices[-1]
            old_price = prices[-model['window']]
            momentum = (current_price - old_price) / old_price
            
            # Generate signal based on momentum threshold
            signal_type = None
            strength = abs(momentum)
            
            if momentum > 0.02:  # 2% positive momentum
                signal_type = 'buy'
            elif momentum < -0.02:  # 2% negative momentum
                signal_type = 'sell'
            
            if signal_type and strength > 0.01:  # Minimum strength threshold
                return ProcessedSignal(
                    signal_id=f"momentum_{symbol}_{int(data_point.timestamp.timestamp())}",
                    timestamp=data_point.timestamp,
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(1.0, strength * 10),  # Scale strength
                    confidence=0.6,
                    features={
                        'momentum': momentum,
                        'price_change': current_price - old_price,
                        'current_price': current_price
                    },
                    metadata={'model': 'momentum'}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Momentum signal generation error: {e}")
            return None
    
    def _generate_volatility_signal(self, symbol: str, data_point: MarketDataPoint) -> Optional[ProcessedSignal]:
        """Generate volatility-based signal"""
        try:
            model = self.streaming_models['volatility']
            symbol_data = model['data'].get(symbol, {})
            prices = symbol_data.get('prices', [])
            
            if len(prices) < model['window']:
                return None
            
            # Calculate returns and volatility
            returns = np.diff(np.log(prices))
            current_vol = np.std(returns)
            
            # Compare with historical volatility
            if len(prices) >= model['window'] * 2:
                historical_returns = np.diff(np.log(prices[:-model['window']//2]))
                historical_vol = np.std(historical_returns)
                
                vol_ratio = current_vol / (historical_vol + 1e-8)
                
                # Generate volatility regime signal
                if vol_ratio > 1.5:  # High volatility regime
                    return ProcessedSignal(
                        signal_id=f"vol_{symbol}_{int(data_point.timestamp.timestamp())}",
                        timestamp=data_point.timestamp,
                        symbol=symbol,
                        signal_type='alert',
                        strength=min(1.0, vol_ratio - 1.0),
                        confidence=0.8,
                        features={
                            'current_volatility': current_vol,
                            'historical_volatility': historical_vol,
                            'volatility_ratio': vol_ratio
                        },
                        metadata={'model': 'volatility', 'alert_type': 'high_volatility'}
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Volatility signal generation error: {e}")
            return None
    
    def _generate_anomaly_signal(self, symbol: str, data_point: MarketDataPoint) -> Optional[ProcessedSignal]:
        """Generate anomaly detection signal"""
        try:
            model = self.streaming_models['anomaly']
            symbol_data = model['data'].get(symbol, {})
            prices = symbol_data.get('prices', [])
            
            if len(prices) < model['window']:
                return None
            
            # Calculate z-score for current price
            recent_prices = prices[-model['window']:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            if std_price > 0:
                z_score = abs(data_point.price - mean_price) / std_price
                
                if z_score > model['threshold']:
                    signal_type = 'buy' if data_point.price > mean_price else 'sell'
                    
                    return ProcessedSignal(
                        signal_id=f"anomaly_{symbol}_{int(data_point.timestamp.timestamp())}",
                        timestamp=data_point.timestamp,
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=min(1.0, (z_score - model['threshold']) / model['threshold']),
                        confidence=0.9,  # High confidence for anomalies
                        features={
                            'z_score': z_score,
                            'mean_price': mean_price,
                            'std_price': std_price,
                            'current_price': data_point.price
                        },
                        metadata={'model': 'anomaly_detection', 'alert_type': 'price_anomaly'}
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Anomaly signal generation error: {e}")
            return None
    
    def _store_in_redis(self, data_point: MarketDataPoint):
        """Store data point in Redis for quick access"""
        try:
            # Store latest price
            self.redis_client.hset(
                f"latest_price:{data_point.symbol}",
                mapping={
                    'price': data_point.price,
                    'volume': data_point.volume,
                    'timestamp': data_point.timestamp.isoformat(),
                    'bid': data_point.bid or 0,
                    'ask': data_point.ask or 0
                }
            )
            
            # Store in time series (keep last 1000 points)
            self.redis_client.lpush(
                f"price_series:{data_point.symbol}",
                json.dumps(asdict(data_point), default=str)
            )
            self.redis_client.ltrim(f"price_series:{data_point.symbol}", 0, 999)
            
        except Exception as e:
            self.logger.error(f"Redis storage error: {e}")
    
    def _signal_publisher_thread(self):
        """Thread for publishing processed signals"""
        while self.is_running:
            try:
                if not self.processed_buffer.empty():
                    signal = self.processed_buffer.get()
                    
                    # Publish to Kafka
                    self.kafka_producer.send(
                        'trading-signals',
                        value=asdict(signal)
                    )
                    
                    # Broadcast to WebSocket clients
                    asyncio.run(self._broadcast_signal(signal))
                    
                    # Store signal in Redis
                    self._store_signal_in_redis(signal)
                
                else:
                    time.sleep(0.01)  # Short sleep when no signals
                    
            except Exception as e:
                self.logger.error(f"Signal publishing error: {e}")
    
    def _store_signal_in_redis(self, signal: ProcessedSignal):
        """Store signal in Redis"""
        try:
            # Store latest signal for symbol
            self.redis_client.hset(
                f"latest_signal:{signal.symbol}",
                mapping={
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'timestamp': signal.timestamp.isoformat()
                }
            )
            
            # Store in signal history
            self.redis_client.lpush(
                f"signal_history:{signal.symbol}",
                json.dumps(asdict(signal), default=str)
            )
            self.redis_client.ltrim(f"signal_history:{signal.symbol}", 0, 99)
            
        except Exception as e:
            self.logger.error(f"Signal storage error: {e}")
    
    async def _broadcast_signal(self, signal: ProcessedSignal):
        """Broadcast signal to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = json.dumps(asdict(signal), default=str)
        
        # Create tasks for all clients
        tasks = []
        for client in self.websocket_clients.copy():  # Copy to avoid modification during iteration
            tasks.append(self._send_to_client(client, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_client(self, client, message):
        """Send message to individual WebSocket client"""
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            self.websocket_clients.discard(client)
        except Exception as e:
            self.logger.error(f"WebSocket send error: {e}")
            self.websocket_clients.discard(client)
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time data streaming"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            self.logger.info(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.discard(websocket)
                self.logger.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
        
        start_server = websockets.serve(
            handle_client,
            "localhost",
            self.config.websocket_port,
            max_size=1024*1024,  # 1MB max message size
            max_queue=100
        )
        
        await start_server
        self.logger.info(f"WebSocket server started on port {self.config.websocket_port}")
    
    def _stats_updater_thread(self):
        """Thread for updating and logging statistics"""
        while self.is_running:
            try:
                if self.stats['start_time']:
                    runtime = datetime.now() - self.stats['start_time']
                    messages_per_sec = self.stats['messages_processed'] / max(runtime.total_seconds(), 1)
                    
                    self.logger.info(
                        f"Stream Stats - Messages: {self.stats['messages_processed']}, "
                        f"Signals: {self.stats['signals_generated']}, "
                        f"Rate: {messages_per_sec:.2f} msg/sec, "
                        f"Errors: {self.stats['errors']}, "
                        f"WS Clients: {len(self.websocket_clients)}"
                    )
                
                time.sleep(30)  # Update stats every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Stats update error: {e}")
    
    async def _trigger_callbacks(self, event_type: str, data):
        """Trigger registered callbacks for specific event types"""
        callbacks = self.callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event_type}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest data for symbol from Redis"""
        try:
            data = self.redis_client.hgetall(f"latest_price:{symbol}")
            if data:
                return {
                    'symbol': symbol,
                    'price': float(data.get('price', 0)),
                    'volume': float(data.get('volume', 0)),
                    'timestamp': data.get('timestamp'),
                    'bid': float(data.get('bid', 0)),
                    'ask': float(data.get('ask', 0))
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return None
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get price history for symbol from Redis"""
        try:
            history_data = self.redis_client.lrange(f"price_series:{symbol}", 0, limit-1)
            
            history = []
            for item in history_data:
                data_point = json.loads(item)
                history.append(data_point)
            
            return history
        except Exception as e:
            self.logger.error(f"Error getting price history: {e}")
            return []
    
    def get_signal_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get signal history for symbol from Redis"""
        try:
            signal_data = self.redis_client.lrange(f"signal_history:{symbol}", 0, limit-1)
            
            signals = []
            for item in signal_data:
                signal = json.loads(item)
                signals.append(signal)
            
            return signals
        except Exception as e:
            self.logger.error(f"Error getting signal history: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime = datetime.now() - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()
            stats['messages_per_second'] = stats['messages_processed'] / max(runtime.total_seconds(), 1)
            stats['signals_per_second'] = stats['signals_generated'] / max(runtime.total_seconds(), 1)
        
        stats['websocket_clients'] = len(self.websocket_clients)
        stats['buffer_sizes'] = {
            'data_buffer': self.data_buffer.qsize(),
            'processed_buffer': self.processed_buffer.qsize()
        }
        
        return stats
    
    async def stop_processing(self):
        """Stop the stream processing pipeline"""
        self.logger.info("Stopping stream processor")
        
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        # Close connections
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        # Close WebSocket connections
        if self.websocket_clients:
            for client in self.websocket_clients:
                await client.close()
        
        self.logger.info("Stream processor stopped")
