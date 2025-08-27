import asyncio
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import json
from datetime import datetime, timedelta

from src.core.exceptions import MLModelException
from src.core.logging import logger


@dataclass
class Strategy:
    """Individual trading strategy representation"""
    id: str
    genes: List[float]  # Strategy parameters
    fitness: float = 0.0
    age: int = 0
    wins: int = 0
    losses: int = 0
    total_trades: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class NexusAIConfig(BaseModel):
    population_size: int = Field(default=50, description="Population size for genetic algorithm")
    elite_ratio: float = Field(default=0.2, description="Ratio of elite strategies to preserve")
    mutation_rate: float = Field(default=0.1, description="Mutation probability")
    crossover_rate: float = Field(default=0.8, description="Crossover probability")
    max_generations: int = Field(default=100, description="Maximum generations")
    strategy_genes: int = Field(default=20, description="Number of genes per strategy")
    fitness_window: int = Field(default=252, description="Window for fitness evaluation")
    min_trades: int = Field(default=10, description="Minimum trades for fitness calculation")
    convergence_threshold: float = Field(default=0.001, description="Convergence threshold")
    diversification_bonus: float = Field(default=0.1, description="Bonus for strategy diversification")
    adaptive_learning: bool = Field(default=True, description="Enable adaptive learning rate")
    multi_objective: bool = Field(default=True, description="Multi-objective optimization")
    memory_length: int = Field(default=1000, description="Memory length for experience replay")


class StrategyPerformance(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    trades_per_year: float
    risk_adjusted_return: float


class EvolutionaryEngine:
    """Core genetic algorithm engine for strategy evolution"""
    
    def __init__(self, config: NexusAIConfig):
        self.config = config
        self.population: List[Strategy] = []
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.logger = logger.bind(component="EvolutionaryEngine")
        
    def initialize_population(self) -> None:
        """Initialize random population of strategies"""
        self.population = []
        for i in range(self.config.population_size):
            genes = [random.uniform(-1, 1) for _ in range(self.config.strategy_genes)]
            strategy = Strategy(
                id=f"gen0_strategy_{i}",
                genes=genes
            )
            self.population.append(strategy)
        
        self.logger.info(f"Initialized population with {len(self.population)} strategies")
    
    def evaluate_fitness(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        """Evaluate strategy fitness on market data"""
        try:
            # Convert genes to trading parameters
            params = self._genes_to_parameters(strategy.genes)
            
            # Simulate trading strategy
            performance = self._simulate_strategy(params, market_data)
            
            # Multi-objective fitness calculation
            if self.config.multi_objective:
                fitness = self._calculate_multi_objective_fitness(performance)
            else:
                fitness = performance.sharpe_ratio
            
            # Apply diversification bonus
            fitness += self._calculate_diversification_bonus(strategy)
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed for strategy {strategy.id}: {e}")
            return -1000.0  # Severe penalty for failed strategies
    
    def _genes_to_parameters(self, genes: List[float]) -> Dict[str, float]:
        """Convert gene values to trading parameters"""
        params = {
            'ma_short_period': max(1, int(genes[0] * 50 + 50)),      # 1-100
            'ma_long_period': max(1, int(genes[1] * 150 + 150)),     # 1-300
            'rsi_period': max(1, int(genes[2] * 20 + 10)),           # 1-30
            'rsi_oversold': genes[3] * 20 + 30,                      # 10-50
            'rsi_overbought': genes[4] * 20 + 70,                    # 70-90
            'bb_period': max(1, int(genes[5] * 30 + 10)),            # 1-40
            'bb_std': genes[6] * 1.5 + 1.5,                         # 0.5-3.0
            'momentum_period': max(1, int(genes[7] * 20 + 5)),       # 1-25
            'volume_threshold': genes[8] * 2 + 1,                    # 0.5-2.5
            'stop_loss': abs(genes[9]) * 0.05 + 0.01,               # 1%-6%
            'take_profit': abs(genes[10]) * 0.1 + 0.02,             # 2%-12%
            'position_size': abs(genes[11]) * 0.8 + 0.1,            # 10%-90%
            'trend_strength': genes[12] * 0.5 + 0.5,                # 0.0-1.0
            'volatility_adj': abs(genes[13]) * 0.5 + 0.5,           # 0.5-1.0
            'regime_sensitivity': abs(genes[14]) * 0.8 + 0.2,       # 0.2-1.0
        }
        
        # Add additional parameters from remaining genes
        for i, gene in enumerate(genes[15:], 15):
            params[f'custom_param_{i}'] = gene
        
        return params
    
    def _simulate_strategy(self, params: Dict[str, float], data: pd.DataFrame) -> StrategyPerformance:
        """Simulate trading strategy with given parameters"""
        # Initialize simulation variables
        capital = 100000
        position = 0
        entry_price = 0
        trades = []
        portfolio_values = [capital]
        
        # Calculate technical indicators
        data = self._add_technical_indicators(data.copy(), params)
        
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            current_row = data.iloc[i]
            
            # Generate trading signal
            signal = self._generate_signal(current_row, params)
            
            # Execute trades based on signal
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    profit = (entry_price - current_price) * abs(position)
                    capital += profit
                    trades.append(profit)
                
                # Open long position
                position_value = capital * params['position_size']
                position = position_value / current_price
                entry_price = current_price
                capital -= position_value
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    profit = (current_price - entry_price) * position
                    capital += profit
                    trades.append(profit)
                
                # Open short position
                position_value = capital * params['position_size']
                position = -position_value / current_price
                entry_price = current_price
                capital += position_value
            
            # Apply stop loss and take profit
            if position != 0:
                if position > 0:  # Long position
                    if current_price <= entry_price * (1 - params['stop_loss']):
                        # Stop loss
                        profit = (current_price - entry_price) * position
                        capital += current_price * position
                        trades.append(profit)
                        position = 0
                    elif current_price >= entry_price * (1 + params['take_profit']):
                        # Take profit
                        profit = (current_price - entry_price) * position
                        capital += current_price * position
                        trades.append(profit)
                        position = 0
                
                elif position < 0:  # Short position
                    if current_price >= entry_price * (1 + params['stop_loss']):
                        # Stop loss
                        profit = (entry_price - current_price) * abs(position)
                        capital += profit
                        trades.append(profit)
                        position = 0
                    elif current_price <= entry_price * (1 - params['take_profit']):
                        # Take profit
                        profit = (entry_price - current_price) * abs(position)
                        capital += profit
                        trades.append(profit)
                        position = 0
            
            # Calculate portfolio value
            portfolio_value = capital
            if position != 0:
                portfolio_value += position * current_price
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        return self._calculate_performance(portfolio_values, trades)
    
    def _add_technical_indicators(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Add technical indicators to data"""
        # Moving averages
        data[f'ma_short'] = data['close'].rolling(window=int(params['ma_short_period'])).mean()
        data[f'ma_long'] = data['close'].rolling(window=int(params['ma_long_period'])).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=int(params['rsi_period'])).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=int(params['rsi_period'])).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = int(params['bb_period'])
        bb_std = params['bb_std']
        data['bb_middle'] = data['close'].rolling(window=bb_period).mean()
        data['bb_std'] = data['close'].rolling(window=bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (bb_std * data['bb_std'])
        
        # Momentum
        data['momentum'] = data['close'].pct_change(int(params['momentum_period']))
        
        # Volume ratio (if volume available)
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        else:
            data['volume_ratio'] = 1.0
        
        # Volatility
        data['volatility'] = data['close'].pct_change().rolling(window=20).std()
        
        return data
    
    def _generate_signal(self, row: pd.Series, params: Dict[str, float]) -> int:
        """Generate trading signal based on technical indicators"""
        signals = []
        
        # Moving average signal
        if pd.notna(row['ma_short']) and pd.notna(row['ma_long']):
            if row['ma_short'] > row['ma_long']:
                signals.append(1)
            else:
                signals.append(-1)
        
        # RSI signal
        if pd.notna(row['rsi']):
            if row['rsi'] < params['rsi_oversold']:
                signals.append(1)  # Oversold, buy
            elif row['rsi'] > params['rsi_overbought']:
                signals.append(-1)  # Overbought, sell
        
        # Bollinger Bands signal
        if pd.notna(row['bb_upper']) and pd.notna(row['bb_lower']):
            if row['close'] < row['bb_lower']:
                signals.append(1)  # Below lower band, buy
            elif row['close'] > row['bb_upper']:
                signals.append(-1)  # Above upper band, sell
        
        # Momentum signal
        if pd.notna(row['momentum']):
            momentum_threshold = 0.01 * params['trend_strength']
            if row['momentum'] > momentum_threshold:
                signals.append(1)
            elif row['momentum'] < -momentum_threshold:
                signals.append(-1)
        
        # Volume confirmation
        volume_threshold = params['volume_threshold']
        if row['volume_ratio'] < volume_threshold:
            # Low volume, reduce signal strength
            signals = [s * 0.5 for s in signals]
        
        # Aggregate signals
        if not signals:
            return 0
        
        signal_strength = sum(signals) / len(signals)
        
        if signal_strength > 0.3:
            return 1
        elif signal_strength < -0.3:
            return -1
        else:
            return 0
    
    def _calculate_performance(self, portfolio_values: List[float], trades: List[float]) -> StrategyPerformance:
        """Calculate strategy performance metrics"""
        if len(portfolio_values) < 2 or len(trades) == 0:
            return StrategyPerformance(
                total_return=0.0, sharpe_ratio=0.0, max_drawdown=1.0,
                volatility=0.0, win_rate=0.0, profit_factor=0.0,
                trades_per_year=0.0, risk_adjusted_return=0.0
            )
        
        # Calculate returns
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Trade statistics
        trades = np.array(trades)
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        trades_per_year = len(trades) * 252 / len(returns) if len(returns) > 0 else 0
        
        # Risk-adjusted return
        risk_adjusted_return = total_return / max(max_drawdown, 0.01)
        
        return StrategyPerformance(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_per_year=trades_per_year,
            risk_adjusted_return=risk_adjusted_return
        )
    
    def _calculate_multi_objective_fitness(self, performance: StrategyPerformance) -> float:
        """Calculate multi-objective fitness score"""
        # Weighted combination of different objectives
        weights = {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': 0.2,  # Negative weight (lower is better)
            'win_rate': 0.15,
            'profit_factor': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        normalized_sharpe = max(0, min(1, (performance.sharpe_ratio + 2) / 4))  # -2 to 2 -> 0 to 1
        normalized_return = max(0, min(1, (performance.total_return + 0.5) / 1))  # -0.5 to 0.5 -> 0 to 1
        normalized_drawdown = max(0, min(1, 1 - performance.max_drawdown))  # 0 to 1, inverted
        normalized_win_rate = performance.win_rate  # Already 0 to 1
        normalized_profit_factor = max(0, min(1, performance.profit_factor / 3))  # 0 to 3 -> 0 to 1
        
        fitness = (
            weights['sharpe_ratio'] * normalized_sharpe +
            weights['total_return'] * normalized_return +
            weights['max_drawdown'] * normalized_drawdown +
            weights['win_rate'] * normalized_win_rate +
            weights['profit_factor'] * normalized_profit_factor
        )
        
        return fitness
    
    def _calculate_diversification_bonus(self, strategy: Strategy) -> float:
        """Calculate diversification bonus based on genetic diversity"""
        if len(self.population) <= 1:
            return 0
        
        # Calculate genetic distance from other strategies
        distances = []
        for other in self.population:
            if other.id != strategy.id:
                distance = np.sqrt(sum((g1 - g2) ** 2 for g1, g2 in zip(strategy.genes, other.genes)))
                distances.append(distance)
        
        avg_distance = np.mean(distances)
        diversity_bonus = min(self.config.diversification_bonus, avg_distance / 10)
        
        return diversity_bonus
    
    def selection(self) -> List[Strategy]:
        """Select strategies for reproduction using tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.config.population_size):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda s: s.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: Strategy, parent2: Strategy) -> Tuple[Strategy, Strategy]:
        """Create offspring through crossover"""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        
        child1 = Strategy(
            id=f"gen{self.generation}_child_{random.randint(1000, 9999)}",
            genes=child1_genes
        )
        child2 = Strategy(
            id=f"gen{self.generation}_child_{random.randint(1000, 9999)}",
            genes=child2_genes
        )
        
        return child1, child2
    
    def mutate(self, strategy: Strategy) -> Strategy:
        """Apply mutation to strategy"""
        mutated_genes = strategy.genes.copy()
        
        for i in range(len(mutated_genes)):
            if random.random() < self.config.mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1
                mutated_genes[i] += random.gauss(0, mutation_strength)
                # Keep genes in bounds
                mutated_genes[i] = max(-2, min(2, mutated_genes[i]))
        
        mutated_strategy = Strategy(
            id=f"gen{self.generation}_mutated_{random.randint(1000, 9999)}",
            genes=mutated_genes
        )
        
        return mutated_strategy
    
    def evolve_generation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Evolve population for one generation"""
        # Evaluate fitness for all strategies
        for strategy in self.population:
            strategy.fitness = self.evaluate_fitness(strategy, market_data)
        
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Elite preservation
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        next_generation = self.population[:elite_count].copy()
        
        # Fill remaining population through selection, crossover, and mutation
        while len(next_generation) < self.config.population_size:
            selected = self.selection()
            
            for i in range(0, len(selected) - 1, 2):
                if len(next_generation) >= self.config.population_size:
                    break
                
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_generation.extend([child1, child2])
        
        # Trim to exact population size
        next_generation = next_generation[:self.config.population_size]
        
        # Update population
        self.population = next_generation
        self.generation += 1
        
        # Track statistics
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([s.fitness for s in self.population])
        diversity = np.std([s.fitness for s in self.population])
        
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'best_strategy_id': self.population[0].id
        }


class FederatedLearningNode:
    """Federated learning node for distributed strategy evolution"""
    
    def __init__(self, node_id: str, config: NexusAIConfig):
        self.node_id = node_id
        self.config = config
        self.local_population: List[Strategy] = []
        self.global_model_weights = None
        self.communication_round = 0
        self.logger = logger.bind(component="FederatedLearningNode", node_id=node_id)
    
    async def local_training(self, market_data: pd.DataFrame, epochs: int = 10) -> Dict[str, Any]:
        """Perform local training on node-specific data"""
        engine = EvolutionaryEngine(self.config)
        engine.population = self.local_population
        
        results = []
        for epoch in range(epochs):
            result = engine.evolve_generation(market_data)
            results.append(result)
            
            self.logger.info(f"Local epoch {epoch}: best_fitness={result['best_fitness']:.4f}")
        
        self.local_population = engine.population
        return {
            'node_id': self.node_id,
            'epochs': epochs,
            'final_best_fitness': results[-1]['best_fitness'],
            'results': results
        }
    
    def aggregate_strategies(self, global_strategies: List[Strategy]) -> None:
        """Aggregate global strategies with local population"""
        # Replace worst local strategies with best global ones
        self.local_population.sort(key=lambda s: s.fitness, reverse=True)
        global_strategies.sort(key=lambda s: s.fitness, reverse=True)
        
        # Replace bottom 20% with top global strategies
        replace_count = int(len(self.local_population) * 0.2)
        self.local_population[-replace_count:] = global_strategies[:replace_count]
        
        self.communication_round += 1
        self.logger.info(f"Aggregated strategies in round {self.communication_round}")


class NexusAI:
    """Main NexusAI class orchestrating autonomous strategy evolution"""
    
    def __init__(self, config: Union[NexusAIConfig, Dict]):
        if isinstance(config, dict):
            config = NexusAIConfig(**config)
        
        self.config = config
        self.evolutionary_engine = EvolutionaryEngine(config)
        self.federated_nodes: Dict[str, FederatedLearningNode] = {}
        self.strategy_memory: List[Strategy] = []
        self.performance_history = []
        self.market_regime_detector = None
        self.adaptive_parameters = {}
        self.logger = logger.bind(component="NexusAI")
        
    async def initialize(self, market_data: pd.DataFrame) -> None:
        """Initialize NexusAI system"""
        self.logger.info("Initializing NexusAI autonomous system")
        
        # Initialize evolutionary engine
        self.evolutionary_engine.initialize_population()
        
        # Initialize market regime detection
        await self._initialize_regime_detection(market_data)
        
        # Initialize adaptive parameters
        self._initialize_adaptive_parameters()
        
        self.logger.info("NexusAI initialization completed")
    
    async def _initialize_regime_detection(self, market_data: pd.DataFrame) -> None:
        """Initialize market regime detection for adaptive behavior"""
        from src.ml_models.regime_detection import RegimeDetector, RegimeDetectionConfig
        
        regime_config = RegimeDetectionConfig(
            n_regimes=3,
            method="hmm",
            features=["returns", "volatility", "volume_ratio"]
        )
        
        self.market_regime_detector = RegimeDetector(regime_config)
        
        # Prepare data for regime detection
        regime_data = market_data.copy()
        regime_data['returns'] = regime_data['close'].pct_change()
        regime_data['volatility'] = regime_data['returns'].rolling(20).std()
        
        if 'volume' in regime_data.columns:
            regime_data['volume_ma'] = regime_data['volume'].rolling(20).mean()
            regime_data['volume_ratio'] = regime_data['volume'] / regime_data['volume_ma']
        else:
            regime_data['volume_ratio'] = 1.0
        
        # Fit regime detector
        await self.market_regime_detector.fit_predict(regime_data.dropna())
    
    def _initialize_adaptive_parameters(self) -> None:
        """Initialize adaptive parameters for dynamic optimization"""
        self.adaptive_parameters = {
            'learning_rate': 0.1,
            'exploration_rate': 0.3,
            'selection_pressure': 1.0,
            'mutation_intensity': 1.0,
            'diversity_target': 0.5,
            'performance_threshold': 0.1
        }
    
    async def evolve(self, market_data: pd.DataFrame, generations: int = None) -> Dict[str, Any]:
        """Main evolution loop with adaptive behavior"""
        if generations is None:
            generations = self.config.max_generations
        
        self.logger.info(f"Starting evolution for {generations} generations")
        
        evolution_results = []
        best_strategy_history = []
        
        for generation in range(generations):
            # Detect current market regime
            current_regime = await self._detect_current_regime(market_data)
            
            # Adapt parameters based on regime
            await self._adapt_parameters(current_regime, generation)
            
            # Evolve generation
            result = self.evolutionary_engine.evolve_generation(market_data)
            evolution_results.append(result)
            
            # Track best strategy
            best_strategy = self.evolutionary_engine.population[0]
            best_strategy_history.append(best_strategy)
            
            # Update strategy memory
            self._update_strategy_memory(best_strategy)
            
            # Check for convergence
            if self._check_convergence():
                self.logger.info(f"Convergence achieved at generation {generation}")
                break
            
            # Adaptive learning
            if self.config.adaptive_learning:
                await self._adaptive_learning_update(result)
            
            if generation % 10 == 0:
                self.logger.info(
                    f"Generation {generation}: "
                    f"best_fitness={result['best_fitness']:.4f}, "
                    f"diversity={result['diversity']:.4f}, "
                    f"regime={current_regime}"
                )
        
        # Final evaluation
        final_result = await self._final_evaluation(market_data, best_strategy_history)
        
        return {
            'generations_completed': len(evolution_results),
            'evolution_results': evolution_results,
            'best_strategy': self.evolutionary_engine.population[0],
            'final_evaluation': final_result,
            'adaptive_parameters': self.adaptive_parameters
        }
    
    async def _detect_current_regime(self, market_data: pd.DataFrame) -> int:
        """Detect current market regime"""
        if self.market_regime_detector is None:
            return 0  # Default regime
        
        try:
            # Use last window of data for regime detection
            recent_data = market_data.tail(self.config.fitness_window)
            result = await self.market_regime_detector.predict_regime(recent_data)
            return int(result['regimes'][-1])
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return 0
    
    async def _adapt_parameters(self, regime: int, generation: int) -> None:
        """Adapt evolution parameters based on market regime and progress"""
        # Regime-based adaptation
        if regime == 0:  # Bull market
            self.adaptive_parameters['exploration_rate'] = 0.2
            self.adaptive_parameters['mutation_intensity'] = 0.8
        elif regime == 1:  # Bear market
            self.adaptive_parameters['exploration_rate'] = 0.4
            self.adaptive_parameters['mutation_intensity'] = 1.2
        else:  # Sideways market
            self.adaptive_parameters['exploration_rate'] = 0.3
            self.adaptive_parameters['mutation_intensity'] = 1.0
        
        # Progress-based adaptation
        progress = generation / self.config.max_generations
        self.adaptive_parameters['exploration_rate'] *= (1 - progress * 0.5)  # Decrease exploration
        
        # Update evolutionary engine parameters
        self.evolutionary_engine.config.mutation_rate = (
            self.config.mutation_rate * self.adaptive_parameters['mutation_intensity']
        )
    
    def _update_strategy_memory(self, strategy: Strategy) -> None:
        """Update long-term strategy memory"""
        self.strategy_memory.append(strategy)
        
        # Keep only best strategies in memory
        if len(self.strategy_memory) > self.config.memory_length:
            self.strategy_memory.sort(key=lambda s: s.fitness, reverse=True)
            self.strategy_memory = self.strategy_memory[:self.config.memory_length]
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged"""
        if len(self.evolutionary_engine.best_fitness_history) < 10:
            return False
        
        recent_fitness = self.evolutionary_engine.best_fitness_history[-10:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < self.config.convergence_threshold
    
    async def _adaptive_learning_update(self, result: Dict[str, Any]) -> None:
        """Update adaptive parameters based on evolution performance"""
        # Adjust learning rate based on diversity
        if result['diversity'] < self.adaptive_parameters['diversity_target']:
            self.adaptive_parameters['mutation_intensity'] *= 1.1  # Increase mutation
        else:
            self.adaptive_parameters['mutation_intensity'] *= 0.95  # Decrease mutation
        
        # Adjust selection pressure based on improvement
        if len(self.evolutionary_engine.best_fitness_history) >= 2:
            improvement = (
                self.evolutionary_engine.best_fitness_history[-1] -
                self.evolutionary_engine.best_fitness_history[-2]
            )
            
            if improvement > self.adaptive_parameters['performance_threshold']:
                self.adaptive_parameters['selection_pressure'] *= 1.05
            else:
                self.adaptive_parameters['selection_pressure'] *= 0.98
    
    async def _final_evaluation(self, market_data: pd.DataFrame, strategy_history: List[Strategy]) -> Dict[str, Any]:
        """Perform final evaluation of evolved strategies"""
        best_strategy = self.evolutionary_engine.population[0]
        
        # Evaluate on full dataset
        full_performance = self.evolutionary_engine._simulate_strategy(
            self.evolutionary_engine._genes_to_parameters(best_strategy.genes),
            market_data
        )
        
        # Calculate strategy stability
        stability_score = self._calculate_strategy_stability(strategy_history)
        
        # Generate strategy interpretation
        interpretation = await self._interpret_strategy(best_strategy)
        
        return {
            'best_strategy_performance': full_performance.dict(),
            'strategy_stability': stability_score,
            'strategy_interpretation': interpretation,
            'total_strategies_evaluated': len(self.strategy_memory),
            'convergence_generation': len(self.evolutionary_engine.best_fitness_history)
        }
    
    def _calculate_strategy_stability(self, strategy_history: List[Strategy]) -> float:
        """Calculate stability of strategy evolution"""
        if len(strategy_history) < 5:
            return 0.0
        
        # Calculate gene variance over time
        gene_variances = []
        for gene_idx in range(len(strategy_history[0].genes)):
            gene_values = [s.genes[gene_idx] for s in strategy_history[-10:]]
            gene_variances.append(np.var(gene_values))
        
        # Lower variance indicates higher stability
        stability = 1.0 / (1.0 + np.mean(gene_variances))
        return stability
    
    async def _interpret_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Generate human-readable interpretation of strategy"""
        params = self.evolutionary_engine._genes_to_parameters(strategy.genes)
        
        interpretation = {
            'strategy_type': self._classify_strategy_type(params),
            'key_parameters': self._extract_key_parameters(params),
            'trading_style': self._determine_trading_style(params),
            'risk_profile': self._assess_risk_profile(params),
            'market_conditions': self._preferred_market_conditions(params)
        }
        
        return interpretation
    
    def _classify_strategy_type(self, params: Dict[str, float]) -> str:
        """Classify strategy based on parameters"""
        if params['ma_short_period'] < 20 and params['ma_long_period'] < 100:
            return "Short-term Momentum"
        elif params['rsi_oversold'] < 25 and params['rsi_overbought'] > 75:
            return "Mean Reversion"
        elif params['trend_strength'] > 0.7:
            return "Trend Following"
        elif params['volatility_adj'] > 0.8:
            return "Volatility Adaptive"
        else:
            return "Hybrid Strategy"
    
    def _extract_key_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Extract most important parameters"""
        return {
            'short_ma_period': params['ma_short_period'],
            'long_ma_period': params['ma_long_period'],
            'rsi_period': params['rsi_period'],
            'stop_loss': params['stop_loss'],
            'take_profit': params['take_profit'],
            'position_size': params['position_size']
        }
    
    def _determine_trading_style(self, params: Dict[str, float]) -> str:
        """Determine trading style characteristics"""
        avg_holding_period = (params['ma_short_period'] + params['ma_long_period']) / 2
        
        if avg_holding_period < 10:
            return "High Frequency"
        elif avg_holding_period < 50:
            return "Short Term"
        elif avg_holding_period < 150:
            return "Medium Term"
        else:
            return "Long Term"
    
    def _assess_risk_profile(self, params: Dict[str, float]) -> str:
        """Assess risk profile of strategy"""
        risk_score = (
            params['stop_loss'] * 0.4 +
            params['position_size'] * 0.3 +
            (1 - params['volatility_adj']) * 0.3
        )
        
        if risk_score < 0.3:
            return "Conservative"
        elif risk_score < 0.6:
            return "Moderate"
        else:
            return "Aggressive"
    
    def _preferred_market_conditions(self, params: Dict[str, float]) -> List[str]:
        """Determine preferred market conditions"""
        conditions = []
        
        if params['trend_strength'] > 0.7:
            conditions.append("Trending Markets")
        
        if params['rsi_oversold'] < 25:
            conditions.append("High Volatility")
        
        if params['volume_threshold'] > 1.5:
            conditions.append("High Volume")
        
        if params['regime_sensitivity'] > 0.7:
            conditions.append("Regime Changes")
        
        return conditions if conditions else ["All Market Conditions"]
    
    async def federated_evolution(self, node_data: Dict[str, pd.DataFrame], rounds: int = 10) -> Dict[str, Any]:
        """Perform federated evolution across multiple nodes"""
        self.logger.info(f"Starting federated evolution with {len(node_data)} nodes")
        
        # Initialize federated nodes
        for node_id, data in node_data.items():
            node = FederatedLearningNode(node_id, self.config)
            node.local_population = self.evolutionary_engine.population[:self.config.population_size // len(node_data)]
            self.federated_nodes[node_id] = node
        
        federated_results = []
        
        for round_num in range(rounds):
            self.logger.info(f"Federated round {round_num + 1}/{rounds}")
            
            # Local training on each node
            local_results = {}
            for node_id, node in self.federated_nodes.items():
                result = await node.local_training(node_data[node_id], epochs=5)
                local_results[node_id] = result
            
            # Aggregate strategies
            all_strategies = []
            for node in self.federated_nodes.values():
                all_strategies.extend(node.local_population)
            
            # Select best global strategies
            all_strategies.sort(key=lambda s: s.fitness, reverse=True)
            global_best = all_strategies[:self.config.population_size // 4]
            
            # Distribute to all nodes
            for node in self.federated_nodes.values():
                node.aggregate_strategies(global_best)
            
            # Track round results
            round_result = {
                'round': round_num + 1,
                'local_results': local_results,
                'global_best_fitness': global_best[0].fitness if global_best else 0,
                'total_strategies': len(all_strategies)
            }
            federated_results.append(round_result)
        
        # Final aggregation
        final_population = []
        for node in self.federated_nodes.values():
            final_population.extend(node.local_population)
        
        final_population.sort(key=lambda s: s.fitness, reverse=True)
        self.evolutionary_engine.population = final_population[:self.config.population_size]
        
        return {
            'federated_rounds': rounds,
            'participating_nodes': list(node_data.keys()),
            'round_results': federated_results,
            'final_best_strategy': final_population[0] if final_population else None,
            'total_strategies_evolved': len(final_population)
        }
    
    async def real_time_adaptation(self, live_data_stream: Any) -> None:
        """Continuously adapt strategies to live market data"""
        self.logger.info("Starting real-time adaptation mode")
        
        adaptation_buffer = []
        adaptation_frequency = 100  # Adapt every 100 data points
        
        async for data_point in live_data_stream:
            adaptation_buffer.append(data_point)
            
            if len(adaptation_buffer) >= adaptation_frequency:
                # Convert buffer to DataFrame
                recent_data = pd.DataFrame(adaptation_buffer)
                
                # Quick evolution cycle
                quick_result = self.evolutionary_engine.evolve_generation(recent_data)
                
                # Update best strategy if improvement found
                if quick_result['best_fitness'] > self.adaptive_parameters.get('last_best_fitness', 0):
                    self.adaptive_parameters['last_best_fitness'] = quick_result['best_fitness']
                    self.logger.info(f"Real-time adaptation: new best fitness {quick_result['best_fitness']:.4f}")
                
                # Keep only recent data
                adaptation_buffer = adaptation_buffer[-50:]
    
    def get_current_best_strategy(self) -> Dict[str, Any]:
        """Get current best strategy with interpretation"""
        if not self.evolutionary_engine.population:
            return {}
        
        best_strategy = self.evolutionary_engine.population[0]
        params = self.evolutionary_engine._genes_to_parameters(best_strategy.genes)
        
        return {
            'strategy_id': best_strategy.id,
            'fitness': best_strategy.fitness,
            'parameters': params,
            'interpretation': asyncio.run(self._interpret_strategy(best_strategy)),
            'age': best_strategy.age,
            'performance_stats': {
                'wins': best_strategy.wins,
                'losses': best_strategy.losses,
                'total_trades': best_strategy.total_trades
            }
        }
    
    def save_nexus_state(self, path: str) -> None:
        """Save complete NexusAI state"""
        state_data = {
            'config': self.config.dict(),
            'population': [
                {
                    'id': s.id,
                    'genes': s.genes,
                    'fitness': s.fitness,
                    'age': s.age,
                    'wins': s.wins,
                    'losses': s.losses,
                    'total_trades': s.total_trades,
                    'created_at': s.created_at.isoformat() if s.created_at else None
                }
                for s in self.evolutionary_engine.population
            ],
            'strategy_memory': [
                {
                    'id': s.id,
                    'genes': s.genes,
                    'fitness': s.fitness,
                    'age': s.age
                }
                for s in self.strategy_memory
            ],
            'adaptive_parameters': self.adaptive_parameters,
            'generation': self.evolutionary_engine.generation,
            'best_fitness_history': self.evolutionary_engine.best_fitness_history,
            'diversity_history': self.evolutionary_engine.diversity_history
        }
        
        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.info(f"NexusAI state saved to {path}")
    
    def load_nexus_state(self, path: str) -> None:
        """Load complete NexusAI state"""
        with open(path, 'r') as f:
            state_data = json.load(f)
        
        self.config = NexusAIConfig(**state_data['config'])
        
        # Restore population
        population = []
        for s_data in state_data['population']:
            strategy = Strategy(
                id=s_data['id'],
                genes=s_data['genes'],
                fitness=s_data['fitness'],
                age=s_data['age'],
                wins=s_data['wins'],
                losses=s_data['losses'],
                total_trades=s_data['total_trades'],
                created_at=datetime.fromisoformat(s_data['created_at']) if s_data['created_at'] else None
            )
            population.append(strategy)
        
        # Restore strategy memory
        memory = []
        for s_data in state_data['strategy_memory']:
            strategy = Strategy(
                id=s_data['id'],
                genes=s_data['genes'],
                fitness=s_data['fitness'],
                age=s_data['age']
            )
            memory.append(strategy)
        
        self.evolutionary_engine = EvolutionaryEngine(self.config)
        self.evolutionary_engine.population = population
        self.evolutionary_engine.generation = state_data['generation']
        self.evolutionary_engine.best_fitness_history = state_data['best_fitness_history']
        self.evolutionary_engine.diversity_history = state_data['diversity_history']
        
        self.strategy_memory = memory
        self.adaptive_parameters = state_data['adaptive_parameters']
        
        self.logger.info(f"NexusAI state loaded from {path}")

