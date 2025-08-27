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


class FederatedLearningNode
