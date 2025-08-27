import asyncio
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.linalg import cholesky
from pydantic import BaseModel, Field

from src.core.exceptions import MonteCarloException
from src.core.logging import logger


class MultiAssetParameters(BaseModel):
    initial_prices: List[float] = Field(..., description="Initial asset prices")
    drifts: List[float] = Field(..., description="Asset drifts")
    volatilities: List[float] = Field(..., description="Asset volatilities")
    correlation_matrix: List[List[float]] = Field(
        ..., description="Asset correlation matrix"
    )
    T: float = Field(..., description="Time to maturity")
    n_paths: int = Field(default=100000, description="Number of simulation paths")
    n_steps: int = Field(default=252, description="Number of time steps")
    weights: Optional[List[float]] = Field(
        None, description="Portfolio weights (if None, equal weights)"
    )
    seed: Optional[int] = Field(default=None, description="Random seed")


class MultiAssetResult(BaseModel):
    asset_paths: np.ndarray = Field(..., description="Individual asset paths")
    portfolio_paths: np.ndarray = Field(..., description="Portfolio value paths")
    final_prices: np.ndarray = Field(..., description="Final asset prices")
    portfolio_returns: np.ndarray = Field(..., description="Portfolio returns")
    correlation_analysis: Dict[str, np.ndarray] = Field(
        ..., description="Correlation analysis"
    )
    statistics: Dict[str, float] = Field(..., description="Portfolio statistics")
    execution_time: float = Field(..., description="Execution time in seconds")


@jit(nopython=True)
def multi_asset_simulation_numba(
    initial_prices: np.ndarray,
    drifts: np.ndarray,
    volatilities: np.ndarray,
    chol_matrix: np.ndarray,
    T: float,
    n_paths: int,
    n_steps: int,
    n_assets: int,
    random_numbers: np.ndarray,
) -> np.ndarray:
    dt = T / n_steps
    paths = np.zeros((n_assets, n_paths, n_steps + 1))
    
    for i in range(n_assets):
        paths[i, :, 0] = initial_prices[i]

    for path in range(n_paths):
        for step in range(n_steps):
            uncorrelated_randoms = random_numbers[path, step, :]
            correlated_randoms = np.dot(chol_matrix, uncorrelated_randoms)
            
            for asset in range(n_assets):
                dW = correlated_randoms[asset] * np.sqrt(dt)
                paths[asset, path, step + 1] = paths[asset, path, step] * np.exp(
                    (drifts[asset] - 0.5 * volatilities[asset] ** 2) * dt
                    + volatilities[asset] * dW
                )

    return paths


class MultiAssetEngine:
    def __init__(self):
        self.logger = logger.bind(component="MultiAssetEngine")

    async def simulate_portfolio(
        self, parameters: Union[MultiAssetParameters, Dict]
    ) -> MultiAssetResult:
        if isinstance(parameters, dict):
            parameters = MultiAssetParameters(**parameters)

        start_time = asyncio.get_event_loop().time()

        try:
            n_assets = len(parameters.initial_prices)
            
            if parameters.weights is None:
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = np.array(parameters.weights)
                if len(weights) != n_assets:
                    raise MonteCarloException("Weights length must match number of assets")
                if not np.isclose(np.sum(weights), 1.0):
                    raise MonteCarloException("Weights must sum to 1.0")

            correlation_matrix = np.array(parameters.correlation_matrix)
            if correlation_matrix.shape != (n_assets, n_assets):
                raise MonteCarloException("Correlation matrix shape mismatch")

            if not self._is_positive_definite(correlation_matrix):
                correlation_matrix = self._nearest_positive_definite(correlation_matrix)
                self.logger.warning("Correlation matrix was not positive definite, adjusted")

            chol_matrix = cholesky(correlation_matrix, lower=True)

            if parameters.seed:
                np.random.seed(parameters.seed)

            random_numbers = np.random.standard_normal(
                (parameters.n_paths, parameters.n_steps, n_assets)
            )

            asset_paths = multi_asset_simulation_numba(
                np.array(parameters.initial_prices),
                np.array(parameters.drifts),
                np.array(parameters.volatilities),
                chol_matrix,
                parameters.T,
                parameters.n_paths,
                parameters.n_steps,
                n_assets,
                random_numbers,
            )

            portfolio_paths = np.sum(asset_paths * weights.reshape(-1, 1, 1), axis=0)
            
            final_prices = asset_paths[:, :, -1]
            portfolio_returns = (portfolio_paths[:, -1] / portfolio_paths[:, 0]) - 1

            correlation_analysis = self._analyze_correlations(asset_paths)
            statistics = self._calculate_portfolio_statistics(
                portfolio_paths, portfolio_returns, parameters
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            result = MultiAssetResult(
                asset_paths=asset_paths,
                portfolio_paths=portfolio_paths,
                final_prices=final_prices,
                portfolio_returns=portfolio_returns,
                correlation_analysis=correlation_analysis,
                statistics=statistics,
                execution_time=execution_time,
            )

            self.logger.info(
                "Multi-asset simulation completed",
                n_assets=n_assets,
                n_paths=parameters.n_paths,
                execution_time=execution_time,
                portfolio_mean_return=statistics["mean_return"],
            )

            return result

        except Exception as e:
            self.logger.error(f"Multi-asset simulation failed: {str(e)}")
            raise MonteCarloException(f"Multi-asset simulation failed: {str(e)}")

    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        try:
            cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def _nearest_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues[eigenvalues < 1e-8] = 1e-8
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _analyze_correlations(self, asset_paths: np.ndarray) -> Dict[str, np.ndarray]:
        n_assets, n_paths, n_steps = asset_paths.shape
        
        returns = np.diff(np.log(asset_paths), axis=2)
        
        realized_correlations = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    path_correlations = []
                    for path in range(n_paths):
                        corr = np.corrcoef(returns[i, path, :], returns[j, path, :])[0, 1]
                        if not np.isnan(corr):
                            path_correlations.append(corr)
                    realized_correlations[i, j] = np.mean(path_correlations) if path_correlations else 0
                else:
                    realized_correlations[i, j] = 1.0

        return {
            "realized_correlation_matrix": realized_correlations,
            "correlation_evolution": self._calculate_correlation_evolution(returns),
            "average_correlation": np.mean(realized_correlations[np.triu_indices_from(realized_correlations, k=1)]),
        }

    def _calculate_correlation_evolution(self, returns: np.ndarray) -> np.ndarray:
        n_assets, n_paths, n_steps = returns.shape
        correlation_evolution = np.zeros((n_assets, n_assets, n_steps))
        
        window_size = min(20, n_steps // 4)
        
        for step in range(window_size, n_steps):
            window_returns = returns[:, :, step-window_size:step]
            for i in range(n_assets):
                for j in range(n_assets):
                    if i != j:
                        correlations = []
                        for path in range(n_paths):
                            corr = np.corrcoef(
                                window_returns[i, path, :], 
                                window_returns[j, path, :]
                            )[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                        correlation_evolution[i, j, step] = np.mean(correlations) if correlations else 0
                    else:
                        correlation_evolution[i, j, step] = 1.0

        return correlation_evolution

    def _calculate_portfolio_statistics(
        self, portfolio_paths: np.ndarray, portfolio_returns: np.ndarray, parameters: MultiAssetParameters
    ) -> Dict[str, float]:
        daily_returns = np.diff(np.log(portfolio_paths), axis=1)
        
        return {
            "mean_return": float(np.mean(portfolio_returns)),
            "volatility": float(np.std(portfolio_returns)),
            "sharpe_ratio": float(np.mean(portfolio_returns) / np.std(portfolio_returns)) if np.std(portfolio_returns) > 0 else 0.0,
            "sortino_ratio": float(np.mean(portfolio_returns) / np.std(portfolio_returns[portfolio_returns < 0])) if len(portfolio_returns[portfolio_returns < 0]) > 0 else float('inf'),
            "max_drawdown": float(self._calculate_max_drawdown(portfolio_paths)),
            "var_95": float(np.percentile(portfolio_returns, 5)),
            "var_99": float(np.percentile(portfolio_returns, 1)),
            "expected_shortfall_95": float(np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)])),
            "expected_shortfall_99": float(np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)])),
            "skewness": float(self._calculate_skewness(portfolio_returns)),
            "kurtosis": float(self._calculate_kurtosis(portfolio_returns)),
            "probability_of_loss": float(np.mean(portfolio_returns < 0)),
            "daily_volatility": float(np.std(daily_returns)),
            "annualized_volatility": float(np.std(daily_returns) * np.sqrt(252)),
            "calmar_ratio": float(np.mean(portfolio_returns) / abs(self._calculate_max_drawdown(portfolio_paths))) if self._calculate_max_drawdown(portfolio_paths) != 0 else float('inf'),
        }

    def _calculate_max_drawdown(self, paths: np.ndarray) -> float:
        cumulative_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - cumulative_max) / cumulative_max
        return np.min(drawdowns)

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return np.mean(((returns - mean_return) / std_return) ** 3) if std_return > 0 else 0.0

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3 if std_return > 0 else 0.0

    async def calculate_efficient_frontier(
        self, parameters: MultiAssetParameters, n_points: int = 50
    ) -> Dict[str, List[float]]:
        n_assets = len(parameters.initial_prices)
        
        min_risk_weights = await self._find_minimum_variance_portfolio(parameters)
        max_return_weights = np.zeros(n_assets)
        
        asset_returns = []
        for i in range(n_assets):
            single_asset_params = parameters.copy(deep=True)
            single_asset_params.weights = [1.0 if j == i else 0.0 for j in range(n_assets)]
            result = await self.simulate_portfolio(single_asset_params)
            asset_returns.append(result.statistics["mean_return"])
        
        max_return_idx = np.argmax(asset_returns)
        max_return_weights[max_return_idx] = 1.0

        efficient_weights = []
        efficient_returns = []
        efficient_risks = []

        for i in range(n_points):
            alpha = i / (n_points - 1)
            weights = (1 - alpha) * min_risk_weights + alpha * max_return_weights
            weights = weights / np.sum(weights)

            test_params = parameters.copy(deep=True)
            test_params.weights = weights.tolist()
            test_params.n_paths = min(50000, parameters.n_paths)

            result = await self.simulate_portfolio(test_params)
            
            efficient_weights.append(weights.tolist())
            efficient_returns.append(result.statistics["mean_return"])
            efficient_risks.append(result.statistics["volatility"])

        return {
            "weights": efficient_weights,
            "returns": efficient_returns,
            "risks": efficient_risks,
        }

    async def _find_minimum_variance_portfolio(self, parameters: MultiAssetParameters) -> np.ndarray:
        from scipy.optimize import minimize

        n_assets = len(parameters.initial_prices)
        
        async def portfolio_variance(weights):
            test_params = parameters.copy(deep=True)
            test_params.weights = weights.tolist()
            test_params.n_paths = min(10000, parameters.n_paths)
            result = await self.simulate_portfolio(test_params)
            return result.statistics["volatility"] ** 2

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)

        def sync_portfolio_variance(weights):
            return asyncio.run(portfolio_variance(weights))

        result = minimize(
            sync_portfolio_variance,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_guess

    async def calculate_portfolio_var(
        self,
        parameters: MultiAssetParameters,
        confidence_level: float = 0.05,
        holding_period: int = 1,
    ) -> Dict[str, float]:
        result = await self.simulate_portfolio(parameters)
        
        if holding_period > 1:
            scaling_factor = np.sqrt(holding_period)
            scaled_returns = result.portfolio_returns * scaling_factor
        else:
            scaled_returns = result.portfolio_returns

        var_value = np.percentile(scaled_returns, confidence_level * 100)
        cvar_value = np.mean(scaled_returns[scaled_returns <= var_value])

        initial_portfolio_value = np.sum(
            np.array(parameters.initial_prices) * np.array(parameters.weights or [1.0 / len(parameters.initial_prices)] * len(parameters.initial_prices))
        )

        return {
            "var_percentage": float(var_value),
            "var_absolute": float(var_value * initial_portfolio_value),
            "expected_shortfall_percentage": float(cvar_value),
            "expected_shortfall_absolute": float(cvar_value * initial_portfolio_value),
            "confidence_level": confidence_level,
            "holding_period_days": holding_period,
            "initial_portfolio_value": float(initial_portfolio_value),
        }

    async def backtest_portfolio(
        self,
        parameters: MultiAssetParameters,
        rebalancing_frequency: int = 22,  # Monthly
        transaction_cost: float = 0.001,  # 0.1%
    ) -> Dict[str, Union[float, List[float]]]:
        result = await self.simulate_portfolio(parameters)
        
        n_rebalances = parameters.n_steps // rebalancing_frequency
        portfolio_values = []
        transaction_costs = []
        
        weights = np.array(parameters.weights or [1.0 / len(parameters.initial_prices)] * len(parameters.initial_prices))
        current_weights = weights.copy()
        
        for i in range(n_rebalances):
            start_idx = i * rebalancing_frequency
            end_idx = min((i + 1) * rebalancing_frequency, parameters.n_steps)
            
            period_returns = []
            for path in range(min(1000, parameters.n_paths)):  # Sample for efficiency
                period_return = np.sum(
                    current_weights * (
                        result.asset_paths[:, path, end_idx] / result.asset_paths[:, path, start_idx] - 1
                    )
                )
                period_returns.append(period_return)
            
            avg_period_return = np.mean(period_returns)
            portfolio_values.append(avg_period_return)
            
            # Calculate transaction costs from rebalancing
            weight_drift = current_weights * (1 + avg_period_return)
            weight_drift = weight_drift / np.sum(weight_drift)
            
            rebalancing_cost = np.sum(np.abs(weights - weight_drift)) * transaction_cost
            transaction_costs.append(rebalancing_cost)
            
            current_weights = weights.copy()

        total_return = np.prod([1 + pv for pv in portfolio_values]) - 1
        total_transaction_costs = np.sum(transaction_costs)
        net_return = total_return - total_transaction_costs
        
        return {
            "total_gross_return": float(total_return),
            "total_transaction_costs": float(total_transaction_costs),
            "total_net_return": float(net_return),
            "period_returns": portfolio_values,
            "period_transaction_costs": transaction_costs,
            "annualized_net_return": float((1 + net_return) ** (252 / parameters.n_steps) - 1),
            "information_ratio": float(net_return / np.std(portfolio_values)) if np.std(portfolio_values) > 0 else 0.0,
        }
