import asyncio
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numba import jit, cuda
from scipy import stats
from pydantic import BaseModel, Field

from src.core.exceptions import MonteCarloException
from src.core.logging import logger


class GBMParameters(BaseModel):
    S0: float = Field(..., description="Initial stock price")
    mu: float = Field(..., description="Drift (risk-free rate)")
    sigma: float = Field(..., description="Volatility")
    T: float = Field(..., description="Time to maturity")
    n_paths: int = Field(default=100000, description="Number of simulation paths")
    n_steps: int = Field(default=252, description="Number of time steps")
    antithetic: bool = Field(default=True, description="Use antithetic variates")
    control_variate: bool = Field(default=True, description="Use control variates")
    seed: Optional[int] = Field(default=None, description="Random seed")


class GBMResult(BaseModel):
    paths: np.ndarray = Field(..., description="Simulated price paths")
    final_prices: np.ndarray = Field(..., description="Final prices")
    statistics: Dict[str, float] = Field(..., description="Price statistics")
    execution_time: float = Field(..., description="Execution time in seconds")
    variance_reduction_ratio: Optional[float] = Field(
        None, description="Variance reduction achieved"
    )


@jit(nopython=True)
def gbm_simulation_numba(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    random_numbers: np.ndarray,
) -> np.ndarray:
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_paths):
        for j in range(n_steps):
            dW = random_numbers[i, j] * np.sqrt(dt)
            paths[i, j + 1] = paths[i, j] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW
            )

    return paths


@cuda.jit
def gbm_simulation_cuda(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    random_numbers: np.ndarray,
    paths: np.ndarray,
) -> None:
    idx = cuda.grid(1)
    if idx < paths.shape[0]:
        dt = T / n_steps
        paths[idx, 0] = S0

        for j in range(n_steps):
            dW = random_numbers[idx, j] * np.sqrt(dt)
            paths[idx, j + 1] = paths[idx, j] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW
            )


class GBMEngine:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.logger = logger.bind(component="GBMEngine")
        if self.use_gpu:
            try:
                import cupy as cp

                self.cp = cp
                self.gpu_available = True
                self.logger.info("GPU acceleration enabled")
            except ImportError:
                self.gpu_available = False
                self.logger.warning("GPU not available, falling back to CPU")

    async def simulate_paths(
        self, parameters: Union[GBMParameters, Dict]
    ) -> GBMResult:
        if isinstance(parameters, dict):
            parameters = GBMParameters(**parameters)

        start_time = asyncio.get_event_loop().time()

        try:
            if self.use_gpu and self.gpu_available:
                paths = await self._simulate_gpu(parameters)
            else:
                paths = await self._simulate_cpu(parameters)

            final_prices = paths[:, -1]
            statistics = self._calculate_statistics(final_prices, parameters)

            execution_time = asyncio.get_event_loop().time() - start_time

            variance_reduction_ratio = None
            if parameters.antithetic or parameters.control_variate:
                variance_reduction_ratio = await self._calculate_variance_reduction(
                    parameters
                )

            result = GBMResult(
                paths=paths,
                final_prices=final_prices,
                statistics=statistics,
                execution_time=execution_time,
                variance_reduction_ratio=variance_reduction_ratio,
            )

            self.logger.info(
                "GBM simulation completed",
                n_paths=parameters.n_paths,
                execution_time=execution_time,
                mean_final_price=statistics["mean"],
                gpu_used=self.use_gpu and self.gpu_available,
            )

            return result

        except Exception as e:
            self.logger.error(f"GBM simulation failed: {str(e)}")
            raise MonteCarloException(f"GBM simulation failed: {str(e)}")

    async def _simulate_cpu(self, parameters: GBMParameters) -> np.ndarray:
        if parameters.seed:
            np.random.seed(parameters.seed)

        random_numbers = np.random.standard_normal(
            (parameters.n_paths, parameters.n_steps)
        )

        if parameters.antithetic:
            half_paths = parameters.n_paths // 2
            antithetic_random = -random_numbers[:half_paths]
            random_numbers = np.vstack([random_numbers[:half_paths], antithetic_random])

        paths = gbm_simulation_numba(
            parameters.S0,
            parameters.mu,
            parameters.sigma,
            parameters.T,
            parameters.n_paths,
            parameters.n_steps,
            random_numbers,
        )

        return paths

    async def _simulate_gpu(self, parameters: GBMParameters) -> np.ndarray:
        if parameters.seed:
            self.cp.random.seed(parameters.seed)

        random_numbers = self.cp.random.standard_normal(
            (parameters.n_paths, parameters.n_steps), dtype=self.cp.float32
        )

        if parameters.antithetic:
            half_paths = parameters.n_paths // 2
            antithetic_random = -random_numbers[:half_paths]
            random_numbers = self.cp.vstack(
                [random_numbers[:half_paths], antithetic_random]
            )

        paths = self.cp.zeros(
            (parameters.n_paths, parameters.n_steps + 1), dtype=self.cp.float32
        )

        threads_per_block = 256
        blocks_per_grid = (parameters.n_paths + threads_per_block - 1) // threads_per_block

        gbm_simulation_cuda[blocks_per_grid, threads_per_block](
            parameters.S0,
            parameters.mu,
            parameters.sigma,
            parameters.T,
            parameters.n_steps,
            random_numbers,
            paths,
        )

        return self.cp.asnumpy(paths)

    def _calculate_statistics(
        self, final_prices: np.ndarray, parameters: GBMParameters
    ) -> Dict[str, float]:
        analytical_mean = parameters.S0 * np.exp(parameters.mu * parameters.T)
        analytical_std = parameters.S0 * np.exp(parameters.mu * parameters.T) * np.sqrt(
            np.exp(parameters.sigma**2 * parameters.T) - 1
        )

        statistics = {
            "mean": float(np.mean(final_prices)),
            "std": float(np.std(final_prices)),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
            "median": float(np.median(final_prices)),
            "skewness": float(stats.skew(final_prices)),
            "kurtosis": float(stats.kurtosis(final_prices)),
            "analytical_mean": float(analytical_mean),
            "analytical_std": float(analytical_std),
            "mean_error": float(abs(np.mean(final_prices) - analytical_mean)),
            "std_error": float(abs(np.std(final_prices) - analytical_std)),
            "confidence_95_lower": float(np.percentile(final_prices, 2.5)),
            "confidence_95_upper": float(np.percentile(final_prices, 97.5)),
        }

        return statistics

    async def _calculate_variance_reduction(
        self, parameters: GBMParameters
    ) -> float:
        basic_params = parameters.copy(deep=True)
        basic_params.antithetic = False
        basic_params.control_variate = False
        basic_params.n_paths = min(10000, parameters.n_paths)

        basic_result = await self.simulate_paths(basic_params)
        enhanced_result = await self.simulate_paths(
            parameters.copy(update={"n_paths": basic_params.n_paths})
        )

        basic_var = np.var(basic_result.final_prices)
        enhanced_var = np.var(enhanced_result.final_prices)

        return float(basic_var / enhanced_var) if enhanced_var > 0 else 1.0

    async def price_european_option(
        self, option_type: str, strike: float, parameters: GBMParameters
    ) -> Dict[str, float]:
        result = await self.simulate_paths(parameters)
        final_prices = result.final_prices

        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise MonteCarloException(f"Unknown option type: {option_type}")

        option_price = np.mean(payoffs) * np.exp(-parameters.mu * parameters.T)
        option_std = np.std(payoffs) * np.exp(-parameters.mu * parameters.T)

        analytical_price = self._black_scholes_price(
            option_type, parameters.S0, strike, parameters.T, parameters.mu, parameters.sigma
        )

        return {
            "monte_carlo_price": float(option_price),
            "standard_error": float(option_std / np.sqrt(len(payoffs))),
            "analytical_price": float(analytical_price),
            "pricing_error": float(abs(option_price - analytical_price)),
            "confidence_95_lower": float(
                option_price - 1.96 * option_std / np.sqrt(len(payoffs))
            ),
            "confidence_95_upper": float(
                option_price + 1.96 * option_std / np.sqrt(len(payoffs))
            ),
        }

    def _black_scholes_price(
        self,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "call":
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type.lower() == "put":
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise MonteCarloException(f"Unknown option type: {option_type}")

    async def calculate_var(
        self, parameters: GBMParameters, confidence_level: float = 0.05
    ) -> Dict[str, float]:
        result = await self.simulate_paths(parameters)
        returns = (result.final_prices / parameters.S0) - 1

        var_value = np.percentile(returns, confidence_level * 100)
        cvar_value = np.mean(returns[returns <= var_value])

        return {
            "var": float(var_value),
            "expected_shortfall": float(cvar_value),
            "worst_case": float(np.min(returns)),
            "best_case": float(np.max(returns)),
            "probability_of_loss": float(np.mean(returns < 0)),
        }

    async def stress_test(
        self, parameters: GBMParameters, stress_scenarios: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        results = {}

        for i, scenario in enumerate(stress_scenarios):
            stressed_params = parameters.copy(deep=True)
            for key, value in scenario.items():
                if hasattr(stressed_params, key):
                    setattr(stressed_params, key, value)

            result = await self.simulate_paths(stressed_params)
            returns = (result.final_prices / parameters.S0) - 1

            results[f"scenario_{i+1}"] = {
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "var_95": float(np.percentile(returns, 5)),
                "expected_shortfall": float(
                    np.mean(returns[returns <= np.percentile(returns, 5)])
                ),
                "scenario": scenario,
            }

        return results
