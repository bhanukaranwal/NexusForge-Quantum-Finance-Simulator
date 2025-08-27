import asyncio
from typing import Dict, List, Optional, Union

import numpy as np
from numba import jit
from pydantic import BaseModel, Field

from src.core.exceptions import MonteCarloException
from src.core.logging import logger


class PathDependentParameters(BaseModel):
    S0: float = Field(..., description="Initial asset price")
    mu: float = Field(..., description="Drift rate")
    sigma: float = Field(..., description="Volatility")
    T: float = Field(..., description="Time to maturity")
    n_paths: int = Field(default=100000, description="Number of simulation paths")
    n_steps: int = Field(default=252, description="Number of time steps")
    option_type: str = Field(..., description="Option type: asian, barrier, lookback")
    strike: Optional[float] = Field(None, description="Strike price")
    barrier_level: Optional[float] = Field(None, description="Barrier level")
    barrier_type: Optional[str] = Field(None, description="up-and-in, up-and-out, down-and-in, down-and-out")
    averaging_type: Optional[str] = Field(None, description="arithmetic or geometric")
    seed: Optional[int] = Field(default=None, description="Random seed")


class PathDependentResult(BaseModel):
    option_price: float = Field(..., description="Calculated option price")
    standard_error: float = Field(..., description="Monte Carlo standard error")
    payoffs: np.ndarray = Field(..., description="Individual payoffs")
    paths: np.ndarray = Field(..., description="Simulated paths")
    statistics: Dict[str, float] = Field(..., description="Additional statistics")
    execution_time: float = Field(..., description="Execution time in seconds")


@jit(nopython=True)
def asian_option_payoff(paths: np.ndarray, strike: float, option_type: str, averaging_type: str) -> np.ndarray:
    n_paths, n_steps = paths.shape
    payoffs = np.zeros(n_paths)
    
    for i in range(n_paths):
        if averaging_type == "arithmetic":
            average_price = np.mean(paths[i, :])
        else:  # geometric
            average_price = np.exp(np.mean(np.log(paths[i, :])))
        
        if option_type == "call":
            payoffs[i] = max(average_price - strike, 0)
        else:  # put
            payoffs[i] = max(strike - average_price, 0)
    
    return payoffs


@jit(nopython=True)
def barrier_option_payoff(paths: np.ndarray, strike: float, barrier: float, option_type: str, barrier_type: str) -> np.ndarray:
    n_paths, n_steps = paths.shape
    payoffs = np.zeros(n_paths)
    
    for i in range(n_paths):
        path = paths[i, :]
        final_price = path[-1]
        
        # Check barrier condition
        barrier_hit = False
        if "up" in barrier_type:
            barrier_hit = np.any(path >= barrier)
        else:  # down
            barrier_hit = np.any(path <= barrier)
        
        # Calculate payoff based on barrier type
        if "in" in barrier_type:
            # Knock-in: option becomes active if barrier is hit
            if barrier_hit:
                if option_type == "call":
                    payoffs[i] = max(final_price - strike, 0)
                else:  # put
                    payoffs[i] = max(strike - final_price, 0)
        else:  # out
            # Knock-out: option becomes worthless if barrier is hit
            if not barrier_hit:
                if option_type == "call":
                    payoffs[i] = max(final_price - strike, 0)
                else:  # put
                    payoffs[i] = max(strike - final_price, 0)
    
    return payoffs


@jit(nopython=True)
def lookback_option_payoff(paths: np.ndarray, option_type: str) -> np.ndarray:
    n_paths, n_steps = paths.shape
    payoffs = np.zeros(n_paths)
    
    for i in range(n_paths):
        path = paths[i, :]
        final_price = path[-1]
        
        if option_type == "call":
            # Lookback call: S_T - min(S_t)
            payoffs[i] = final_price - np.min(path)
        else:  # put
            # Lookback put: max(S_t) - S_T
            payoffs[i] = np.max(path) - final_price
    
    return payoffs


class PathDependentEngine:
    def __init__(self):
        self.logger = logger.bind(component="PathDependentEngine")

    async def price_option(self, parameters: Union[PathDependentParameters, Dict]) -> PathDependentResult:
        if isinstance(parameters, dict):
            parameters = PathDependentParameters(**parameters)

        start_time = asyncio.get_event_loop().time()

        try:
            # Generate underlying paths
            paths = await self._generate_paths(parameters)
            
            # Calculate payoffs based on option type
            if parameters.option_type.lower() == "asian":
                payoffs = await self._calculate_asian_payoffs(paths, parameters)
            elif parameters.option_type.lower() == "barrier":
                payoffs = await self._calculate_barrier_payoffs(paths, parameters)
            elif parameters.option_type.lower() == "lookback":
                payoffs = await self._calculate_lookback_payoffs(paths, parameters)
            else:
                raise MonteCarloException(f"Unknown option type: {parameters.option_type}")

            # Discount payoffs to present value
            discount_factor = np.exp(-parameters.mu * parameters.T)
            discounted_payoffs = payoffs * discount_factor
            
            option_price = np.mean(discounted_payoffs)
            standard_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
            
            statistics = self._calculate_statistics(discounted_payoffs, payoffs, paths)
            execution_time = asyncio.get_event_loop().time() - start_time

            result = PathDependentResult(
                option_price=float(option_price),
                standard_error=float(standard_error),
                payoffs=payoffs,
                paths=paths,
                statistics=statistics,
                execution_time=execution_time,
            )

            self.logger.info(
                "Path-dependent option priced",
                option_type=parameters.option_type,
                option_price=option_price,
                standard_error=standard_error,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"Path-dependent option pricing failed: {str(e)}")
            raise MonteCarloException(f"Path-dependent option pricing failed: {str(e)}")

    async def _generate_paths(self, parameters: PathDependentParameters) -> np.ndarray:
        if parameters.seed:
            np.random.seed(parameters.seed)

        dt = parameters.T / parameters.n_steps
        random_increments = np.random.standard_normal((parameters.n_paths, parameters.n_steps))
        
        paths = np.zeros((parameters.n_paths, parameters.n_steps + 1))
        paths[:, 0] = parameters.S0
        
        for i in range(parameters.n_steps):
            dW = random_increments[:, i] * np.sqrt(dt)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (parameters.mu - 0.5 * parameters.sigma**2) * dt + parameters.sigma * dW
            )
        
        return paths

    async def _calculate_asian_payoffs(self, paths: np.ndarray, parameters: PathDependentParameters) -> np.ndarray:
        if not parameters.strike:
            raise MonteCarloException("Strike price required for Asian options")
        
        averaging_type = parameters.averaging_type or "arithmetic"
        
        if "call" in parameters.option_type.lower():
            option_type = "call"
        else:
            option_type = "put"
        
        return asian_option_payoff(paths, parameters.strike, option_type, averaging_type)

    async def _calculate_barrier_payoffs(self, paths: np.ndarray, parameters: PathDependentParameters) -> np.ndarray:
        if not parameters.barrier_level or not parameters.barrier_type or not parameters.strike:
            raise MonteCarloException("Barrier level, barrier type, and strike required for barrier options")
        
        if "call" in parameters.option_type.lower():
            option_type = "call"
        else:
            option_type = "put"
        
        return barrier_option_payoff(paths, parameters.strike, parameters.barrier_level, option_type, parameters.barrier_type)

    async def _calculate_lookback_payoffs(self, paths: np.ndarray, parameters: PathDependentParameters) -> np.ndarray:
        if "call" in parameters.option_type.lower():
            option_type = "call"
        else:
            option_type = "put"
        
        return lookback_option_payoff(paths, option_type)

    def _calculate_statistics(self, discounted_payoffs: np.ndarray, payoffs: np.ndarray, paths: np.ndarray) -> Dict[str, float]:
        non_zero_payoffs = payoffs[payoffs > 0]
        
        return {
            "mean_payoff": float(np.mean(payoffs)),
            "std_payoff": float(np.std(payoffs)),
            "max_payoff": float(np.max(payoffs)),
            "min_payoff": float(np.min(payoffs)),
            "median_payoff": float(np.median(payoffs)),
            "probability_itm": float(np.mean(payoffs > 0)),
            "mean_itm_payoff": float(np.mean(non_zero_payoffs)) if len(non_zero_payoffs) > 0 else 0.0,
            "confidence_95_lower": float(np.percentile(discounted_payoffs, 2.5)),
            "confidence_95_upper": float(np.percentile(discounted_payoffs, 97.5)),
            "gamma_coefficient": float(self._calculate_gamma(payoffs)),
            "delta_estimate": float(self._estimate_delta(paths, payoffs)),
            "vega_estimate": float(self._estimate_vega(paths, payoffs)),
        }

    def _calculate_gamma(self, payoffs: np.ndarray) -> float:
        # Simple gamma approximation using payoff distribution
        return np.std(payoffs) / np.mean(payoffs) if np.mean(payoffs) > 0 else 0.0

    def _estimate_delta(self, paths: np.ndarray, payoffs: np.ndarray) -> float:
        # Estimate delta using finite difference approximation
        final_prices = paths[:, -1]
        
        # Simple linear regression approximation
        if len(final_prices) > 1 and np.std(final_prices) > 0:
            correlation = np.corrcoef(final_prices, payoffs)[0, 1]
            return correlation * np.std(payoffs) / np.std(final_prices)
        return 0.0

    def _estimate_vega(self, paths: np.ndarray, payoffs: np.ndarray) -> float:
        # Rough vega estimate based on path volatility sensitivity
        path_volatilities = np.std(np.diff(np.log(paths), axis=1), axis=1)
        
        if len(path_volatilities) > 1 and np.std(path_volatilities) > 0:
            correlation = np.corrcoef(path_volatilities, payoffs)[0, 1]
            return correlation * np.std(payoffs) / np.std(path_volatilities)
        return 0.0

    async def price_rainbow_option(
        self,
        initial_prices: List[float],
        drifts: List[float],
        volatilities: List[float],
        correlation_matrix: List[List[float]],
        T: float,
        strike: float,
        option_type: str = "best_of_call",
        n_paths: int = 100000,
        n_steps: int = 252,
    ) -> PathDependentResult:
        from src.monte_carlo_engine.multi_asset import MultiAssetEngine
        
        multi_asset_engine = MultiAssetEngine()
        
        # Use multi-asset engine to generate correlated paths
        from src.monte_carlo_engine.multi_asset import MultiAssetParameters
        
        multi_params = MultiAssetParameters(
            initial_prices=initial_prices,
            drifts=drifts,
            volatilities=volatilities,
            correlation_matrix=correlation_matrix,
            T=T,
            n_paths=n_paths,
            n_steps=n_steps,
        )
        
        multi_result = await multi_asset_engine.simulate_portfolio(multi_params)
        asset_paths = multi_result.asset_paths
        
        # Calculate rainbow option payoffs
        payoffs = self._calculate_rainbow_payoffs(asset_paths, strike, option_type)
        
        # Discount to present value
        discount_factor = np.exp(-drifts[0] * T)  # Use first asset's drift as risk-free rate
        discounted_payoffs = payoffs * discount_factor
        
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # Use first asset's paths for statistics calculation
        paths = asset_paths[0, :, :].T
        statistics = self._calculate_statistics(discounted_payoffs, payoffs, paths)

        return PathDependentResult(
            option_price=float(option_price),
            standard_error=float(standard_error),
            payoffs=payoffs,
            paths=paths,
            statistics=statistics,
            execution_time=0.0,  # Would be calculated properly in real implementation
        )

    def _calculate_rainbow_payoffs(self, asset_paths: np.ndarray, strike: float, option_type: str) -> np.ndarray:
        n_assets, n_paths, n_steps = asset_paths.shape
        final_prices = asset_paths[:, :, -1]  # Shape: (n_assets, n_paths)
        payoffs = np.zeros(n_paths)
        
        for i in range(n_paths):
            prices = final_prices[:, i]
            
            if option_type == "best_of_call":
                payoffs[i] = max(np.max(prices) - strike, 0)
            elif option_type == "worst_of_call":
                payoffs[i] = max(np.min(prices) - strike, 0)
            elif option_type == "best_of_put":
                payoffs[i] = max(strike - np.min(prices), 0)
            elif option_type == "worst_of_put":
                payoffs[i] = max(strike - np.max(prices), 0)
            elif option_type == "spread_call":
                payoffs[i] = max(np.max(prices) - np.min(prices) - strike, 0)
            elif option_type == "basket_call":
                basket_price = np.mean(prices)  # Equal weighted basket
                payoffs[i] = max(basket_price - strike, 0)
            elif option_type == "basket_put":
                basket_price = np.mean(prices)  # Equal weighted basket
                payoffs[i] = max(strike - basket_price, 0)
            else:
                raise MonteCarloException(f"Unknown rainbow option type: {option_type}")
        
        return payoffs

    async def price_exotic_option(
        self,
        parameters: PathDependentParameters,
        payoff_function: callable,
        **kwargs
    ) -> PathDependentResult:
        start_time = asyncio.get_event_loop().time()

        try:
            paths = await self._generate_paths(parameters)
            
            # Apply custom payoff function
            payoffs = np.array([payoff_function(path, **kwargs) for path in paths])
            
            discount_factor = np.exp(-parameters.mu * parameters.T)
            discounted_payoffs = payoffs * discount_factor
            
            option_price = np.mean(discounted_payoffs)
            standard_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
            
            statistics = self._calculate_statistics(discounted_payoffs, payoffs, paths)
            execution_time = asyncio.get_event_loop().time() - start_time

            result = PathDependentResult(
                option_price=float(option_price),
                standard_error=float(standard_error),
                payoffs=payoffs,
                paths=paths,
                statistics=statistics,
                execution_time=execution_time,
            )

            self.logger.info(
                "Exotic option priced",
                option_price=option_price,
                standard_error=standard_error,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"Exotic option pricing failed: {str(e)}")
            raise MonteCarloException(f"Exotic option pricing failed: {str(e)}")

    async def calculate_option_greeks(
        self, parameters: PathDependentParameters, bump_size: float = 0.01
    ) -> Dict[str, float]:
        base_result = await self.price_option(parameters)
        base_price = base_result.option_price
        
        # Delta - sensitivity to underlying price
        delta_params = parameters.copy(deep=True)
        delta_params.S0 = parameters.S0 * (1 + bump_size)
        delta_result = await self.price_option(delta_params)
        delta = (delta_result.option_price - base_price) / (parameters.S0 * bump_size)
        
        # Gamma - second derivative with respect to underlying price
        gamma_params = parameters.copy(deep=True)
        gamma_params.S0 = parameters.S0 * (1 - bump_size)
        gamma_result = await self.price_option(gamma_params)
        gamma = (delta_result.option_price - 2 * base_price + gamma_result.option_price) / ((parameters.S0 * bump_size) ** 2)
        
        # Vega - sensitivity to volatility
        vega_params = parameters.copy(deep=True)
        vega_params.sigma = parameters.sigma + bump_size
        vega_result = await self.price_option(vega_params)
        vega = (vega_result.option_price - base_price) / bump_size
        
        # Theta - time decay
        theta_params = parameters.copy(deep=True)
        theta_params.T = max(parameters.T - bump_size, 0.001)  # Avoid negative time
        theta_result = await self.price_option(theta_params)
        theta = (theta_result.option_price - base_price) / (-bump_size)  # Negative because time decreases
        
        # Rho - sensitivity to interest rate
        rho_params = parameters.copy(deep=True)
        rho_params.mu = parameters.mu + bump_size
        rho_result = await self.price_option(rho_params)
        rho = (rho_result.option_price - base_price) / bump_size

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }
