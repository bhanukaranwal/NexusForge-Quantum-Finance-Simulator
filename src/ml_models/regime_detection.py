import asyncio
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from pydantic import BaseModel, Field

from src.core.exceptions import MLModelException
from src.core.logging import logger


class RegimeDetectionConfig(BaseModel):
    n_regimes: int = Field(default=3, description="Number of market regimes")
    features: List[str] = Field(default_factory=list, description="Features for regime detection")
    lookback_window: int = Field(default=60, description="Lookback window for features")
    method: str = Field(default="hmm", description="Method: hmm, gmm, or kmeans")
    covariance_type: str = Field(default="full", description="HMM covariance type")
    max_iterations: int = Field(default=100, description="Maximum EM iterations")
    tolerance: float = Field(default=1e-4, description="Convergence tolerance")
    random_state: int = Field(default=42, description="Random state for reproducibility")


class RegimeResult(BaseModel):
    regimes: np.ndarray = Field(..., description="Detected regime sequence")
    probabilities: np.ndarray = Field(..., description="Regime probabilities")
    regime_stats: Dict[int, Dict[str, float]] = Field(..., description="Statistics per regime")
    transition_matrix: np.ndarray = Field(..., description="Regime transition matrix")
    model_score: float = Field(..., description="Model likelihood score")
    regime_names: Dict[int, str] = Field(..., description="Regime interpretations")


class RegimeDetector:
    def __init__(self, config: Union[RegimeDetectionConfig, Dict]):
        if isinstance(config, dict):
            config = RegimeDetectionConfig(**config)
        
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_data = None
        self.logger = logger.bind(component="RegimeDetector")
        
    async def fit_predict(
        self, 
        data: pd.DataFrame, 
        target_column: str = "returns"
    ) -> RegimeResult:
        try:
            self.logger.info(f"Starting regime detection with {self.config.method} method")
            
            # Prepare features
            features = await self._prepare_features(data, target_column)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            self.feature_data = scaled_features
            
            # Fit model based on method
            if self.config.method == "hmm":
                regimes, probabilities, model_score = await self._fit_hmm(scaled_features)
            elif self.config.method == "gmm":
                regimes, probabilities, model_score = await self._fit_gmm(scaled_features)
            elif self.config.method == "kmeans":
                regimes, probabilities, model_score = await self._fit_kmeans(scaled_features)
            else:
                raise MLModelException(f"Unknown method: {self.config.method}")
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(
                data, regimes, target_column
            )
            
            # Calculate transition matrix
            transition_matrix = self._calculate_transition_matrix(regimes)
            
            # Generate regime names/interpretations
            regime_names = self._generate_regime_names(regime_stats)
            
            result = RegimeResult(
                regimes=regimes,
                probabilities=probabilities,
                regime_stats=regime_stats,
                transition_matrix=transition_matrix,
                model_score=model_score,
                regime_names=regime_names,
            )
            
            self.logger.info(
                "Regime detection completed",
                n_regimes=self.config.n_regimes,
                model_score=model_score,
                method=self.config.method,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {str(e)}")
            raise MLModelException(f"Regime detection failed: {str(e)}")

    async def _prepare_features(self, data: pd.DataFrame, target_column: str) -> np.ndarray:
        features_df = data.copy()
        
        # Calculate returns if not present
        if target_column not in features_df.columns:
            if "close" in features_df.columns:
                features_df[target_column] = features_df["close"].pct_change()
            else:
                raise MLModelException(f"Target column '{target_column}' not found")
        
        # Technical indicators for regime detection
        feature_columns = []
        
        # Returns-based features
        features_df["returns_ma_5"] = features_df[target_column].rolling(5).mean()
        features_df["returns_ma_20"] = features_df[target_column].rolling(20).mean()
        features_df["returns_std_20"] = features_df[target_column].rolling(20).std()
        features_df["returns_skew_20"] = features_df[target_column].rolling(20).skew()
        features_df["returns_kurt_20"] = features_df[target_column].rolling(20).kurt()
        
        feature_columns.extend([
            "returns_ma_5", "returns_ma_20", "returns_std_20", 
            "returns_skew_20", "returns_kurt_20"
        ])
        
        # Volatility features
        features_df["volatility"] = features_df[target_column].rolling(20).std()
        features_df["volatility_ma"] = features_df["volatility"].rolling(10).mean()
        features_df["vol_of_vol"] = features_df["volatility"].rolling(10).std()
        
        feature_columns.extend(["volatility", "volatility_ma", "vol_of_vol"])
        
        # Price-based features (if price data available)
        if "close" in features_df.columns:
            features_df["price_ma_20"] = features_df["close"].rolling(20).mean()
            features_df["price_ma_50"] = features_df["close"].rolling(50).mean()
            features_df["price_momentum"] = features_df["close"] / features_df["close"].shift(20)
            features_df["price_trend"] = (features_df["price_ma_20"] / features_df["price_ma_50"]).fillna(1)
            
            feature_columns.extend(["price_momentum", "price_trend"])
        
        # Volume features (if available)
        if "volume" in features_df.columns:
            features_df["volume_ma"] = features_df["volume"].rolling(20).mean()
            features_df["volume_ratio"] = features_df["volume"] / features_df["volume_ma"]
            feature_columns.extend(["volume_ratio"])
        
        # Select final features
        if self.config.features:
            # Use user-specified features
            available_features = [f for f in self.config.features if f in features_df.columns]
            if not available_features:
                raise MLModelException("None of the specified features are available")
            final_features = available_features
        else:
            # Use automatically generated features
            final_features = feature_columns
        
        # Extract feature matrix
        feature_matrix = features_df[final_features].values
        
        # Remove rows with NaN values
        valid_rows = ~np.isnan(feature_matrix).any(axis=1)
        feature_matrix = feature_matrix[valid_rows]
        
        if len(feature_matrix) == 0:
            raise MLModelException("No valid feature data after preprocessing")
        
        return feature_matrix

    async def _fit_hmm(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        try:
            # Initialize HMM model
            model = hmm.GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.max_iterations,
                tol=self.config.tolerance,
                random_state=self.config.random_state,
            )
            
            # Fit model
            model.fit(features)
            
            # Predict regimes and probabilities
            regimes = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Calculate model score
            model_score = model.score(features)
            
            self.model = model
            
            return regimes, probabilities, model_score
            
        except Exception as e:
            raise MLModelException(f"HMM fitting failed: {str(e)}")

    async def _fit_gmm(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        try:
            # Initialize GMM model
            model = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                max_iter=self.config.max_iterations,
                tol=self.config.tolerance,
                random_state=self.config.random_state,
            )
            
            # Fit model
            model.fit(features)
            
            # Predict regimes and probabilities
            regimes = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Calculate model score
            model_score = model.score(features)
            
            self.model = model
            
            return regimes, probabilities, model_score
            
        except Exception as e:
            raise MLModelException(f"GMM fitting failed: {str(e)}")

    async def _fit_kmeans(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        try:
            # Initialize K-means model
            model = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=self.config.random_state,
                max_iter=self.config.max_iterations,
                tol=self.config.tolerance,
            )
            
            # Fit model
            model.fit(features)
            
            # Predict regimes
            regimes = model.predict(features)
            
            # For K-means, we need to generate probabilities
            distances = model.transform(features)
            inv_distances = 1.0 / (distances + 1e-8)
            probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)
            
            # Calculate model score (negative inertia)
            model_score = -model.inertia_
            
            self.model = model
            
            return regimes, probabilities, model_score
            
        except Exception as e:
            raise MLModelException(f"K-means fitting failed: {str(e)}")

    def _calculate_regime_statistics(
        self, 
        data: pd.DataFrame, 
        regimes: np.ndarray, 
        target_column: str
    ) -> Dict[int, Dict[str, float]]:
        # Align regimes with original data
        regime_stats = {}
        
        # Calculate returns if not present
        if target_column not in data.columns and "close" in data.columns:
            returns = data["close"].pct_change().dropna()
        else:
            returns = data[target_column].dropna()
        
        # Align lengths
        min_length = min(len(returns), len(regimes))
        returns = returns.iloc[-min_length:].values
        regimes = regimes[-min_length:]
        
        for regime_id in range(self.config.n_regimes):
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats[regime_id] = {
                    "mean_return": float(np.mean(regime_returns)),
                    "volatility": float(np.std(regime_returns)),
                    "skewness": float(self._calculate_skewness(regime_returns)),
                    "kurtosis": float(self._calculate_kurtosis(regime_returns)),
                    "sharpe_ratio": float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)),
                    "max_drawdown": float(self._calculate_max_drawdown(regime_returns)),
                    "frequency": float(np.sum(regime_mask) / len(regimes)),
                    "avg_duration": float(self._calculate_average_duration(regimes, regime_id)),
                }
            else:
                regime_stats[regime_id] = {
                    "mean_return": 0.0,
                    "volatility": 0.0,
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "frequency": 0.0,
                    "avg_duration": 0.0,
                }
        
        return regime_stats

    def _calculate_skewness(self, data: np.ndarray) -> float:
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        
        return np.min(drawdowns)

    def _calculate_average_duration(self, regimes: np.ndarray, regime_id: int) -> float:
        if len(regimes) == 0:
            return 0.0
        
        durations = []
        current_duration = 0
        
        for regime in regimes:
            if regime == regime_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0

    def _calculate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        n_regimes = self.config.n_regimes
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_counts[current_regime, next_regime] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_counts / row_sums
        
        return transition_matrix

    def _generate_regime_names(self, regime_stats: Dict[int, Dict[str, float]]) -> Dict[int, str]:
        regime_names = {}
        
        # Sort regimes by mean return
        sorted_regimes = sorted(
            regime_stats.items(),
            key=lambda x: x[1]["mean_return"]
        )
        
        for i, (regime_id, stats) in enumerate(sorted_regimes):
            mean_return = stats["mean_return"]
            volatility = stats["volatility"]
            
            if mean_return > 0.001:  # > 0.1% daily
                if volatility > 0.02:  # > 2% daily vol
                    name = "High Volatility Bull"
                else:
                    name = "Low Volatility Bull"
            elif mean_return < -0.001:  # < -0.1% daily
                if volatility > 0.02:
                    name = "High Volatility Bear"
                else:
                    name = "Low Volatility Bear"
            else:
                if volatility > 0.02:
                    name = "High Volatility Neutral"
                else:
                    name = "Low Volatility Neutral"
            
            regime_names[regime_id] = name
        
        return regime_names

    async def predict_regime(
        self, 
        new_data: pd.DataFrame, 
        target_column: str = "returns"
    ) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise MLModelException("Model not fitted. Call fit_predict() first.")
        
        try:
            # Prepare features for new data
            features = await self._prepare_features(new_data, target_column)
            scaled_features = self.scaler.transform(features)
            
            # Predict regimes
            if hasattr(self.model, 'predict'):
                regimes = self.model.predict(scaled_features)
                probabilities = self.model.predict_proba(scaled_features)
            else:
                raise MLModelException("Model does not support prediction")
            
            return {
                "regimes": regimes,
                "probabilities": probabilities,
            }
            
        except Exception as e:
            self.logger.error(f"Regime prediction failed: {str(e)}")
            raise MLModelException(f"Regime prediction failed: {str(e)}")

    async def regime_forecast(
        self, 
        current_regime: int, 
        n_steps: int = 5
    ) -> Dict[str, np.ndarray]:
        if not hasattr(self, 'transition_matrix'):
            raise MLModelException("Transition matrix not available. Run fit_predict() first.")
        
        try:
            # Initialize with current regime probability
            current_prob = np.zeros(self.config.n_regimes)
            current_prob[current_regime] = 1.0
            
            forecasts = [current_prob.copy()]
            
            # Forecast using transition matrix
            for _ in range(n_steps):
                next_prob = current_prob @ self.transition_matrix
                forecasts.append(next_prob.copy())
                current_prob = next_prob
            
            forecasts = np.array(forecasts)
            
            # Most likely regime sequence
            likely_regimes = np.argmax(forecasts, axis=1)
            
            return {
                "regime_probabilities": forecasts,
                "likely_regimes": likely_regimes,
                "confidence": np.max(forecasts, axis=1),
            }
            
        except Exception as e:
            self.logger.error(f"Regime forecasting failed: {str(e)}")
            raise MLModelException(f"Regime forecasting failed: {str(e)}")

    def get_regime_insights(self, result: RegimeResult) -> Dict[str, any]:
        insights = {
            "dominant_regime": int(np.argmax([stats["frequency"] for stats in result.regime_stats.values()])),
            "regime_persistence": {},
            "regime_transitions": {},
            "volatility_ranking": {},
            "return_ranking": {},
        }
        
        # Regime persistence (diagonal elements of transition matrix)
        for i in range(self.config.n_regimes):
            insights["regime_persistence"][i] = float(result.transition_matrix[i, i])
        
        # Most likely transitions
        for i in range(self.config.n_regimes):
            non_diagonal = [(j, result.transition_matrix[i, j]) for j in range(self.config.n_regimes) if i != j]
            most_likely = max(non_diagonal, key=lambda x: x[1]) if non_diagonal else (i, 0.0)
            insights["regime_transitions"][i] = {
                "most_likely_next": int(most_likely[0]),
                "probability": float(most_likely[1])
            }
        
        # Rankings
        regimes_by_vol = sorted(
            result.regime_stats.items(),
            key=lambda x: x[1]["volatility"],
            reverse=True
        )
        
        regimes_by_return = sorted(
            result.regime_stats.items(),
            key=lambda x: x[1]["mean_return"],
            reverse=True
        )
        
        insights["volatility_ranking"] = [int(r[0]) for r in regimes_by_vol]
        insights["return_ranking"] = [int(r[0]) for r in regimes_by_return]
        
        return insights

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise MLModelException("No model to save")
        
        import joblib
        
        model_data = {
            "model": self.model,
            "config": self.config.dict(),
            "scaler": self.scaler,
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.config = RegimeDetectionConfig(**model_data["config"])
        self.scaler = model_data["scaler"]
        
        self.logger.info(f"Model loaded from {path}")
