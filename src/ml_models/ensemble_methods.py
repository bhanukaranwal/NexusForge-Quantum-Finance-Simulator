import asyncio
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pydantic import BaseModel, Field

from src.core.exceptions import MLModelException
from src.core.logging import logger


class EnsembleConfig(BaseModel):
    models: List[str] = Field(
        default=["random_forest", "xgboost", "lightgbm", "neural_network"],
        description="Base models to include"
    )
    ensemble_method: str = Field(
        default="voting", description="Ensemble method: voting, stacking, blending"
    )
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    test_size: float = Field(default=0.2, description="Test set proportion")
    random_state: int = Field(default=42, description="Random state")
    feature_selection: bool = Field(default=True, description="Enable feature selection")
    hyperparameter_tuning: bool = Field(default=True, description="Enable hyperparameter tuning")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")


class ModelPerformance(BaseModel):
    model_name: str
    mse: float
    mae: float
    r2: float
    cv_score: float
    cv_std: float
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class EnsembleResult(BaseModel):
    predictions: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    performance: ModelPerformance
    individual_performance: List[ModelPerformance]
    feature_importance: Dict[str, float]
    model_weights: Dict[str, float] = Field(default_factory=dict)


class EnsemblePredictor:
    def __init__(self, config: Union[EnsembleConfig, Dict]):
        if isinstance(config, dict):
            config = EnsembleConfig(**config)
        
        self.config = config
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.logger = logger.bind(component="EnsemblePredictor")
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize base models with default parameters"""
        models = {}
        
        if "linear_regression" in self.config.models:
            models["linear_regression"] = LinearRegression()
        
        if "ridge" in self.config.models:
            models["ridge"] = Ridge(alpha=1.0, random_state=self.config.random_state)
        
        if "lasso" in self.config.models:
            models["lasso"] = Lasso(alpha=1.0, random_state=self.config.random_state)
        
        if "elastic_net" in self.config.models:
            models["elastic_net"] = ElasticNet(alpha=1.0, random_state=self.config.random_state)
        
        if "random_forest" in self.config.models:
            models["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        if "gradient_boosting" in self.config.models:
            models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.config.random_state
            )
        
        if "xgboost" in self.config.models:
            models["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=0
            )
        
        if "lightgbm" in self.config.models:
            models["lightgbm"] = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=-1
            )
        
        if "svm" in self.config.models:
            models["svm"] = SVR(kernel='rbf', C=1.0)
        
        if "neural_network" in self.config.models:
            models["neural_network"] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                random_state=self.config.random_state,
                max_iter=1000
            )
        
        return models

    async def fit(self, X: pd.DataFrame, y: pd.Series, 
                  optimize_hyperparameters: bool = None) -> Dict[str, Any]:
        """Fit the ensemble model"""
        try:
            self.logger.info("Starting ensemble model training")
            
            if optimize_hyperparameters is None:
                optimize_hyperparameters = self.config.hyperparameter_tuning
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Feature selection if enabled
            if self.config.feature_selection:
                X_scaled_df = await self._feature_selection(X_scaled_df, y)
                self.feature_names = list(X_scaled_df.columns)
            
            # Initialize models
            self.models = self._initialize_models()
            
            # Hyperparameter tuning if enabled
            if optimize_hyperparameters:
                self.models = await self._tune_hyperparameters(X_scaled_df, y)
            
            # Fit ensemble based on method
            if self.config.ensemble_method == "voting":
                await self._fit_voting_ensemble(X_scaled_df, y)
            elif self.config.ensemble_method == "stacking":
                await self._fit_stacking_ensemble(X_scaled_df, y)
            elif self.config.ensemble_method == "blending":
                await self._fit_blending_ensemble(X_scaled_df, y)
            else:
                raise MLModelException(f"Unknown ensemble method: {self.config.ensemble_method}")
            
            # Evaluate individual models
            individual_performance = await self._evaluate_individual_models(X_scaled_df, y)
            
            self.is_fitted = True
            
            self.logger.info("Ensemble model training completed")
            
            return {
                "individual_performance": individual_performance,
                "ensemble_method": self.config.ensemble_method,
                "selected_features": self.feature_names,
                "n_models": len(self.models),
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble fitting failed: {str(e)}")
            raise MLModelException(f"Ensemble fitting failed: {str(e)}")

    async def _feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select important features using multiple methods"""
        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.ensemble import RandomForestRegressor
        
        # Method 1: Statistical tests
        k_best = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
        k_best.fit(X, y)
        statistical_features = X.columns[k_best.get_support()].tolist()
        
        # Method 2: Tree-based feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        rf.fit(X, y)
        importance_scores = pd.Series(rf.feature_importances_, index=X.columns)
        tree_features = importance_scores.nlargest(min(20, X.shape[1])).index.tolist()
        
        # Method 3: Recursive Feature Elimination
        rfe = RFE(
            estimator=RandomForestRegressor(n_estimators=50, random_state=self.config.random_state),
            n_features_to_select=min(15, X.shape[1])
        )
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_].tolist()
        
        # Combine features (union of all methods)
        selected_features = list(set(statistical_features + tree_features + rfe_features))
        
        self.logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        return X[selected_features]

    async def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Tune hyperparameters for each model"""
        from sklearn.model_selection import RandomizedSearchCV
        
        tuned_models = {}
        
        # Hyperparameter grids
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 100]
            },
            "ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0]
            },
            "svm": {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto", 0.001, 0.01],
                "epsilon": [0.01, 0.1, 0.2]
            },
            "neural_network": {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01, 0.1]
            }
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        for model_name, model in self.models.items():
            if model_name in param_grids:
                self.logger.info(f"Tuning hyperparameters for {model_name}")
                
                try:
                    search = RandomizedSearchCV(
                        model,
                        param_grids[model_name],
                        n_iter=20,
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs
                    )
                    
                    search.fit(X, y)
                    tuned_models[model_name] = search.best_estimator_
                    
                    self.logger.info(f"Best parameters for {model_name}: {search.best_params_}")
                    
                except Exception as e:
                    self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
                    tuned_models[model_name] = model
            else:
                tuned_models[model_name] = model
        
        return tuned_models

    async def _fit_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Fit voting ensemble"""
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = VotingRegressor(
            estimators=estimators,
            n_jobs=self.config.n_jobs
        )
        
        self.ensemble_model.fit(X, y)

    async def _fit_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Fit stacking ensemble"""
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression
        
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
            n_jobs=self.config.n_jobs
        )
        
        self.ensemble_model.fit(X, y)

    async def _fit_blending_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Fit blending ensemble"""
        from sklearn.model_selection import train_test_split
        
        # Split into train and blend sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state
        )
        
        # Train base models on train set
        blend_features = np.zeros((X_blend.shape[0], len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            model.fit(X_train, y_train)
            blend_features[:, i] = model.predict(X_blend)
        
        # Train meta-model on blend set
        self.meta_model = LinearRegression()
        self.meta_model.fit(blend_features, y_blend)
        
        # Retrain base models on full dataset
        for model in self.models.values():
            model.fit(X, y)

    async def _evaluate_individual_models(self, X: pd.DataFrame, y: pd.Series) -> List[ModelPerformance]:
        """Evaluate individual model performance"""
        performances = []
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        for name, model in self.models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=self.config.n_jobs
            )
            
            # Fit model for other metrics
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importance_values))
            elif hasattr(model, 'coef_'):
                importance_values = np.abs(model.coef_)
                feature_importance = dict(zip(self.feature_names, importance_values))
            
            performance = ModelPerformance(
                model_name=name,
                mse=mse,
                mae=mae,
                r2=r2,
                cv_score=-cv_scores.mean(),
                cv_std=cv_scores.std(),
                feature_importance=feature_importance
            )
            
            performances.append(performance)
        
        return performances

    async def predict(self, X: pd.DataFrame) -> EnsembleResult:
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise MLModelException("Model not fitted. Call fit() first.")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Select features used during training
            X_scaled_df = X_scaled_df[self.feature_names]
            
            # Get individual predictions
            individual_predictions = {}
            for name, model in self.models.items():
                individual_predictions[name] = model.predict(X_scaled_df)
            
            # Get ensemble predictions
            if self.config.ensemble_method in ["voting", "stacking"]:
                ensemble_predictions = self.ensemble_model.predict(X_scaled_df)
            elif self.config.ensemble_method == "blending":
                blend_features = np.column_stack(list(individual_predictions.values()))
                ensemble_predictions = self.meta_model.predict(blend_features)
            
            # Calculate model weights (for voting ensemble)
            model_weights = {}
            if hasattr(self.ensemble_model, 'named_estimators_'):
                total_weight = len(self.models)
                for name in self.models.keys():
                    model_weights[name] = 1.0 / total_weight
            
            # Aggregate feature importance
            feature_importance = await self._aggregate_feature_importance()
            
            result = EnsembleResult(
                predictions=ensemble_predictions,
                individual_predictions=individual_predictions,
                performance=ModelPerformance(
                    model_name="ensemble",
                    mse=0.0,  # Would be calculated if ground truth is available
                    mae=0.0,
                    r2=0.0,
                    cv_score=0.0,
                    cv_std=0.0
                ),
                individual_performance=[],  # Would be populated if needed
                feature_importance=feature_importance,
                model_weights=model_weights
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise MLModelException(f"Prediction failed: {str(e)}")

    async def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across models"""
        aggregated_importance = {}
        total_models = 0
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                for feature, importance in zip(self.feature_names, importance_values):
                    aggregated_importance[feature] = aggregated_importance.get(feature, 0) + importance
                total_models += 1
            elif hasattr(model, 'coef_'):
                importance_values = np.abs(model.coef_)
                for feature, importance in zip(self.feature_names, importance_values):
                    aggregated_importance[feature] = aggregated_importance.get(feature, 0) + importance
                total_models += 1
        
        # Normalize by number of models that contributed
        if total_models > 0:
            for feature in aggregated_importance:
                aggregated_importance[feature] /= total_models
        
        return aggregated_importance

    async def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            if self.config.feature_selection:
                X_scaled_df = await self._feature_selection(X_scaled_df, y)
            
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            cv_results = {}
            
            # Initialize models
            models = self._initialize_models()
            
            for name, model in models.items():
                scores = cross_val_score(
                    model, X_scaled_df, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=self.config.n_jobs
                )
                
                cv_results[name] = {
                    "mean_score": -scores.mean(),
                    "std_score": scores.std(),
                    "scores": -scores
                }
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise MLModelException(f"Cross-validation failed: {str(e)}")

    async def feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance across different methods"""
        try:
            # Fit models first
            await self.fit(X, y)
            
            importance_analysis = {
                "model_importance": {},
                "consensus_importance": {},
                "feature_rankings": {}
            }
            
            # Get importance from each model
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_values = model.feature_importances_
                    importance_analysis["model_importance"][name] = dict(zip(self.feature_names, importance_values))
                elif hasattr(model, 'coef_'):
                    importance_values = np.abs(model.coef_)
                    importance_analysis["model_importance"][name] = dict(zip(self.feature_names, importance_values))
            
            # Calculate consensus importance
            all_importances = pd.DataFrame(importance_analysis["model_importance"]).fillna(0)
            importance_analysis["consensus_importance"] = all_importances.mean(axis=1).to_dict()
            
            # Feature rankings
            consensus_ranking = sorted(
                importance_analysis["consensus_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            importance_analysis["feature_rankings"] = [feature for feature, _ in consensus_ranking]
            
            return importance_analysis
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {str(e)}")
            raise MLModelException(f"Feature importance analysis failed: {str(e)}")

    def save_model(self, path: str) -> None:
        """Save the ensemble model"""
        if not self.is_fitted:
            raise MLModelException("Model not fitted. Cannot save.")
        
        import joblib
        
        model_data = {
            "config": self.config.dict(),
            "models": self.models,
            "ensemble_model": getattr(self, 'ensemble_model', None),
            "meta_model": getattr(self, 'meta_model', None),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Ensemble model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load the ensemble model"""
        import joblib
        
        model_data = joblib.load(path)
        
        self.config = EnsembleConfig(**model_data["config"])
        self.models = model_data["models"]
        self.ensemble_model = model_data.get("ensemble_model")
        self.meta_model = model_data.get("meta_model")
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = model_data["is_fitted"]
        
        self.logger.info(f"Ensemble model loaded from {path}")

    async def backtest_ensemble(
        self,
        data: pd.DataFrame,
        target_column: str,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        rebalance_frequency: int = 1
    ) -> Dict[str, Any]:
        """Backtest the ensemble trading strategy"""
        try:
            # Prepare features and target
            features = data.drop(columns=[target_column])
            target = data[target_column]
            
            # Fit the ensemble
            await self.fit(features.iloc[:-252], target.iloc[:-252])  # Use all but last year for training
            
            # Make predictions for the test period
            test_features = features.iloc[-252:]
            test_target = target.iloc[-252:]
            
            result = await self.predict(test_features)
            predictions = result.predictions
            
            # Simple trading strategy based on predictions
            positions = []
            portfolio_values = [initial_capital]
            current_capital = initial_capital
            current_position = 0
            
            for i, prediction in enumerate(predictions):
                if i == 0:
                    continue
                
                actual_return = test_target.iloc[i]
                predicted_return = prediction
                
                # Simple strategy: long if positive prediction, short if negative
                if predicted_return > 0.02:  # Buy signal
                    target_position = 1
                elif predicted_return < -0.02:  # Sell signal
                    target_position = -1
                else:  # Hold
                    target_position = 0
                
                # Calculate position change and transaction cost
                position_change = abs(target_position - current_position)
                transaction_fee = position_change * transaction_cost * current_capital
                
                # Update portfolio
                if i % rebalance_frequency == 0:  # Rebalance
                    portfolio_return = current_position * actual_return
                    current_capital = current_capital * (1 + portfolio_return) - transaction_fee
                    current_position = target_position
                
                positions.append(current_position)
                portfolio_values.append(current_capital)
            
            # Calculate performance metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            max_drawdown = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values))
            
            # Calculate hit rate
            prediction_accuracy = np.mean((predictions[1:] > 0) == (test_target.iloc[1:].values > 0))
            
            return {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "hit_rate": float(prediction_accuracy),
                "final_capital": float(portfolio_values[-1]),
                "n_trades": len(positions),
                "portfolio_values": portfolio_values.tolist(),
                "positions": positions,
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble backtesting failed: {str(e)}")
            raise MLModelException(f"Ensemble backtesting failed: {str(e)}")
