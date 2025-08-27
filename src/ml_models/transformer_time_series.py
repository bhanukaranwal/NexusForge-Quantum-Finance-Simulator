import asyncio
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TimeSeriesTransformerConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from pydantic import BaseModel, Field

from src.core.exceptions import MLModelException
from src.core.logging import logger


class TransformerConfig(BaseModel):
    sequence_length: int = Field(default=60, description="Input sequence length")
    prediction_horizon: int = Field(default=1, description="Number of steps to predict")
    d_model: int = Field(default=128, description="Model dimension")
    n_heads: int = Field(default=8, description="Number of attention heads")
    n_layers: int = Field(default=6, description="Number of transformer layers")
    dropout: float = Field(default=0.1, description="Dropout rate")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    epochs: int = Field(default=100, description="Training epochs")
    features: List[str] = Field(default_factory=list, description="Feature columns")
    target: str = Field(default="close", description="Target column")


class FinancialTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        features: List[str],
        target: str,
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features
        self.target = target
        
        # Normalize data
        self.feature_data = data[features].values
        self.target_data = data[target].values
        
        # Calculate normalization parameters
        self.feature_mean = np.mean(self.feature_data, axis=0)
        self.feature_std = np.std(self.feature_data, axis=0) + 1e-8
        self.target_mean = np.mean(self.target_data)
        self.target_std = np.std(self.target_data) + 1e-8
        
        # Normalize
        self.feature_data = (self.feature_data - self.feature_mean) / self.feature_std
        self.target_data = (self.target_data - self.target_mean) / self.target_std

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        x = torch.FloatTensor(
            self.feature_data[idx : idx + self.sequence_length]
        )
        y = torch.FloatTensor(
            self.target_data[
                idx + self.sequence_length : idx + self.sequence_length + self.prediction_horizon
            ]
        )
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class FinancialTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(len(config.features), config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.prediction_horizon)
        
        # Attention weights storage for interpretation
        self.attention_weights = None

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(x)
        
        # Global average pooling
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Output projection
        output = self.layer_norm(pooled_output)
        output = self.dropout(output)
        output = self.output_projection(output)
        
        return output

    def get_attention_weights(self):
        return self.attention_weights


class TransformerTimeSeriesModel:
    def __init__(self, config: Union[TransformerConfig, Dict]):
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger.bind(component="TransformerTimeSeriesModel")
        self.training_history = []
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    async def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        try:
            self.logger.info("Starting transformer training")
            
            # Prepare datasets
            train_dataset = FinancialTimeSeriesDataset(
                train_data,
                self.config.sequence_length,
                self.config.prediction_horizon,
                self.config.features,
                self.config.target,
            )
            
            # Store normalization parameters
            self.feature_mean = train_dataset.feature_mean
            self.feature_std = train_dataset.feature_std
            self.target_mean = train_dataset.target_mean
            self.target_std = train_dataset.target_std
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
            )
            
            val_loader = None
            if val_data is not None:
                val_dataset = FinancialTimeSeriesDataset(
                    val_data,
                    self.config.sequence_length,
                    self.config.prediction_horizon,
                    self.config.features,
                    self.config.target,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            
            # Initialize model
            self.model = FinancialTransformer(self.config).to(self.device)
            
            # Training setup
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5,
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
            
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / train_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    val_batches = 0
                    
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            predictions = self.model(batch_x)
                            loss = criterion(predictions, batch_y)
                            val_loss += loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    val_losses.append(avg_val_loss)
                    
                    # Learning rate scheduling
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(
                            self.model.state_dict(),
                            f"best_transformer_model_epoch_{epoch}.pth",
                        )
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs}: "
                        f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs}: Train Loss: {avg_train_loss:.6f}"
                    )
            
            self.training_history = {
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            
            self.logger.info("Transformer training completed")
            
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise MLModelException(f"Training failed: {str(e)}")

    async def predict(
        self, data: pd.DataFrame, return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise MLModelException("Model not trained. Call train() first.")
        
        try:
            self.model.eval()
            
            # Prepare dataset
            dataset = FinancialTimeSeriesDataset(
                data,
                self.config.sequence_length,
                self.config.prediction_horizon,
                self.config.features,
                self.config.target,
            )
            
            # Override normalization with training parameters
            dataset.feature_mean = self.feature_mean
            dataset.feature_std = self.feature_std
            dataset.target_mean = self.target_mean
            dataset.target_std = self.target_std
            
            # Normalize data
            dataset.feature_data = (dataset.data[self.config.features].values - self.feature_mean) / self.feature_std
            
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.device)
                    
                    if return_confidence:
                        # Monte Carlo Dropout for uncertainty estimation
                        self.model.train()  # Enable dropout
                        mc_predictions = []
                        
                        for _ in range(100):  # 100 MC samples
                            pred = self.model(batch_x)
                            mc_predictions.append(pred.cpu().numpy())
                        
                        mc_predictions = np.array(mc_predictions)
                        pred_mean = np.mean(mc_predictions, axis=0)
                        pred_std = np.std(mc_predictions, axis=0)
                        
                        predictions.append(pred_mean)
                        confidences.append(pred_std)
                        
                        self.model.eval()  # Disable dropout
                    else:
                        pred = self.model(batch_x)
                        predictions.append(pred.cpu().numpy())
            
            # Concatenate all predictions
            predictions = np.concatenate(predictions, axis=0)
            
            # Denormalize predictions
            predictions = predictions * self.target_std + self.target_mean
            
            result = {"predictions": predictions}
            
            if return_confidence:
                confidences = np.concatenate(confidences, axis=0)
                confidences = confidences * self.target_std  # Scale uncertainty
                result["confidence_intervals"] = confidences
                result["lower_bound"] = predictions - 1.96 * confidences
                result["upper_bound"] = predictions + 1.96 * confidences
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise MLModelException(f"Prediction failed: {str(e)}")

    def get_attention_analysis(self, data: pd.DataFrame, sample_idx: int = 0) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise MLModelException("Model not trained. Call train() first.")
        
        try:
            # Prepare single sample
            dataset = FinancialTimeSeriesDataset(
                data,
                self.config.sequence_length,
                self.config.prediction_horizon,
                self.config.features,
                self.config.target,
            )
            
            sample_x, _ = dataset[sample_idx]
            sample_x = sample_x.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Forward pass with attention extraction
            self.model.eval()
            
            # Hook to capture attention weights
            attention_weights = []
            
            def hook_fn(module, input, output):
                if hasattr(output, 'attention_weights'):
                    attention_weights.append(output.attention_weights)
            
            # Register hooks on transformer layers
            hooks = []
            for layer in self.model.transformer_encoder.layers:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                _ = self.model(sample_x)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Process attention weights
            if attention_weights:
                # Average attention weights across layers and heads
                avg_attention = torch.stack(attention_weights).mean(dim=[0, 2])  # Average over layers and heads
                avg_attention = avg_attention.cpu().numpy()[0]  # Remove batch dimension
            else:
                # Fallback: uniform attention
                avg_attention = np.ones((self.config.sequence_length, self.config.sequence_length))
                avg_attention = avg_attention / np.sum(avg_attention, axis=1, keepdims=True)
            
            return {
                "attention_matrix": avg_attention,
                "feature_importance": np.mean(avg_attention, axis=0),  # Importance per timestep
                "temporal_focus": np.mean(avg_attention, axis=1),  # Focus per query position
            }
            
        except Exception as e:
            self.logger.error(f"Attention analysis failed: {str(e)}")
            raise MLModelException(f"Attention analysis failed: {str(e)}")

    async def fine_tune(
        self,
        new_data: pd.DataFrame,
        learning_rate: float = 1e-5,
        epochs: int = 10,
    ) -> Dict[str, List[float]]:
        if self.model is None:
            raise MLModelException("Model not trained. Call train() first.")
        
        try:
            self.logger.info("Starting fine-tuning")
            
            # Prepare dataset
            dataset = FinancialTimeSeriesDataset(
                new_data,
                self.config.sequence_length,
                self.config.prediction_horizon,
                self.config.features,
                self.config.target,
            )
            
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True
            )
            
            # Fine-tuning setup
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=1e-6
            )
            criterion = nn.MSELoss()
            
            fine_tune_losses = []
            
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                batches = 0
                
                for batch_x, batch_y in dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # Smaller gradient clipping for fine-tuning
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batches += 1
                
                avg_loss = epoch_loss / batches
                fine_tune_losses.append(avg_loss)
                
                self.logger.info(f"Fine-tune Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.6f}")
            
            self.logger.info("Fine-tuning completed")
            
            return {"fine_tune_losses": fine_tune_losses}
            
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {str(e)}")
            raise MLModelException(f"Fine-tuning failed: {str(e)}")

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise MLModelException("No model to save")
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "training_history": self.training_history,
        }, path)
        
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = TransformerConfig(**checkpoint["config"])
        self.model = FinancialTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.feature_mean = checkpoint.get("feature_mean")
        self.feature_std = checkpoint.get("feature_std")
        self.target_mean = checkpoint.get("target_mean")
        self.target_std = checkpoint.get("target_std")
        self.training_history = checkpoint.get("training_history", [])
        
        self.logger.info(f"Model loaded from {path}")

    async def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
    ) -> Dict[str, float]:
        try:
            predictions_result = await self.predict(data, return_confidence=True)
            predictions = predictions_result["predictions"]
            
            # Simple trading strategy: long when prediction > current price
            actual_prices = data[self.config.target].values
            returns = []
            positions = []
            capital = initial_capital
            portfolio_values = [capital]
            
            for i in range(len(predictions)):
                if i + self.config.sequence_length < len(actual_prices):
                    current_price = actual_prices[i + self.config.sequence_length - 1]
                    predicted_price = predictions[i, 0]  # First prediction
                    
                    # Simple strategy: long if predicted return > 0
                    predicted_return = (predicted_price - current_price) / current_price
                    
                    if predicted_return > 0.02:  # Go long if prediction > 2% return
                        position = 1
                    elif predicted_return < -0.02:  # Go short if prediction < -2% return
                        position = -1
                    else:
                        position = 0  # Stay flat
                    
                    positions.append(position)
                    
                    # Calculate returns if we have next price
                    if i + self.config.sequence_length < len(actual_prices) - 1:
                        next_price = actual_prices[i + self.config.sequence_length]
                        actual_return = (next_price - current_price) / current_price
                        
                        # Apply transaction costs
                        if position != 0:
                            actual_return -= transaction_cost
                        
                        portfolio_return = position * actual_return
                        returns.append(portfolio_return)
                        
                        capital *= (1 + portfolio_return)
                        portfolio_values.append(capital)
            
            returns = np.array(returns)
            portfolio_values = np.array(portfolio_values)
            
            # Calculate metrics
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
            
            win_rate = np.mean(np.array(returns) > 0)
            
            return {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "total_trades": len(returns),
                "final_capital": float(portfolio_values[-1]),
            }
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {str(e)}")
            raise MLModelException(f"Backtesting failed: {str(e)}")
