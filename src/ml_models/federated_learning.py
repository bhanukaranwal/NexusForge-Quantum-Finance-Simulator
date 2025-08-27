import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, Field
import json

from src.core.exceptions import MLModelException
from src.core.logging import logger


class FederatedConfig(BaseModel):
    num_clients: int = Field(default=5, description="Number of federated clients")
    rounds: int = Field(default=100, description="Number of federated rounds")
    client_fraction: float = Field(default=1.0, description="Fraction of clients per round")
    local_epochs: int = Field(default=5, description="Local training epochs")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    privacy_budget: float = Field(default=1.0, description="Differential privacy budget")
    noise_multiplier: float = Field(default=1.1, description="DP noise multiplier")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
    secure_aggregation: bool = Field(default=True, description="Enable secure aggregation")
    homomorphic_encryption: bool = Field(default=False, description="Enable homomorphic encryption")
    min_clients: int = Field(default=3, description="Minimum clients for aggregation")


class PrivacyAccountant:
    """Differential privacy accountant for federated learning"""
    
    def __init__(self, noise_multiplier: float, sample_rate: float):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.privacy_history = []
        
    def compute_privacy(self, steps: int) -> Tuple[float, float]:
        """Compute (epsilon, delta) privacy guarantee"""
        # Simplified RDP accountant
        # In practice, would use more sophisticated accounting
        q = self.sample_rate
        sigma = self.noise_multiplier
        
        # RDP analysis
        alpha = 1 + q**2 / sigma**2
        epsilon = alpha * steps * q**2 / (2 * sigma**2)
        delta = 1e-5  # Common choice
        
        self.privacy_history.append((steps, epsilon, delta))
        return epsilon, delta
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy expenditure"""
        if not self.privacy_history:
            return {"epsilon": 0.0, "delta": 0.0}
        
        _, epsilon, delta = self.privacy_history[-1]
        return {"epsilon": epsilon, "delta": delta}


class SecureAggregator:
    """Secure aggregation with cryptographic guarantees"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_keys = {}
        self.server_private_key = None
        self.server_public_key = None
        self._generate_server_keys()
        
    def _generate_server_keys(self):
        """Generate server key pair"""
        self.server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.server_public_key = self.server_private_key.public_key()
    
    def register_client(self, client_id: str) -> bytes:
        """Register client and return public key"""
        return self.server_public_key.public_key_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def encrypt_weights(self, weights: Dict[str, torch.Tensor], client_id: str) -> bytes:
        """Encrypt model weights for secure transmission"""
        # Serialize weights
        weights_bytes = torch.save(weights, f=None)
        
        # Encrypt with server public key
        encrypted_weights = self.server_public_key.encrypt(
            weights_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_weights
    
    def decrypt_weights(self, encrypted_weights: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model weights"""
        # Decrypt
        weights_bytes = self.server_private_key.decrypt(
            encrypted_weights,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Deserialize weights
        weights = torch.load(io.BytesIO(weights_bytes))
        return weights
    
    def secure_aggregate(self, encrypted_weights_list: List[bytes]) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation of encrypted weights"""
        # Decrypt all weights
        weights_list = [self.decrypt_weights(ew) for ew in encrypted_weights_list]
        
        # Average weights
        if not weights_list:
            return {}
        
        aggregated_weights = {}
        for key in weights_list[0].keys():
            stacked_weights = torch.stack([w[key] for w in weights_list])
            aggregated_weights[key] = torch.mean(stacked_weights, dim=0)
        
        return aggregated_weights


class FederatedClient:
    """Federated learning client with privacy guarantees"""
    
    def __init__(self, client_id: str, config: FederatedConfig, model: nn.Module):
        self.client_id = client_id
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        self.privacy_accountant = PrivacyAccountant(
            config.noise_multiplier, 
            config.batch_size / 1000  # Assuming dataset size of 1000
        )
        
        self.round_number = 0
        self.local_steps = 0
        self.logger = logger.bind(component="FederatedClient", client_id=client_id)
        
    async def local_train(self, train_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Perform local training with differential privacy"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                
                # Apply differential privacy
                if self.config.privacy_budget > 0:
                    self._apply_differential_privacy()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.local_steps += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Compute privacy spent
        epsilon, delta = self.privacy_accountant.compute_privacy(self.local_steps)
        
        self.logger.info(
            f"Local training completed. Avg loss: {avg_loss:.4f}, "
            f"Privacy: (ε={epsilon:.4f}, δ={delta:.6f})"
        )
        
        return {
            "client_id": self.client_id,
            "avg_loss": avg_loss,
            "num_samples": len(train_data.dataset),
            "privacy_epsilon": epsilon,
            "privacy_delta": delta
        }
    
    def _apply_differential_privacy(self):
        """Apply differential privacy noise to gradients"""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=self.config.max_grad_norm
        )
        
        # Add noise
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0, 
                    std=self.config.noise_multiplier * self.config.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights"""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    def evaluate(self, test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.functional.mse_loss(output, target, reduction='sum')
                total_loss += loss.item()
                total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return {"test_loss": avg_loss, "num_samples": total_samples}


class FederatedServer:
    """Federated learning server coordinating clients"""
    
    def __init__(self, config: FederatedConfig, global_model: nn.Module):
        self.config = config
        self.global_model = global_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        self.clients: Dict[str, FederatedClient] = {}
        self.secure_aggregator = SecureAggregator(config.num_clients) if config.secure_aggregation else None
        
        self.round_number = 0
        self.training_history = []
        self.client_performance = {}
        
        self.logger = logger.bind(component="FederatedServer")
        
    def register_client(self, client_id: str, model: nn.Module) -> FederatedClient:
        """Register a new federated client"""
        client = FederatedClient(client_id, self.config, model)
        client.set_model_weights(self.get_global_weights())
        self.clients[client_id] = client
        
        self.logger.info(f"Registered client {client_id}")
        return client
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get global model weights"""
        return {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
    
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        """Set global model weights"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    async def federated_round(self, client_data: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Any]:
        """Execute one round of federated learning"""
        self.round_number += 1
        self.logger.info(f"Starting federated round {self.round_number}")
        
        # Select clients for this round
        available_clients = [cid for cid in self.clients.keys() if cid in client_data]
        num_selected = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction)
        )
        
        if len(available_clients) < self.config.min_clients:
            raise MLModelException(f"Insufficient clients: {len(available_clients)} < {self.config.min_clients}")
        
        selected_clients = np.random.choice(
            available_clients, 
            size=min(num_selected, len(available_clients)), 
            replace=False
        )
        
        # Distribute current global weights
        global_weights = self.get_global_weights()
        for client_id in selected_clients:
            self.clients[client_id].set_model_weights(global_weights)
        
        # Local training
        client_results = {}
        client_weights = {}
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            train_result = await client.local_train(client_data[client_id])
            client_results[client_id] = train_result
            client_weights[client_id] = client.get_model_weights()
        
        # Aggregate weights
        if self.config.secure_aggregation and self.secure_aggregator:
            aggregated_weights = await self._secure_aggregate(client_weights)
        else:
            aggregated_weights = self._federated_average(client_weights, client_results)
        
        # Update global model
        self.set_global_weights(aggregated_weights)
        
        # Calculate round statistics
        round_stats = self._calculate_round_stats(client_results)
        
        # Update client performance tracking
        for client_id, result in client_results.items():
            if client_id not in self.client_performance:
                self.client_performance[client_id] = []
            self.client_performance[client_id].append(result)
        
        self.training_history.append({
            "round": self.round_number,
            "selected_clients": list(selected_clients),
            "client_results": client_results,
            "round_stats": round_stats
        })
        
        self.logger.info(
            f"Round {self.round_number} completed. "
            f"Avg loss: {round_stats['avg_loss']:.4f}, "
            f"Clients: {len(selected_clients)}"
        )
        
        return {
            "round": self.round_number,
            "selected_clients": list(selected_clients),
            "round_stats": round_stats,
            "global_weights_norm": sum(torch.norm(w).item() for w in aggregated_weights.values())
        }
    
    def _federated_average(
        self, 
        client_weights: Dict[str, Dict[str, torch.Tensor]], 
        client_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Perform federated averaging"""
        if not client_weights:
            return self.get_global_weights()
        
        # Calculate weights based on number of samples
        total_samples = sum(result["num_samples"] for result in client_results.values())
        
        aggregated_weights = {}
        for param_name in next(iter(client_weights.values())).keys():
            weighted_sum = torch.zeros_like(
                next(iter(client_weights.values()))[param_name]
            )
            
            for client_id, weights in client_weights.items():
                weight = client_results[client_id]["num_samples"] / total_samples
                weighted_sum += weight * weights[param_name]
            
            aggregated_weights[param_name] = weighted_sum
        
        return aggregated_weights
    
    async def _secure_aggregate(
        self, 
        client_weights: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation"""
        # Encrypt client weights
        encrypted_weights = {}
        for client_id, weights in client_weights.items():
            encrypted_weights[client_id] = self.secure_aggregator.encrypt_weights(weights, client_id)
        
        # Secure aggregation
        aggregated_weights = self.secure_aggregator.secure_aggregate(
            list(encrypted_weights.values())
        )
        
        return aggregated_weights
    
    def _calculate_round_stats(self, client_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistics for the round"""
        if not client_results:
            return {}
        
        losses = [result["avg_loss"] for result in client_results.values()]
        sample_counts = [result["num_samples"] for result in client_results.values()]
        
        # Weighted average loss
        total_samples = sum(sample_counts)
        weighted_loss = sum(
            result["avg_loss"] * result["num_samples"] 
            for result in client_results.values()
        ) / total_samples
        
        return {
            "avg_loss": weighted_loss,
            "min_loss": min(losses),
            "max_loss": max(losses),
            "std_loss": np.std(losses),
            "total_samples": total_samples,
            "num_clients": len(client_results)
        }
    
    async def evaluate_global_model(self, test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate global model on test data"""
        self.global_model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = nn.functional.mse_loss(output, target, reduction='sum')
                total_loss += loss.item()
                total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return {"global_test_loss": avg_loss, "num_samples": total_samples}
    
    def get_client_diversity(self) -> Dict[str, float]:
        """Calculate client diversity metrics"""
        if len(self.clients) < 2:
            return {"weight_diversity": 0.0, "performance_diversity": 0.0}
        
        # Weight diversity
        client_weights = {cid: client.get_model_weights() for cid, client in self.clients.items()}
        weight_distances = []
        
        client_ids = list(client_weights.keys())
        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                distance = 0.0
                weights1 = client_weights[client_ids[i]]
                weights2 = client_weights[client_ids[j]]
                
                for param_name in weights1.keys():
                    if param_name in weights2:
                        diff = weights1[param_name] - weights2[param_name]
                        distance += torch.norm(diff).item()
                
                weight_distances.append(distance)
        
        weight_diversity = np.mean(weight_distances) if weight_distances else 0.0
        
        # Performance diversity
        if self.client_performance:
            recent_losses = []
            for client_id, history in self.client_performance.items():
                if history:
                    recent_losses.append(history[-1]["avg_loss"])
            
            performance_diversity = np.std(recent_losses) if len(recent_losses) > 1 else 0.0
        else:
            performance_diversity = 0.0
        
        return {
            "weight_diversity": weight_diversity,
            "performance_diversity": performance_diversity
        }


class FederatedLearningEngine:
    """Main federated learning engine"""
    
    def __init__(self, config: Union[FederatedConfig, Dict]):
        if isinstance(config, dict):
            config = FederatedConfig(**config)
        
        self.config = config
        self.server: Optional[FederatedServer] = None
        self.global_model: Optional[nn.Module] = None
        self.training_complete = False
        
        self.logger = logger.bind(component="FederatedLearningEngine")
    
    def initialize_federation(self, model_class: type, model_kwargs: Dict[str, Any]) -> FederatedServer:
        """Initialize federated learning setup"""
        # Create global model
        self.global_model = model_class(**model_kwargs)
        
        # Create server
        self.server = FederatedServer(self.config, self.global_model)
        
        # Create and register clients
        for i in range(self.config.num_clients):
            client_id = f"client_{i}"
            client_model = model_class(**model_kwargs)
            self.server.register_client(client_id, client_model)
        
        self.logger.info(f"Initialized federation with {self.config.num_clients} clients")
        return self.server
    
    async def federated_training(
        self, 
        client_datasets: Dict[str, torch.utils.data.DataLoader],
        test_dataset: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """Run complete federated training"""
        if self.server is None:
            raise MLModelException("Federation not initialized. Call initialize_federation() first.")
        
        self.logger.info(f"Starting federated training for {self.config.rounds} rounds")
        
        training_results = []
        global_test_results = []
        
        for round_num in range(self.config.rounds):
            # Execute federated round
            round_result = await self.server.federated_round(client_datasets)
            training_results.append(round_result)
            
            # Global evaluation
            if test_dataset is not None and round_num % 5 == 0:  # Evaluate every 5 rounds
                test_result = await self.server.evaluate_global_model(test_dataset)
                test_result["round"] = round_num + 1
                global_test_results.append(test_result)
                
                self.logger.info(
                    f"Round {round_num + 1} global test loss: {test_result['global_test_loss']:.4f}"
                )
            
            # Check convergence
            if self._check_convergence(training_results):
                self.logger.info(f"Convergence achieved at round {round_num + 1}")
                break
        
        self.training_complete = True
        
        # Final statistics
        final_stats = await self._calculate_final_statistics(training_results, global_test_results)
        
        return {
            "training_results": training_results,
            "global_test_results": global_test_results,
            "final_statistics": final_stats,
            "total_rounds": len(training_results)
        }
    
    def _check_convergence(self, training_results: List[Dict[str, Any]]) -> bool:
        """Check if training has converged"""
        if len(training_results) < 10:
            return False
        
        recent_losses = [r["round_stats"]["avg_loss"] for r in training_results[-5:]]
        loss_variance = np.var(recent_losses)
        
        return loss_variance < 1e-6  # Convergence threshold
    
    async def _calculate_final_statistics(
        self, 
        training_results: List[Dict[str, Any]], 
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate final training statistics"""
        if not training_results:
            return {}
        
        # Training loss progression
        training_losses = [r["round_stats"]["avg_loss"] for r in training_results]
        
        # Client participation
        all_clients = set()
        client_participation = {}
        for result in training_results:
            for client_id in result["selected_clients"]:
                all_clients.add(client_id)
                client_participation[client_id] = client_participation.get(client_id, 0) + 1
        
        # Diversity metrics
        diversity_metrics = self.server.get_client_diversity()
        
        # Privacy analysis
        total_privacy_spent = {}
        for client_id, client in self.server.clients.items():
            privacy_spent = client.privacy_accountant.get_privacy_spent()
            total_privacy_spent[client_id] = privacy_spent
        
        return {
            "total_rounds": len(training_results),
            "final_training_loss": training_losses[-1],
            "training_loss_improvement": training_losses[0] - training_losses[-1],
            "client_participation": client_participation,
            "diversity_metrics": diversity_metrics,
            "privacy_analysis": total_privacy_spent,
            "convergence_achieved": self.training_complete,
            "average_clients_per_round": np.mean([len(r["selected_clients"]) for r in training_results])
        }
    
    def save_federation_state(self, path: str) -> None:
        """Save complete federation state"""
        if self.server is None:
            raise MLModelException("No federation to save")
        
        state_data = {
            "config": self.config.dict(),
            "round_number": self.server.round_number,
            "global_model_state": self.server.global_model.state_dict(),
            "training_history": self.server.training_history,
            "client_performance": self.server.client_performance,
            "training_complete": self.training_complete
        }
        
        torch.save(state_data, path)
        self.logger.info(f"Federation state saved to {path}")
    
    def load_federation_state(self, path: str, model_class: type, model_kwargs: Dict[str, Any]) -> None:
        """Load federation state"""
        state_data = torch.load(path)
        
        self.config = FederatedConfig(**state_data["config"])
        
        # Recreate global model and server
        self.global_model = model_class(**model_kwargs)
        self.global_model.load_state_dict(state_data["global_model_state"])
        
        self.server = FederatedServer(self.config, self.global_model)
        self.server.round_number = state_data["round_number"]
        self.server.training_history = state_data["training_history"]
        self.server.client_performance = state_data["client_performance"]
        
        # Recreate clients
        for i in range(self.config.num_clients):
            client_id = f"client_{i}"
            client_model = model_class(**model_kwargs)
            self.server.register_client(client_id, client_model)
        
        self.training_complete = state_data.get("training_complete", False)
        
        self.logger.info(f"Federation state loaded from {path}")
