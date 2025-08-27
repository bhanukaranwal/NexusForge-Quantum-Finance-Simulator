import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field
import io
import base64

from src.core.exceptions import MLModelException
from src.core.logging import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class XAIConfig(BaseModel):
    methods: List[str] = Field(
        default=["shap", "lime", "permutation", "attention"],
        description="Explanation methods to use"
    )
    sample_size: int = Field(default=100, description="Sample size for explanations")
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    categorical_features: List[int] = Field(default_factory=list, description="Categorical feature indices")
    background_size: int = Field(default=50, description="Background dataset size for SHAP")
    lime_samples: int = Field(default=5000, description="Number of samples for LIME")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold for explanations")


class ExplanationResult(BaseModel):
    method: str
    feature_importance: Dict[str, float]
    explanation_text: str
    confidence_score: float
    visualization_data: Optional[str] = None  # Base64 encoded plot
    local_explanations: Optional[Dict[str, Any]] = None
    global_explanations: Optional[Dict[str, Any]] = None


class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) explainer"""
    
    def __init__(self, model, background_data: np.ndarray, feature_names: List[str]):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        
        if SHAP_AVAILABLE:
            try:
                if hasattr(model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
                else:
                    self.explainer = shap.KernelExplainer(model.predict, background_data)
            except:
                self.explainer = shap.KernelExplainer(model, background_data)
    
    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single instance"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP not available"}
        
        try:
            shap_values = self.explainer.shap_values(instance.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            feature_importance = dict(zip(self.feature_names, shap_values[0]))
            
            # Generate explanation text
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            explanation_text = "Top contributing features:\n"
            for feature, importance in sorted_importance[:5]:
                direction = "increases" if importance > 0 else "decreases"
                explanation_text += f"- {feature}: {direction} prediction by {abs(importance):.4f}\n"
            
            return {
                "feature_importance": feature_importance,
                "explanation_text": explanation_text,
                "shap_values": shap_values[0].tolist(),
                "base_value": self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            }
            
        except Exception as e:
            return {"error": f"SHAP explanation failed: {str(e)}"}
    
    def explain_global(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """Generate global explanations"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP not available"}
        
        try:
            sample_size = min(max_samples, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            
            shap_values = self.explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate global feature importance
            global_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance = dict(zip(self.feature_names, global_importance))
            
            return {
                "global_feature_importance": feature_importance,
                "sample_size": sample_size,
                "mean_shap_values": np.mean(shap_values, axis=0).tolist(),
                "std_shap_values": np.std(shap_values, axis=0).tolist()
            }
            
        except Exception as e:
            return {"error": f"Global SHAP explanation failed: {str(e)}"}
    
    def create_visualization(self, shap_values: np.ndarray, instance: np.ndarray) -> str:
        """Create SHAP visualization"""
        if not SHAP_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values,
                    base_values=0,
                    data=instance,
                    feature_names=self.feature_names
                )
            )
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            return f"Visualization failed: {str(e)}"


class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) explainer"""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: List[str], 
                 categorical_features: List[int] = None):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.explainer = None
        
        if LIME_AVAILABLE:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                categorical_features=categorical_features,
                verbose=False,
                mode='regression'
            )
    
    def explain_instance(self, instance: np.ndarray, num_samples: int = 5000) -> Dict[str, Any]:
        """Explain a single instance"""
        if not LIME_AVAILABLE or self.explainer is None:
            return {"error": "LIME not available"}
        
        try:
            explanation = self.explainer.explain_instance(
                instance,
                self.model.predict,
                num_samples=num_samples,
                num_features=len(self.feature_names)
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_importance[feature_name] = importance
            
            # Generate explanation text
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanation_text = "LIME explanation - Top contributing features:\n"
            for feature, importance in sorted_importance[:5]:
                direction = "increases" if importance > 0 else "decreases"
                explanation_text += f"- {feature}: {direction} prediction by {abs(importance):.4f}\n"
            
            return {
                "feature_importance": feature_importance,
                "explanation_text": explanation_text,
                "lime_score": explanation.score,
                "intercept": explanation.intercept[1] if hasattr(explanation, 'intercept') else 0
            }
            
        except Exception as e:
            return {"error": f"LIME explanation failed: {str(e)}"}


class PermutationExplainer:
    """Permutation importance explainer"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def explain_global(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            feature_importance = dict(zip(
                self.feature_names, 
                perm_importance.importances_mean
            ))
            
            feature_std = dict(zip(
                self.feature_names,
                perm_importance.importances_std
            ))
            
            # Generate explanation text
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanation_text = "Permutation importance - Most important features:\n"
            for feature, importance in sorted_importance[:5]:
                std_dev = feature_std[feature]
                explanation_text += f"- {feature}: {importance:.4f} (±{std_dev:.4f})\n"
            
            return {
                "feature_importance": feature_importance,
                "feature_importance_std": feature_std,
                "explanation_text": explanation_text
            }
            
        except Exception as e:
            return {"error": f"Permutation importance failed: {str(e)}"}


class AttentionExplainer:
    """Attention-based explainer for transformer models"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def explain_instance(self, instance: torch.Tensor) -> Dict[str, Any]:
        """Extract attention weights for explanation"""
        if not hasattr(self.model, 'get_attention_weights'):
            return {"error": "Model does not support attention extraction"}
        
        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model(instance.unsqueeze(0))
                attention_weights = self.model.get_attention_weights()
            
            if attention_weights is None:
                return {"error": "No attention weights available"}
            
            # Average attention across heads and layers if multi-dimensional
            if attention_weights.dim() > 2:
                attention_weights = attention_weights.mean(dim=list(range(attention_weights.dim() - 2)))
            
            # Get attention for the instance
            attention_scores = attention_weights[0].cpu().numpy()
            
            # Map to feature names
            feature_importance = {}
            for i, score in enumerate(attention_scores):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = float(score)
            
            # Generate explanation text
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            explanation_text = "Attention-based explanation - Most attended features:\n"
            for feature, attention in sorted_importance[:5]:
                explanation_text += f"- {feature}: attention score {attention:.4f}\n"
            
            return {
                "feature_importance": feature_importance,
                "explanation_text": explanation_text,
                "attention_matrix": attention_weights.cpu().numpy().tolist()
            }
            
        except Exception as e:
            return {"error": f"Attention explanation failed: {str(e)}"}


class ExplainableAI:
    """Main XAI class combining multiple explanation methods"""
    
    def __init__(self, config: Union[XAIConfig, Dict]):
        if isinstance(config, dict):
            config = XAIConfig(**config)
        
        self.config = config
        self.explainers = {}
        self.model = None
        self.training_data = None
        self.scaler = StandardScaler()
        self.logger = logger.bind(component="ExplainableAI")
    
    def initialize_explainers(self, model, training_data: np.ndarray, 
                            target_data: Optional[np.ndarray] = None) -> None:
        """Initialize all explainers"""
        self.model = model
        self.training_data = training_data
        
        # Scale training data
        training_data_scaled = self.scaler.fit_transform(training_data)
        
        # Sample background data for SHAP
        background_size = min(self.config.background_size, len(training_data_scaled))
        background_indices = np.random.choice(len(training_data_scaled), background_size, replace=False)
        background_data = training_data_scaled[background_indices]
        
        # Initialize explainers based on config
        if "shap" in self.config.methods and SHAP_AVAILABLE:
            self.explainers["shap"] = SHAPExplainer(
                model, background_data, self.config.feature_names
            )
        
        if "lime" in self.config.methods and LIME_AVAILABLE:
            self.explainers["lime"] = LIMEExplainer(
                model, training_data_scaled, self.config.feature_names,
                self.config.categorical_features
            )
        
        if "permutation" in self.config.methods:
            self.explainers["permutation"] = PermutationExplainer(
                model, self.config.feature_names
            )
        
        if "attention" in self.config.methods:
            self.explainers["attention"] = AttentionExplainer(
                model, self.config.feature_names
            )
        
        self.logger.info(f"Initialized {len(self.explainers)} explainers: {list(self.explainers.keys())}")
    
    async def explain_prediction(self, instance: np.ndarray, 
                               methods: Optional[List[str]] = None) -> List[ExplanationResult]:
        """Generate explanations for a single prediction"""
        if self.model is None:
            raise MLModelException("Model not initialized. Call initialize_explainers() first.")
        
        methods = methods or self.config.methods
        results = []
        
        # Scale instance
        instance_scaled = self.scaler.transform(instance.reshape(1, -1))[0]
        
        for method in methods:
            if method not in self.explainers:
                continue
            
            try:
                explainer = self.explainers[method]
                
                if method == "shap":
                    explanation = explainer.explain_instance(instance_scaled)
                elif method == "lime":
                    explanation = explainer.explain_instance(instance_scaled, self.config.lime_samples)
                elif method == "attention" and isinstance(instance, torch.Tensor):
                    explanation = explainer.explain_instance(instance)
                else:
                    continue
                
                if "error" not in explanation:
                    # Calculate confidence score
                    confidence = self._calculate_confidence(explanation, method)
                    
                    # Create visualization if applicable
                    visualization = None
                    if method == "shap" and "shap_values" in explanation:
                        visualization = explainer.create_visualization(
                            np.array(explanation["shap_values"]), instance_scaled
                        )
                    
                    result = ExplanationResult(
                        method=method,
                        feature_importance=explanation["feature_importance"],
                        explanation_text=explanation["explanation_text"],
                        confidence_score=confidence,
                        visualization_data=visualization,
                        local_explanations=explanation
                    )
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"Explanation failed for method {method}: {str(e)}")
        
        return results
    
    async def explain_model_globally(self, X: Optional[np.ndarray] = None, 
                                   y: Optional[np.ndarray] = None) -> List[ExplanationResult]:
        """Generate global model explanations"""
        if self.model is None:
            raise MLModelException("Model not initialized. Call initialize_explainers() first.")
        
        if X is None:
            X = self.training_data
        
        results = []
        X_scaled = self.scaler.transform(X)
        
        for method, explainer in self.explainers.items():
            try:
                if method == "shap":
                    explanation = explainer.explain_global(X_scaled, self.config.sample_size)
                elif method == "permutation" and y is not None:
                    explanation = explainer.explain_global(X_scaled, y)
                else:
                    continue
                
                if "error" not in explanation:
                    # Use global feature importance
                    if "global_feature_importance" in explanation:
                        feature_importance = explanation["global_feature_importance"]
                    else:
                        feature_importance = explanation["feature_importance"]
                    
                    confidence = self._calculate_confidence(explanation, method)
                    
                    result = ExplanationResult(
                        method=f"{method}_global",
                        feature_importance=feature_importance,
                        explanation_text=explanation["explanation_text"],
                        confidence_score=confidence,
                        global_explanations=explanation
                    )
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"Global explanation failed for method {method}: {str(e)}")
        
        return results
    
    def _calculate_confidence(self, explanation: Dict[str, Any], method: str) -> float:
        """Calculate confidence score for explanation"""
        try:
            if method == "lime" and "lime_score" in explanation:
                return min(1.0, max(0.0, explanation["lime_score"]))
            
            elif method == "shap" and "shap_values" in explanation:
                # Confidence based on magnitude of SHAP values
                shap_values = np.array(explanation["shap_values"])
                total_magnitude = np.sum(np.abs(shap_values))
                return min(1.0, total_magnitude / 10.0)  # Normalize
            
            elif "feature_importance" in explanation:
                # Confidence based on feature importance distribution
                importances = list(explanation["feature_importance"].values())
                if not importances:
                    return 0.0
                
                # Calculate entropy-based confidence
                abs_importances = np.abs(importances)
                total = np.sum(abs_importances)
                if total == 0:
                    return 0.0
                
                probs = abs_importances / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(importances))
                confidence = 1.0 - (entropy / max_entropy)
                
                return confidence
            
        except:
            pass
        
        return 0.5  # Default confidence
    
    async def comparative_explanation(self, instances: List[np.ndarray], 
                                    labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare explanations across multiple instances"""
        if len(instances) < 2:
            raise MLModelException("At least 2 instances required for comparison")
        
        all_explanations = []
        for i, instance in enumerate(instances):
            explanations = await self.explain_prediction(instance)
            label = labels[i] if labels and i < len(labels) else f"Instance_{i}"
            all_explanations.append((label, explanations))
        
        # Compare feature importance across instances
        comparison_results = {}
        
        for method in self.config.methods:
            method_explanations = []
            for label, explanations in all_explanations:
                method_exp = next((exp for exp in explanations if exp.method == method), None)
                if method_exp:
                    method_explanations.append((label, method_exp.feature_importance))
            
            if len(method_explanations) >= 2:
                comparison_results[method] = self._compare_feature_importance(method_explanations)
        
        return {
            "individual_explanations": all_explanations,
            "comparative_analysis": comparison_results,
            "consistency_score": self._calculate_consistency_score(all_explanations)
        }
    
    def _compare_feature_importance(self, explanations: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Compare feature importance across explanations"""
        # Get all features
        all_features = set()
        for _, importance in explanations:
            all_features.update(importance.keys())
        
        all_features = list(all_features)
        
        # Create importance matrix
        importance_matrix = []
        labels = []
        
        for label, importance in explanations:
            labels.append(label)
            row = [importance.get(feature, 0.0) for feature in all_features]
            importance_matrix.append(row)
        
        importance_matrix = np.array(importance_matrix)
        
        # Calculate statistics
        feature_means = np.mean(importance_matrix, axis=0)
        feature_stds = np.std(importance_matrix, axis=0)
        
        # Find most consistent and most variable features
        consistency_scores = 1 / (1 + feature_stds)  # Higher is more consistent
        
        most_consistent = all_features[np.argmax(consistency_scores)]
        most_variable = all_features[np.argmax(feature_stds)]
        
        return {
            "features": all_features,
            "importance_matrix": importance_matrix.tolist(),
            "instance_labels": labels,
            "feature_means": feature_means.tolist(),
            "feature_stds": feature_stds.tolist(),
            "most_consistent_feature": most_consistent,
            "most_variable_feature": most_variable,
            "consistency_scores": consistency_scores.tolist()
        }
    
    def _calculate_consistency_score(self, all_explanations: List[Tuple[str, List[ExplanationResult]]]) -> float:
        """Calculate consistency score across explanations"""
        try:
            method_consistency = []
            
            for method in self.config.methods:
                method_importances = []
                
                for label, explanations in all_explanations:
                    method_exp = next((exp for exp in explanations if exp.method == method), None)
                    if method_exp:
                        # Get top 5 features
                        sorted_features = sorted(
                            method_exp.feature_importance.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:5]
                        top_features = [f[0] for f in sorted_features]
                        method_importances.append(set(top_features))
                
                if len(method_importances) >= 2:
                    # Calculate Jaccard similarity of top features
                    similarities = []
                    for i in range(len(method_importances)):
                        for j in range(i + 1, len(method_importances)):
                            intersection = len(method_importances[i] & method_importances[j])
                            union = len(method_importances[i] | method_importances[j])
                            if union > 0:
                                similarities.append(intersection / union)
                    
                    if similarities:
                        method_consistency.append(np.mean(similarities))
            
            return np.mean(method_consistency) if method_consistency else 0.0
            
        except:
            return 0.0
    
    async def generate_explanation_report(self, predictions: List[Dict[str, Any]], 
                                        output_format: str = "html") -> str:
        """Generate comprehensive explanation report"""
        if not predictions:
            return "No predictions to explain"
        
        report_sections = []
        
        # Executive Summary
        report_sections.append("# Model Explanation Report\n")
        report_sections.append(f"Generated {len(predictions)} explanations using methods: {', '.join(self.config.methods)}\n\n")
        
        # Method Overview
        report_sections.append("## Explanation Methods\n")
        for method in self.config.methods:
            if method == "shap":
                report_sections.append("- **SHAP**: Unified framework for interpreting predictions using Shapley values from game theory\n")
            elif method == "lime":
                report_sections.append("- **LIME**: Local explanations by approximating the model locally with interpretable models\n")
            elif method == "permutation":
                report_sections.append("- **Permutation Importance**: Measures feature importance by permuting feature values\n")
            elif method == "attention":
                report_sections.append("- **Attention Weights**: Neural network attention mechanisms showing feature focus\n")
        
        report_sections.append("\n")
        
        # Individual Explanations
        for i, pred_data in enumerate(predictions):
            report_sections.append(f"## Prediction {i + 1}\n")
            
            if "prediction" in pred_data:
                report_sections.append(f"**Predicted Value**: {pred_data['prediction']}\n")
            
            if "explanations" in pred_data:
                for explanation in pred_data["explanations"]:
                    report_sections.append(f"### {explanation.method.upper()}\n")
                    report_sections.append(f"**Confidence**: {explanation.confidence_score:.3f}\n\n")
                    report_sections.append(explanation.explanation_text)
                    report_sections.append("\n")
        
        # Global Analysis
        if len(predictions) > 1:
            report_sections.append("## Global Analysis\n")
            
            # Aggregate feature importance
            all_importances = {}
            for pred_data in predictions:
                if "explanations" in pred_data:
                    for explanation in pred_data["explanations"]:
                        for feature, importance in explanation.feature_importance.items():
                            if feature not in all_importances:
                                all_importances[feature] = []
                            all_importances[feature].append(importance)
            
            # Calculate average importance
            avg_importances = {
                feature: np.mean(importances) 
                for feature, importances in all_importances.items()
            }
            
            sorted_features = sorted(
                avg_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            report_sections.append("### Most Important Features (Average)\n")
            for feature, importance in sorted_features[:10]:
                report_sections.append(f"- {feature}: {importance:.4f}\n")
            report_sections.append("\n")
        
        # Recommendations
        report_sections.append("## Recommendations\n")
        report_sections.append("- Review high-importance features for domain expertise validation\n")
        report_sections.append("- Consider feature engineering for low-importance but domain-relevant features\n")
        report_sections.append("- Monitor explanation consistency across similar predictions\n")
        report_sections.append("- Use explanations for model debugging and improvement\n")
        
        report_content = "".join(report_sections)
        
        if output_format == "html":
            import markdown
            html_content = markdown.markdown(report_content)
            return f"<html><body>{html_content}</body></html>"
        
        return report_content
    
    def create_explanation_dashboard(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Create interactive dashboard data for explanations"""
        if not explanations:
            return {}
        
        # Aggregate feature importance across methods
        feature_importance_data = {}
        method_data = {}
        
        for explanation in explanations:
            method_data[explanation.method] = {
                "confidence": explanation.confidence_score,
                "feature_count": len(explanation.feature_importance)
            }
            
            for feature, importance in explanation.feature_importance.items():
                if feature not in feature_importance_data:
                    feature_importance_data[feature] = {}
                feature_importance_data[feature][explanation.method] = importance
        
        # Create visualization data
        dashboard_data = {
            "feature_importance": feature_importance_data,
            "method_summary": method_data,
            "explanation_texts": {exp.method: exp.explanation_text for exp in explanations},
            "confidence_scores": {exp.method: exp.confidence_score for exp in explanations},
            "visualizations": {exp.method: exp.visualization_data for exp in explanations if exp.visualization_data}
        }
        
        return dashboard_data
    
    def save_explanations(self, explanations: List[ExplanationResult], path: str) -> None:
        """Save explanations to file"""
        explanation_data = []
        
        for explanation in explanations:
            data = {
                "method": explanation.method,
                "feature_importance": explanation.feature_importance,
                "explanation_text": explanation.explanation_text,
                "confidence_score": explanation.confidence_score
            }
            
            if explanation.local_explanations:
                data["local_explanations"] = explanation.local_explanations
            
            if explanation.global_explanations:
                data["global_explanations"] = explanation.global_explanations
            
            explanation_data.append(data)
        
        with open(path, 'w') as f:
            import json
            json.dump(explanation_data, f, indent=2, default=str)
        
        self.logger.info(f"Explanations saved to {path}")
