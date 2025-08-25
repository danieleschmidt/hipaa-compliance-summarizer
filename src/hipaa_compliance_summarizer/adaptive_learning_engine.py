"""
Adaptive Learning Engine - Machine Learning for Quality Optimization

This module implements an adaptive learning system that continuously learns
from quality gate execution patterns, predicts potential issues, and automatically
optimizes quality processes based on historical data and real-time feedback.
"""

import json
import logging
import math
import pickle
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    print("NumPy not available, using Python math fallback")
    import math
    import statistics
    
    # Create numpy-like functions using built-in modules
    class NumpyFallback:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0
        
        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0
        
        @staticmethod
        def polyfit(x, y, deg):
            # Simple linear regression for degree 1
            if deg == 1 and len(x) == len(y) and len(x) > 1:
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                return [slope, 0]  # Return [slope, intercept]
            return [0, 0]
        
        @staticmethod
        def percentile(values, q):
            if not values:
                return 0
            sorted_vals = sorted(values)
            index = (len(sorted_vals) - 1) * q / 100
            lower = int(index)
            upper = min(lower + 1, len(sorted_vals) - 1)
            weight = index - lower
            return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def arange(n):
            return list(range(n))
    
    np = NumpyFallback()


@dataclass
class QualityDataPoint:
    """A single quality measurement data point."""
    timestamp: datetime
    gate_type: str
    score: float
    duration: float
    status: str
    context: Dict[str, Any] = field(default_factory=dict)
    remediation_applied: bool = False
    

@dataclass
class LearningFeatures:
    """Features extracted for machine learning."""
    time_of_day: float
    day_of_week: int
    project_size_category: str
    recent_failure_rate: float
    avg_duration_trend: float
    code_complexity_score: float
    change_velocity: float
    historical_success_rate: float


@dataclass
class PredictionResult:
    """Result of quality prediction."""
    predicted_score: float
    confidence: float
    risk_level: str
    recommendations: List[str]
    optimal_parameters: Dict[str, Any]


class QualityPredictor:
    """
    Machine learning predictor for quality gate outcomes.
    
    Uses historical data to predict quality gate scores, identify potential
    failures before they occur, and recommend optimization strategies.
    """
    
    def __init__(self, history_limit: int = 10000):
        self.history_limit = history_limit
        self.data_points: deque = deque(maxlen=history_limit)
        self.feature_weights: Dict[str, float] = {}
        self.learned_patterns: Dict[str, Any] = {}
        self.prediction_accuracy: Dict[str, float] = defaultdict(float)
        self.logger = logging.getLogger(f"{__name__}.QualityPredictor")
        
    def add_data_point(self, data_point: QualityDataPoint):
        """Add a new quality measurement to the learning dataset."""
        self.data_points.append(data_point)
        self._update_learned_patterns(data_point)
        
    def predict_quality_outcome(
        self, 
        gate_type: str, 
        context: Dict[str, Any]
    ) -> PredictionResult:
        """
        Predict the outcome of a quality gate execution.
        
        Args:
            gate_type: Type of quality gate
            context: Current context and parameters
            
        Returns:
            Prediction result with score, confidence, and recommendations
        """
        if len(self.data_points) < 10:
            # Insufficient data for prediction
            return PredictionResult(
                predicted_score=0.8,
                confidence=0.1,
                risk_level="unknown",
                recommendations=["Insufficient historical data for accurate prediction"],
                optimal_parameters={}
            )
        
        # Extract features from current context
        features = self._extract_features(gate_type, context)
        
        # Apply learned prediction model
        predicted_score = self._predict_score(gate_type, features)
        confidence = self._calculate_prediction_confidence(gate_type, features)
        risk_level = self._assess_risk_level(predicted_score, confidence)
        recommendations = self._generate_recommendations(gate_type, features, predicted_score)
        optimal_parameters = self._suggest_optimal_parameters(gate_type, features)
        
        return PredictionResult(
            predicted_score=predicted_score,
            confidence=confidence,
            risk_level=risk_level,
            recommendations=recommendations,
            optimal_parameters=optimal_parameters
        )
    
    def _extract_features(self, gate_type: str, context: Dict[str, Any]) -> LearningFeatures:
        """Extract machine learning features from context."""
        now = datetime.now()
        
        # Time-based features
        time_of_day = now.hour + now.minute / 60.0
        day_of_week = now.weekday()
        
        # Project characteristics
        project_size = context.get("project_size", "medium")
        code_complexity = context.get("code_complexity", 0.5)
        
        # Historical patterns
        recent_data = [dp for dp in list(self.data_points)[-100:] 
                      if dp.gate_type == gate_type]
        
        if recent_data:
            recent_failure_rate = sum(1 for dp in recent_data if dp.score < 0.7) / len(recent_data)
            avg_duration_trend = statistics.mean([dp.duration for dp in recent_data])
            historical_success_rate = sum(1 for dp in recent_data if dp.score >= 0.8) / len(recent_data)
        else:
            recent_failure_rate = 0.1
            avg_duration_trend = 30.0
            historical_success_rate = 0.85
        
        # Change velocity (simplified)
        change_velocity = context.get("change_velocity", 0.5)
        
        return LearningFeatures(
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            project_size_category=project_size,
            recent_failure_rate=recent_failure_rate,
            avg_duration_trend=avg_duration_trend,
            code_complexity_score=code_complexity,
            change_velocity=change_velocity,
            historical_success_rate=historical_success_rate
        )
    
    def _predict_score(self, gate_type: str, features: LearningFeatures) -> float:
        """Predict quality score using learned patterns."""
        # Simple weighted prediction model
        base_score = self.learned_patterns.get(f"{gate_type}_base_score", 0.8)
        
        # Apply feature-based adjustments
        score_adjustments = [
            # Time-based patterns
            -0.05 if 22 <= features.time_of_day or features.time_of_day <= 6 else 0,  # Late night/early morning
            -0.03 if features.day_of_week in [0, 6] else 0,  # Monday/Sunday effect
            
            # Historical performance
            -0.1 * features.recent_failure_rate,  # Recent failures predict more failures
            0.05 * features.historical_success_rate,  # Historical success predicts success
            
            # Code characteristics
            -0.02 * features.code_complexity_score,  # Complex code more likely to fail
            -0.03 * features.change_velocity,  # High change velocity increases risk
        ]
        
        predicted_score = base_score + sum(score_adjustments)
        
        # Apply learned gate-specific patterns
        gate_patterns = self.learned_patterns.get(f"{gate_type}_patterns", {})
        if gate_patterns:
            pattern_adjustment = self._apply_gate_patterns(features, gate_patterns)
            predicted_score += pattern_adjustment
        
        return max(0.0, min(1.0, predicted_score))
    
    def _apply_gate_patterns(self, features: LearningFeatures, patterns: Dict[str, Any]) -> float:
        """Apply learned patterns specific to a gate type."""
        adjustment = 0.0
        
        # Apply duration-based patterns
        if "duration_patterns" in patterns:
            duration_pattern = patterns["duration_patterns"]
            if features.avg_duration_trend > duration_pattern.get("slow_threshold", 60):
                adjustment -= 0.05  # Slow operations tend to have lower quality
        
        # Apply complexity-based patterns
        if "complexity_patterns" in patterns:
            complexity_pattern = patterns["complexity_patterns"]
            if features.code_complexity_score > complexity_pattern.get("high_complexity_threshold", 0.8):
                adjustment -= 0.08  # High complexity correlates with quality issues
        
        return adjustment
    
    def _calculate_prediction_confidence(self, gate_type: str, features: LearningFeatures) -> float:
        """Calculate confidence in prediction based on data quality and patterns."""
        base_confidence = 0.5
        
        # More historical data increases confidence
        gate_data_count = sum(1 for dp in self.data_points if dp.gate_type == gate_type)
        data_confidence = min(0.4, gate_data_count / 100)
        
        # Consistent patterns increase confidence
        pattern_consistency = self.learned_patterns.get(f"{gate_type}_consistency", 0.5)
        pattern_confidence = 0.3 * pattern_consistency
        
        # Recent prediction accuracy
        accuracy_confidence = 0.2 * self.prediction_accuracy.get(gate_type, 0.5)
        
        total_confidence = base_confidence + data_confidence + pattern_confidence + accuracy_confidence
        return min(1.0, total_confidence)
    
    def _assess_risk_level(self, predicted_score: float, confidence: float) -> str:
        """Assess risk level based on predicted score and confidence."""
        if predicted_score < 0.6:
            if confidence > 0.7:
                return "high"
            else:
                return "medium"
        elif predicted_score < 0.8:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(
        self, 
        gate_type: str, 
        features: LearningFeatures, 
        predicted_score: float
    ) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if predicted_score < 0.7:
            recommendations.append(f"High failure risk for {gate_type} - consider postponing or adding pre-checks")
        
        if features.recent_failure_rate > 0.3:
            recommendations.append("Recent failure pattern detected - review recent changes")
        
        if features.code_complexity_score > 0.8:
            recommendations.append("High code complexity - consider refactoring before quality gates")
        
        if features.change_velocity > 0.8:
            recommendations.append("High change velocity - implement additional testing")
        
        # Time-based recommendations
        if 22 <= features.time_of_day or features.time_of_day <= 6:
            recommendations.append("Running during off-hours - consider scheduling during peak hours")
        
        # Gate-specific recommendations
        if gate_type == "testing" and predicted_score < 0.8:
            recommendations.append("Consider running subset of tests first to identify issues quickly")
        elif gate_type == "security" and predicted_score < 0.9:
            recommendations.append("Pre-run security scans on changed files only")
        
        return recommendations
    
    def _suggest_optimal_parameters(self, gate_type: str, features: LearningFeatures) -> Dict[str, Any]:
        """Suggest optimal parameters for quality gate execution."""
        optimal_params = {}
        
        # Timeout adjustments based on historical patterns
        if features.avg_duration_trend > 60:
            optimal_params["timeout"] = int(features.avg_duration_trend * 1.5)
        
        # Retry parameters based on failure patterns
        if features.recent_failure_rate > 0.2:
            optimal_params["max_retries"] = 2
            optimal_params["retry_delay"] = 5.0
        
        # Threshold adjustments based on code characteristics
        if features.code_complexity_score > 0.7:
            optimal_params["quality_threshold"] = 0.75  # Lower threshold for complex code
        
        return optimal_params
    
    def _update_learned_patterns(self, data_point: QualityDataPoint):
        """Update learned patterns with new data point."""
        gate_type = data_point.gate_type
        
        # Update base score
        current_base = self.learned_patterns.get(f"{gate_type}_base_score", 0.8)
        learning_rate = 0.01
        new_base = current_base + learning_rate * (data_point.score - current_base)
        self.learned_patterns[f"{gate_type}_base_score"] = new_base
        
        # Update consistency metrics
        gate_scores = [dp.score for dp in list(self.data_points) if dp.gate_type == gate_type]
        if len(gate_scores) > 5:
            consistency = 1.0 - (statistics.stdev(gate_scores) / max(statistics.mean(gate_scores), 0.1))
            self.learned_patterns[f"{gate_type}_consistency"] = max(0.0, consistency)
        
        # Update duration patterns
        gate_durations = [dp.duration for dp in list(self.data_points) if dp.gate_type == gate_type]
        if len(gate_durations) > 5:
            avg_duration = statistics.mean(gate_durations)
            slow_threshold = avg_duration + statistics.stdev(gate_durations)
            
            patterns = self.learned_patterns.get(f"{gate_type}_patterns", {})
            patterns["duration_patterns"] = {
                "avg_duration": avg_duration,
                "slow_threshold": slow_threshold
            }
            self.learned_patterns[f"{gate_type}_patterns"] = patterns
    
    def update_prediction_accuracy(self, gate_type: str, predicted_score: float, actual_score: float):
        """Update prediction accuracy metrics."""
        error = abs(predicted_score - actual_score)
        accuracy = 1.0 - min(error, 1.0)
        
        current_accuracy = self.prediction_accuracy[gate_type]
        learning_rate = 0.1
        self.prediction_accuracy[gate_type] = current_accuracy + learning_rate * (accuracy - current_accuracy)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns and accuracy."""
        return {
            "total_data_points": len(self.data_points),
            "learned_patterns": self.learned_patterns,
            "prediction_accuracy": dict(self.prediction_accuracy),
            "gate_type_distribution": self._get_gate_type_distribution(),
        }
    
    def _get_gate_type_distribution(self) -> Dict[str, int]:
        """Get distribution of gate types in historical data."""
        distribution = defaultdict(int)
        for dp in self.data_points:
            distribution[dp.gate_type] += 1
        return dict(distribution)
    
    def save_model(self, file_path: Path):
        """Save learned model to file."""
        model_data = {
            "learned_patterns": self.learned_patterns,
            "prediction_accuracy": dict(self.prediction_accuracy),
            "feature_weights": self.feature_weights,
            "data_points": list(self.data_points)[-1000:],  # Save recent data points
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self, file_path: Path):
        """Load learned model from file."""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.learned_patterns = model_data.get("learned_patterns", {})
            self.prediction_accuracy = defaultdict(float, model_data.get("prediction_accuracy", {}))
            self.feature_weights = model_data.get("feature_weights", {})
            
            # Restore data points
            saved_data_points = model_data.get("data_points", [])
            for dp_data in saved_data_points:
                if isinstance(dp_data, QualityDataPoint):
                    self.data_points.append(dp_data)
            
            self.logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")


class AdaptiveOptimizer:
    """
    Adaptive optimizer that adjusts quality gate parameters based on learning.
    
    Uses machine learning insights to automatically optimize quality gate
    configurations, thresholds, and execution strategies.
    """
    
    def __init__(self, predictor: QualityPredictor):
        self.predictor = predictor
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_optimizations: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.AdaptiveOptimizer")
    
    def optimize_gate_parameters(
        self, 
        gate_type: str, 
        current_config: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize quality gate parameters based on learned patterns.
        
        Args:
            gate_type: Type of quality gate
            current_config: Current configuration
            context: Execution context
            
        Returns:
            Optimized configuration
        """
        # Get prediction for current context
        prediction = self.predictor.predict_quality_outcome(gate_type, context)
        
        # Start with current config
        optimized_config = current_config.copy()
        
        # Apply learned optimizations
        if prediction.optimal_parameters:
            for param, value in prediction.optimal_parameters.items():
                if param in optimized_config:
                    old_value = optimized_config[param]
                    optimized_config[param] = value
                    self.logger.info(f"Optimized {gate_type}.{param}: {old_value} -> {value}")
        
        # Apply risk-based optimizations
        if prediction.risk_level == "high":
            optimized_config = self._apply_high_risk_optimizations(gate_type, optimized_config)
        elif prediction.risk_level == "low":
            optimized_config = self._apply_low_risk_optimizations(gate_type, optimized_config)
        
        # Store optimization
        self._record_optimization(gate_type, current_config, optimized_config, prediction)
        
        return optimized_config
    
    def _apply_high_risk_optimizations(self, gate_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations for high-risk scenarios."""
        optimized = config.copy()
        
        # Increase timeouts for high-risk scenarios
        if "timeout" in optimized:
            optimized["timeout"] = int(optimized["timeout"] * 1.5)
        
        # Increase retry attempts
        if "max_retries" in optimized:
            optimized["max_retries"] = min(optimized.get("max_retries", 1) + 1, 5)
        
        # Lower quality thresholds temporarily to reduce false negatives
        if "threshold" in optimized and gate_type != "security":  # Never lower security thresholds
            optimized["threshold"] = max(optimized["threshold"] * 0.95, 0.6)
        
        # Enable additional logging
        optimized["verbose_logging"] = True
        
        return optimized
    
    def _apply_low_risk_optimizations(self, gate_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations for low-risk scenarios."""
        optimized = config.copy()
        
        # Reduce timeouts for faster execution
        if "timeout" in optimized:
            optimized["timeout"] = max(int(optimized["timeout"] * 0.8), 30)
        
        # Reduce retry attempts
        if "max_retries" in optimized:
            optimized["max_retries"] = max(optimized.get("max_retries", 3) - 1, 1)
        
        # Slightly increase quality thresholds
        if "threshold" in optimized:
            optimized["threshold"] = min(optimized["threshold"] * 1.02, 1.0)
        
        # Reduce verbosity
        optimized["verbose_logging"] = False
        
        return optimized
    
    def _record_optimization(
        self, 
        gate_type: str, 
        original_config: Dict[str, Any], 
        optimized_config: Dict[str, Any], 
        prediction: PredictionResult
    ):
        """Record optimization for learning and analysis."""
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "gate_type": gate_type,
            "original_config": original_config,
            "optimized_config": optimized_config,
            "prediction": {
                "predicted_score": prediction.predicted_score,
                "confidence": prediction.confidence,
                "risk_level": prediction.risk_level,
            },
            "changes": self._calculate_config_changes(original_config, optimized_config),
        }
        
        self.optimization_history.append(optimization_record)
        self.current_optimizations[gate_type] = optimized_config
    
    def _calculate_config_changes(
        self, 
        original: Dict[str, Any], 
        optimized: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate changes between original and optimized configurations."""
        changes = []
        
        for key, new_value in optimized.items():
            original_value = original.get(key)
            if original_value != new_value:
                changes.append({
                    "parameter": key,
                    "original_value": original_value,
                    "new_value": new_value,
                    "change_type": "modified" if key in original else "added",
                })
        
        for key in original:
            if key not in optimized:
                changes.append({
                    "parameter": key,
                    "original_value": original[key],
                    "new_value": None,
                    "change_type": "removed",
                })
        
        return changes
    
    def evaluate_optimization_effectiveness(
        self, 
        gate_type: str, 
        actual_result: Dict[str, Any]
    ):
        """Evaluate the effectiveness of applied optimizations."""
        if gate_type not in self.current_optimizations:
            return
        
        # Find the most recent optimization for this gate type
        recent_optimizations = [
            opt for opt in self.optimization_history[-10:]  # Last 10 optimizations
            if opt["gate_type"] == gate_type
        ]
        
        if not recent_optimizations:
            return
        
        latest_opt = recent_optimizations[-1]
        predicted_score = latest_opt["prediction"]["predicted_score"]
        actual_score = actual_result.get("score", 0.0)
        
        # Update predictor accuracy
        self.predictor.update_prediction_accuracy(gate_type, predicted_score, actual_score)
        
        # Evaluate optimization success
        effectiveness = self._calculate_optimization_effectiveness(latest_opt, actual_result)
        
        # Update optimization record
        latest_opt["actual_result"] = actual_result
        latest_opt["effectiveness"] = effectiveness
        
        self.logger.info(f"Optimization effectiveness for {gate_type}: {effectiveness:.2f}")
    
    def _calculate_optimization_effectiveness(
        self, 
        optimization: Dict[str, Any], 
        actual_result: Dict[str, Any]
    ) -> float:
        """Calculate the effectiveness of an optimization."""
        predicted_score = optimization["prediction"]["predicted_score"]
        actual_score = actual_result.get("score", 0.0)
        
        # Prediction accuracy component (50% weight)
        prediction_error = abs(predicted_score - actual_score)
        prediction_accuracy = 1.0 - min(prediction_error, 1.0)
        
        # Performance improvement component (30% weight)
        # Compare with historical baseline
        baseline_score = 0.8  # Simplified baseline
        performance_improvement = max(0, actual_score - baseline_score)
        
        # Risk mitigation component (20% weight)
        risk_level = optimization["prediction"]["risk_level"]
        risk_mitigation_score = {"high": 0.8, "medium": 0.9, "low": 1.0}.get(risk_level, 0.8)
        if actual_score >= 0.8:  # Successful execution
            risk_mitigation_score = 1.0
        
        # Weighted effectiveness score
        effectiveness = (
            0.5 * prediction_accuracy +
            0.3 * performance_improvement +
            0.2 * risk_mitigation_score
        )
        
        return max(0.0, min(1.0, effectiveness))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history and effectiveness."""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        total_optimizations = len(self.optimization_history)
        effectiveness_scores = [
            opt.get("effectiveness", 0.0) for opt in self.optimization_history
            if "effectiveness" in opt
        ]
        
        avg_effectiveness = statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
        
        # Gate type distribution
        gate_type_counts = defaultdict(int)
        for opt in self.optimization_history:
            gate_type_counts[opt["gate_type"]] += 1
        
        return {
            "total_optimizations": total_optimizations,
            "average_effectiveness": avg_effectiveness,
            "gate_type_distribution": dict(gate_type_counts),
            "recent_optimizations": self.optimization_history[-5:],  # Last 5
            "current_active_optimizations": list(self.current_optimizations.keys()),
        }


class AdaptiveLearningEngine:
    """
    Main adaptive learning engine that coordinates prediction and optimization.
    
    Provides a unified interface for machine learning-driven quality optimization,
    combining prediction, optimization, and continuous learning capabilities.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("/root/repo/adaptive_quality_model.pkl")
        self.predictor = QualityPredictor()
        self.optimizer = AdaptiveOptimizer(self.predictor)
        self.logger = logging.getLogger(f"{__name__}.AdaptiveLearningEngine")
        
        # Load existing model if available
        if self.model_path.exists():
            self.predictor.load_model(self.model_path)
    
    def learn_from_execution(
        self, 
        gate_type: str, 
        execution_result: Dict[str, Any], 
        context: Dict[str, Any]
    ):
        """Learn from a quality gate execution result."""
        data_point = QualityDataPoint(
            timestamp=datetime.now(),
            gate_type=gate_type,
            score=execution_result.get("score", 0.0),
            duration=execution_result.get("duration", 0.0),
            status=execution_result.get("status", "unknown"),
            context=context,
            remediation_applied=execution_result.get("remediation_applied", False)
        )
        
        self.predictor.add_data_point(data_point)
        self.optimizer.evaluate_optimization_effectiveness(gate_type, execution_result)
        
        # Periodically save model
        if len(self.predictor.data_points) % 100 == 0:
            self.save_model()
    
    def predict_and_optimize(
        self, 
        gate_type: str, 
        current_config: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Tuple[PredictionResult, Dict[str, Any]]:
        """
        Predict quality outcome and optimize configuration.
        
        Returns:
            Tuple of (prediction_result, optimized_config)
        """
        # Get prediction
        prediction = self.predictor.predict_quality_outcome(gate_type, context)
        
        # Optimize configuration based on prediction
        optimized_config = self.optimizer.optimize_gate_parameters(
            gate_type, current_config, context
        )
        
        return prediction, optimized_config
    
    def get_comprehensive_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights from the learning engine."""
        return {
            "learning_summary": self.predictor.get_learning_summary(),
            "optimization_summary": self.optimizer.get_optimization_summary(),
            "model_performance": {
                "data_points": len(self.predictor.data_points),
                "prediction_accuracies": dict(self.predictor.prediction_accuracy),
                "learned_patterns_count": len(self.predictor.learned_patterns),
            },
            "recommendations": self._generate_strategic_recommendations(),
        }
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on learning insights."""
        recommendations = []
        
        # Analyze prediction accuracy
        accuracies = list(self.predictor.prediction_accuracy.values())
        if accuracies and statistics.mean(accuracies) < 0.7:
            recommendations.append("Consider collecting more diverse training data to improve prediction accuracy")
        
        # Analyze data distribution
        distribution = self.predictor._get_gate_type_distribution()
        if distribution:
            min_count = min(distribution.values())
            max_count = max(distribution.values())
            if max_count > 3 * min_count:
                recommendations.append("Data distribution is imbalanced - consider collecting more data for underrepresented gate types")
        
        # Analyze optimization effectiveness
        opt_summary = self.optimizer.get_optimization_summary()
        if opt_summary.get("average_effectiveness", 0) < 0.6:
            recommendations.append("Optimization effectiveness is low - consider reviewing optimization strategies")
        
        return recommendations
    
    def save_model(self):
        """Save the learned model to disk."""
        self.predictor.save_model(self.model_path)
    
    def reset_learning(self):
        """Reset all learned patterns (use with caution)."""
        self.predictor.learned_patterns.clear()
        self.predictor.prediction_accuracy.clear()
        self.predictor.data_points.clear()
        self.optimizer.optimization_history.clear()
        self.optimizer.current_optimizations.clear()
        
        self.logger.warning("All learned patterns have been reset")


# Global adaptive learning engine instance
adaptive_learning_engine = AdaptiveLearningEngine()


def learning_enabled_quality_gate(gate_type: str):
    """
    Decorator that enables adaptive learning for a quality gate.
    
    Usage:
        @learning_enabled_quality_gate("syntax_check")
        async def check_syntax(context):
            # Your quality gate implementation
            return {"score": 0.95, "duration": 30.0, "status": "passed"}
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = kwargs.get("context", {})
            
            # Get current config (simplified)
            current_config = kwargs.get("config", {})
            
            # Predict and optimize
            prediction, optimized_config = adaptive_learning_engine.predict_and_optimize(
                gate_type, current_config, context
            )
            
            # Update config for execution
            kwargs["config"] = optimized_config
            
            # Execute the quality gate
            result = await func(*args, **kwargs)
            
            # Learn from the execution
            adaptive_learning_engine.learn_from_execution(gate_type, result, context)
            
            return result
        
        return wrapper
    return decorator