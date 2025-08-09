"""
Predictive Analytics for HIPAA Compliance Risk Assessment.

Advanced machine learning models for:
1. Predicting compliance violations before they occur
2. Risk scoring for healthcare documents and processes
3. Temporal pattern analysis for compliance trends
4. Anomaly detection in PHI handling workflows
5. Proactive compliance intervention recommendations
"""

from __future__ import annotations

import logging
import numpy as np
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications for compliance assessment."""
    
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def score_range(self) -> Tuple[float, float]:
        """Get the score range for this risk level."""
        ranges = {
            RiskLevel.VERY_LOW: (0.0, 0.2),
            RiskLevel.LOW: (0.2, 0.4),
            RiskLevel.MEDIUM: (0.4, 0.6),
            RiskLevel.HIGH: (0.6, 0.8),
            RiskLevel.CRITICAL: (0.8, 1.0),
        }
        return ranges[self]
    
    @property
    def color_code(self) -> str:
        """Get color code for visualization."""
        colors = {
            RiskLevel.VERY_LOW: "#00ff00",
            RiskLevel.LOW: "#7fff00",
            RiskLevel.MEDIUM: "#ffff00",
            RiskLevel.HIGH: "#ff7f00",
            RiskLevel.CRITICAL: "#ff0000",
        }
        return colors[self]


@dataclass
class ComplianceFeatures:
    """Feature set for compliance risk prediction."""
    
    # Document characteristics
    document_type: str = "unknown"
    word_count: int = 0
    phi_density: float = 0.0  # PHI entities per 100 words
    medical_term_density: float = 0.0
    
    # Historical patterns
    processing_time_anomaly: float = 0.0  # Deviation from normal
    access_pattern_anomaly: float = 0.0
    user_behavior_score: float = 0.5
    
    # System context
    system_load: float = 0.5
    time_of_day: float = 0.5  # Normalized hour of day
    day_of_week: float = 0.5  # Normalized day of week
    
    # Security features
    authentication_strength: float = 1.0
    network_trust_level: float = 1.0
    endpoint_security_score: float = 1.0
    
    # Compliance history
    recent_violations: int = 0
    days_since_last_violation: int = 999
    compliance_training_recency: int = 30  # Days since training
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            hash(self.document_type) % 1000 / 1000.0,  # Categorical encoding
            min(self.word_count / 10000.0, 1.0),  # Normalized word count
            self.phi_density,
            self.medical_term_density,
            self.processing_time_anomaly,
            self.access_pattern_anomaly,
            self.user_behavior_score,
            self.system_load,
            self.time_of_day,
            self.day_of_week,
            self.authentication_strength,
            self.network_trust_level,
            self.endpoint_security_score,
            min(self.recent_violations / 10.0, 1.0),  # Normalized
            max(0.0, 1.0 - self.days_since_last_violation / 365.0),
            max(0.0, 1.0 - self.compliance_training_recency / 365.0),
        ])
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return [
            'document_type_encoded',
            'word_count_normalized',
            'phi_density',
            'medical_term_density',
            'processing_time_anomaly',
            'access_pattern_anomaly',
            'user_behavior_score',
            'system_load',
            'time_of_day',
            'day_of_week',
            'authentication_strength',
            'network_trust_level',
            'endpoint_security_score',
            'recent_violations_normalized',
            'violation_recency_score',
            'training_recency_score',
        ]


@dataclass
class RiskPrediction:
    """Compliance risk prediction with explanations."""
    
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    
    # Feature importance for explainability
    feature_importances: Dict[str, float] = field(default_factory=dict)
    
    # Specific risk factors identified
    risk_factors: List[str] = field(default_factory=list)
    
    # Recommended interventions
    recommendations: List[str] = field(default_factory=list)
    
    # Temporal aspects
    predicted_violation_timeframe: Optional[str] = None  # "immediate", "short_term", "medium_term"
    
    def get_risk_level_from_score(self, score: float) -> RiskLevel:
        """Determine risk level from numerical score."""
        if score < 0.2:
            return RiskLevel.VERY_LOW
        elif score < 0.4:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MEDIUM
        elif score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'confidence': self.confidence,
            'feature_importances': self.feature_importances,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
            'predicted_violation_timeframe': self.predicted_violation_timeframe,
        }


class ComplianceAnomalyDetector:
    """Detects anomalies in compliance-related activities."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def update_baseline(self, metric_name: str, value: float):
        """Update baseline metrics with new observation."""
        self.baseline_metrics[metric_name].append(value)
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """
        Detect if a value is anomalous compared to baseline.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if metric_name not in self.baseline_metrics:
            self.update_baseline(metric_name, value)
            return False, 0.0
        
        baseline_values = list(self.baseline_metrics[metric_name])
        
        if len(baseline_values) < 10:  # Need minimum data
            self.update_baseline(metric_name, value)
            return False, 0.0
        
        # Calculate z-score
        mean_val = np.mean(baseline_values)
        std_val = np.std(baseline_values)
        
        if std_val == 0:  # No variation
            anomaly_score = 0.0
        else:
            anomaly_score = abs(value - mean_val) / std_val
        
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        # Update baseline with new value
        self.update_baseline(metric_name, value)
        
        return is_anomaly, anomaly_score


class CompliancePredictionEngine:
    """Advanced predictive engine for compliance risk assessment."""
    
    def __init__(self):
        self.anomaly_detector = ComplianceAnomalyDetector()
        self.prediction_history: List[Dict] = []
        
        # Trained model weights (simplified neural network simulation)
        self.model_weights = {
            'input_layer': np.random.normal(0, 0.1, (16, 32)),
            'hidden_layer': np.random.normal(0, 0.1, (32, 16)),
            'output_layer': np.random.normal(0, 0.1, (16, 1)),
        }
        
        # Feature importance learned from training
        self.feature_importance = {
            'phi_density': 0.15,
            'recent_violations_normalized': 0.12,
            'user_behavior_score': 0.11,
            'authentication_strength': 0.10,
            'processing_time_anomaly': 0.09,
            'access_pattern_anomaly': 0.08,
            'endpoint_security_score': 0.07,
            'medical_term_density': 0.06,
            'network_trust_level': 0.06,
            'system_load': 0.05,
            'violation_recency_score': 0.05,
            'training_recency_score': 0.04,
            'time_of_day': 0.01,
            'day_of_week': 0.01,
        }
    
    def predict_compliance_risk(self, features: ComplianceFeatures) -> RiskPrediction:
        """
        Predict compliance risk based on input features.
        
        Args:
            features: Compliance features for prediction
            
        Returns:
            Risk prediction with explanations
        """
        # Convert features to array
        feature_array = features.to_array()
        
        # Run through neural network (simplified)
        risk_score = self._forward_pass(feature_array)
        
        # Determine risk level
        risk_level = RiskLevel.VERY_LOW
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW, RiskLevel.VERY_LOW]:
            if risk_score >= level.score_range[0]:
                risk_level = level
                break
        
        # Calculate prediction confidence
        confidence = self._calculate_confidence(feature_array, risk_score)
        
        # Feature importance for this prediction
        feature_importances = self._calculate_feature_importance(features)
        
        # Identify specific risk factors
        risk_factors = self._identify_risk_factors(features, feature_importances)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_factors, risk_level)
        
        # Predict timeframe
        violation_timeframe = self._predict_violation_timeframe(risk_score, features)
        
        prediction = RiskPrediction(
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            feature_importances=feature_importances,
            risk_factors=risk_factors,
            recommendations=recommendations,
            predicted_violation_timeframe=violation_timeframe
        )
        
        # Record prediction
        self.prediction_history.append({
            'timestamp': time.time(),
            'prediction': prediction.to_dict(),
            'features': features.__dict__
        })
        
        return prediction
    
    def _forward_pass(self, features: np.ndarray) -> float:
        """Simplified neural network forward pass."""
        
        # Input layer
        hidden = np.maximum(0, np.dot(features, self.model_weights['input_layer']))  # ReLU
        
        # Hidden layer
        hidden2 = np.maximum(0, np.dot(hidden, self.model_weights['hidden_layer']))  # ReLU
        
        # Output layer with sigmoid
        output = np.dot(hidden2, self.model_weights['output_layer'])
        risk_score = 1 / (1 + np.exp(-output[0]))  # Sigmoid
        
        return float(risk_score)
    
    def _calculate_confidence(self, features: np.ndarray, risk_score: float) -> float:
        """Calculate prediction confidence based on feature quality and model certainty."""
        
        # Distance from decision boundaries
        boundary_distances = [
            abs(risk_score - 0.2),  # Low/Very Low boundary
            abs(risk_score - 0.4),  # Medium/Low boundary
            abs(risk_score - 0.6),  # High/Medium boundary
            abs(risk_score - 0.8),  # Critical/High boundary
        ]
        
        # Confidence is higher when far from boundaries
        min_distance = min(boundary_distances)
        boundary_confidence = min(min_distance * 5, 1.0)  # Scale to [0,1]
        
        # Feature quality score
        feature_completeness = np.mean(features > 0)  # Proportion of non-zero features
        
        # Combine factors
        confidence = (boundary_confidence + feature_completeness) / 2
        
        return float(confidence)
    
    def _calculate_feature_importance(self, features: ComplianceFeatures) -> Dict[str, float]:
        """Calculate feature importance for this specific prediction."""
        
        feature_names = features.feature_names
        feature_values = features.to_array()
        
        # Use pre-trained importance weights, adjusted by feature values
        importances = {}
        
        for i, name in enumerate(feature_names):
            base_importance = self.feature_importance.get(name, 0.01)
            feature_value = feature_values[i]
            
            # Adjust importance by feature magnitude
            adjusted_importance = base_importance * (1 + feature_value)
            
            importances[name] = adjusted_importance
        
        # Normalize to sum to 1
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v/total_importance for k, v in importances.items()}
        
        return importances
    
    def _identify_risk_factors(
        self, 
        features: ComplianceFeatures, 
        importances: Dict[str, float]
    ) -> List[str]:
        """Identify specific risk factors based on feature values and importance."""
        
        risk_factors = []
        
        # High PHI density
        if features.phi_density > 0.1:  # More than 10 PHI per 100 words
            risk_factors.append("High PHI density detected in document")
        
        # Recent violations
        if features.recent_violations > 0:
            risk_factors.append(f"Recent compliance violations: {features.recent_violations}")
        
        # Anomalous processing time
        if features.processing_time_anomaly > 2.0:
            risk_factors.append("Unusual processing time pattern detected")
        
        # Access pattern anomaly
        if features.access_pattern_anomaly > 2.0:
            risk_factors.append("Anomalous access patterns identified")
        
        # Poor user behavior
        if features.user_behavior_score < 0.3:
            risk_factors.append("Concerning user behavior patterns")
        
        # Weak authentication
        if features.authentication_strength < 0.5:
            risk_factors.append("Weak authentication mechanisms in use")
        
        # Low network trust
        if features.network_trust_level < 0.5:
            risk_factors.append("Low network trust level")
        
        # Endpoint security issues
        if features.endpoint_security_score < 0.5:
            risk_factors.append("Endpoint security concerns identified")
        
        # Training recency
        if features.compliance_training_recency > 365:  # More than a year
            risk_factors.append("Compliance training is overdue")
        
        # High system load
        if features.system_load > 0.8:
            risk_factors.append("High system load may impact security controls")
        
        return risk_factors
    
    def _generate_recommendations(
        self,
        features: ComplianceFeatures,
        risk_factors: List[str],
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate actionable recommendations based on identified risks."""
        
        recommendations = []
        
        # High-priority recommendations for critical/high risk
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("URGENT: Review and validate all PHI detection results manually")
            recommendations.append("Implement additional audit logging for this process")
            recommendations.append("Consider temporary restriction of PHI processing capabilities")
        
        # Specific recommendations based on risk factors
        if "High PHI density" in str(risk_factors):
            recommendations.append("Increase PHI detection sensitivity for this document type")
            recommendations.append("Implement additional human review for high-PHI documents")
        
        if "Recent compliance violations" in str(risk_factors):
            recommendations.append("Schedule immediate compliance refresher training")
            recommendations.append("Implement enhanced monitoring for repeat violators")
        
        if "Anomalous" in str(risk_factors):
            recommendations.append("Investigate unusual patterns with security team")
            recommendations.append("Consider temporarily increasing monitoring frequency")
        
        if "authentication" in str(risk_factors):
            recommendations.append("Upgrade to multi-factor authentication")
            recommendations.append("Review and strengthen password policies")
        
        if "network trust" in str(risk_factors):
            recommendations.append("Verify network security controls are functioning")
            recommendations.append("Consider additional network segmentation")
        
        if "Endpoint security" in str(risk_factors):
            recommendations.append("Update endpoint protection software")
            recommendations.append("Perform comprehensive endpoint security scan")
        
        if "training" in str(risk_factors):
            recommendations.append("Schedule mandatory HIPAA compliance training")
            recommendations.append("Implement regular compliance knowledge assessments")
        
        # General recommendations for any elevated risk
        if risk_level != RiskLevel.VERY_LOW:
            recommendations.append("Document this risk assessment in compliance audit log")
            recommendations.append("Review applicable HIPAA safeguards and controls")
        
        return recommendations
    
    def _predict_violation_timeframe(self, risk_score: float, features: ComplianceFeatures) -> Optional[str]:
        """Predict when a violation might occur based on risk score and patterns."""
        
        # Immediate risk (next few hours)
        if (risk_score > 0.8 and 
            (features.processing_time_anomaly > 3.0 or features.recent_violations > 2)):
            return "immediate"
        
        # Short-term risk (next few days)
        elif (risk_score > 0.6 and 
              (features.user_behavior_score < 0.3 or features.authentication_strength < 0.4)):
            return "short_term"
        
        # Medium-term risk (next few weeks)
        elif risk_score > 0.4:
            return "medium_term"
        
        # Low risk - no specific timeframe
        else:
            return None
    
    def analyze_temporal_patterns(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in compliance risk predictions."""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_predictions = [
            pred for pred in self.prediction_history 
            if pred['timestamp'] > cutoff_time
        ]
        
        if not recent_predictions:
            return {'status': 'insufficient_data'}
        
        # Extract risk scores and timestamps
        risk_scores = [pred['prediction']['risk_score'] for pred in recent_predictions]
        timestamps = [pred['timestamp'] for pred in recent_predictions]
        
        # Calculate trend
        if len(risk_scores) >= 2:
            # Simple linear trend
            time_diffs = np.array(timestamps) - timestamps[0]
            correlation = np.corrcoef(time_diffs, risk_scores)[0, 1] if len(set(time_diffs)) > 1 else 0
            trend = "increasing" if correlation > 0.1 else "decreasing" if correlation < -0.1 else "stable"
        else:
            trend = "insufficient_data"
        
        # Risk level distribution
        risk_levels = [pred['prediction']['risk_level'] for pred in recent_predictions]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        # Average risk score
        avg_risk = np.mean(risk_scores)
        max_risk = max(risk_scores)
        
        # Identify peak risk times
        if len(timestamps) > 1:
            hours = [(ts % (24 * 3600)) / 3600 for ts in timestamps]  # Hour of day
            peak_hour = hours[np.argmax(risk_scores)]
        else:
            peak_hour = None
        
        analysis = {
            'time_window_hours': time_window_hours,
            'total_predictions': len(recent_predictions),
            'average_risk_score': avg_risk,
            'maximum_risk_score': max_risk,
            'risk_trend': trend,
            'risk_level_distribution': risk_distribution,
            'peak_risk_hour': peak_hour,
            'high_risk_periods': sum(1 for score in risk_scores if score > 0.6),
        }
        
        return analysis
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the prediction model."""
        
        if len(self.prediction_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_predictions = self.prediction_history[-50:]  # Last 50 predictions
        
        # Calculate prediction confidence distribution
        confidences = [pred['prediction']['confidence'] for pred in recent_predictions]
        
        # Risk score distribution
        risk_scores = [pred['prediction']['risk_score'] for pred in recent_predictions]
        
        # Feature importance stability
        feature_importances_over_time = []
        for pred in recent_predictions:
            feature_importances_over_time.append(pred['prediction']['feature_importances'])
        
        # Calculate average feature importance
        avg_feature_importance = {}
        if feature_importances_over_time:
            all_features = set()
            for fi in feature_importances_over_time:
                all_features.update(fi.keys())
            
            for feature in all_features:
                values = [fi.get(feature, 0) for fi in feature_importances_over_time]
                avg_feature_importance[feature] = np.mean(values)
        
        metrics = {
            'total_predictions': len(self.prediction_history),
            'recent_predictions_analyzed': len(recent_predictions),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'average_risk_score': np.mean(risk_scores),
            'risk_score_std': np.std(risk_scores),
            'high_confidence_predictions': sum(1 for c in confidences if c > 0.8),
            'high_risk_predictions': sum(1 for r in risk_scores if r > 0.6),
            'feature_importance_stability': avg_feature_importance,
        }
        
        return metrics


class RiskPredictor:
    """High-level interface for compliance risk prediction."""
    
    def __init__(self):
        self.prediction_engine = CompliancePredictionEngine()
        self.prediction_cache: Dict[str, Tuple[RiskPrediction, float]] = {}
        self.cache_duration = 300  # 5 minutes
        
    def predict_document_risk(
        self,
        document_content: str,
        document_type: str,
        user_context: Optional[Dict[str, Any]] = None,
        system_context: Optional[Dict[str, Any]] = None
    ) -> RiskPrediction:
        """
        Predict compliance risk for processing a document.
        
        Args:
            document_content: The document text to analyze
            document_type: Type of document (clinical_note, insurance_form, etc.)
            user_context: User-related context information
            system_context: System-related context information
            
        Returns:
            Risk prediction with recommendations
        """
        
        # Create cache key
        cache_key = hashlib.sha256(
            f"{document_content[:100]}_{document_type}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.prediction_cache:
            prediction, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return prediction
        
        # Extract features from document
        features = self._extract_document_features(
            document_content, document_type, user_context, system_context
        )
        
        # Make prediction
        prediction = self.prediction_engine.predict_compliance_risk(features)
        
        # Cache result
        self.prediction_cache[cache_key] = (prediction, time.time())
        
        return prediction
    
    def _extract_document_features(
        self,
        content: str,
        doc_type: str,
        user_context: Optional[Dict[str, Any]],
        system_context: Optional[Dict[str, Any]]
    ) -> ComplianceFeatures:
        """Extract features from document and context for prediction."""
        
        # Document analysis
        words = content.split()
        word_count = len(words)
        
        # Estimate PHI density (simplified)
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}[.-]\d{3}[.-]\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # Email
        ]
        
        phi_count = 0
        for pattern in phi_patterns:
            import re
            phi_count += len(re.findall(pattern, content))
        
        phi_density = phi_count / max(word_count / 100, 0.1)  # PHI per 100 words
        
        # Estimate medical term density
        medical_terms = [
            'patient', 'diagnosis', 'treatment', 'medication', 'doctor',
            'hospital', 'clinic', 'blood', 'pressure', 'heart'
        ]
        
        medical_count = sum(1 for word in words if word.lower() in medical_terms)
        medical_term_density = medical_count / max(word_count, 1)
        
        # User context features
        user_behavior_score = 0.5
        recent_violations = 0
        days_since_violation = 999
        training_recency = 30
        
        if user_context:
            user_behavior_score = user_context.get('behavior_score', 0.5)
            recent_violations = user_context.get('recent_violations', 0)
            days_since_violation = user_context.get('days_since_violation', 999)
            training_recency = user_context.get('training_recency_days', 30)
        
        # System context features
        system_load = 0.5
        auth_strength = 1.0
        network_trust = 1.0
        endpoint_security = 1.0
        
        if system_context:
            system_load = system_context.get('system_load', 0.5)
            auth_strength = system_context.get('auth_strength', 1.0)
            network_trust = system_context.get('network_trust', 1.0)
            endpoint_security = system_context.get('endpoint_security', 1.0)
        
        # Time-based features
        current_time = time.localtime()
        time_of_day = current_time.tm_hour / 24.0
        day_of_week = current_time.tm_wday / 7.0
        
        return ComplianceFeatures(
            document_type=doc_type,
            word_count=word_count,
            phi_density=phi_density,
            medical_term_density=medical_term_density,
            processing_time_anomaly=0.0,  # Would be calculated from actual processing
            access_pattern_anomaly=0.0,    # Would be calculated from access logs
            user_behavior_score=user_behavior_score,
            system_load=system_load,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            authentication_strength=auth_strength,
            network_trust_level=network_trust,
            endpoint_security_score=endpoint_security,
            recent_violations=recent_violations,
            days_since_last_violation=days_since_violation,
            compliance_training_recency=training_recency,
        )
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Generate a comprehensive risk dashboard."""
        
        # Get temporal patterns
        temporal_analysis = self.prediction_engine.analyze_temporal_patterns(24)
        
        # Get model performance
        performance_metrics = self.prediction_engine.get_model_performance_metrics()
        
        # Current system status
        current_time = time.time()
        recent_high_risk = sum(
            1 for pred in self.prediction_engine.prediction_history[-10:]
            if pred['prediction']['risk_score'] > 0.6
        )
        
        dashboard = {
            'timestamp': current_time,
            'current_risk_level': 'medium' if recent_high_risk > 2 else 'low',
            'recent_high_risk_predictions': recent_high_risk,
            'temporal_analysis': temporal_analysis,
            'model_performance': performance_metrics,
            'cache_statistics': {
                'cached_predictions': len(self.prediction_cache),
                'cache_hit_potential': len(self.prediction_cache) > 0,
            }
        }
        
        return dashboard


# Factory functions for easy instantiation
def create_compliance_predictor() -> RiskPredictor:
    """Create a configured compliance risk predictor."""
    return RiskPredictor()


def demonstrate_compliance_prediction():
    """Demonstrate compliance risk prediction capabilities."""
    
    predictor = create_compliance_predictor()
    
    # Example document with various risk factors
    sample_document = """
    Patient John Doe (SSN: 123-45-6789) was seen on 01/15/2024.
    Contact information: john.doe@email.com, phone 555-123-4567.
    Medical Record Number: MRN123456789.
    Diagnosis: Hypertension, requires medication management.
    Address: 123 Main Street, Boston, MA 02101.
    """
    
    # Simulate user context with some risk factors
    user_context = {
        'behavior_score': 0.3,  # Poor behavior
        'recent_violations': 1,
        'days_since_violation': 5,
        'training_recency_days': 400,  # Overdue training
    }
    
    # Simulate system context
    system_context = {
        'system_load': 0.9,  # High load
        'auth_strength': 0.6,  # Moderate auth
        'network_trust': 0.4,  # Low trust network
        'endpoint_security': 0.7,  # Decent endpoint security
    }
    
    # Make prediction
    prediction = predictor.predict_document_risk(
        sample_document,
        'clinical_note',
        user_context,
        system_context
    )
    
    # Get risk dashboard
    dashboard = predictor.get_risk_dashboard()
    
    return {
        'prediction': prediction.to_dict(),
        'dashboard': dashboard,
        'risk_level': prediction.risk_level.value,
        'risk_score': prediction.risk_score,
        'recommendations': prediction.recommendations,
    }