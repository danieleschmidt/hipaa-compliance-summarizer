"""
Research and experimental algorithms for HIPAA compliance optimization.

This module contains cutting-edge research implementations for:
- Novel PHI detection algorithms with statistical validation
- Advanced ML-based compliance scoring
- Federated learning for privacy-preserving model improvement
- Predictive analytics for compliance risk assessment
"""

from .adaptive_phi_detection import AdaptivePHIDetector, PHIConfidenceModel
from .benchmark_suite import ComparativeAnalysis, ResearchBenchmarkSuite
from .compliance_prediction import (
    CompliancePredictionEngine,
    RiskLevel,
    RiskPrediction,
    RiskPredictor,
)
from .federated_learning import (
    FederatedComplianceModel,
    PrivacyBudget,
    PrivacyPreservingTrainer,
)
from .statistical_validation import StatisticalValidator, ValidationMetrics

__all__ = [
    "AdaptivePHIDetector",
    "PHIConfidenceModel",
    "CompliancePredictionEngine",
    "RiskPredictor",
    "RiskLevel",
    "RiskPrediction",
    "FederatedComplianceModel",
    "PrivacyPreservingTrainer",
    "PrivacyBudget",
    "StatisticalValidator",
    "ValidationMetrics",
    "ResearchBenchmarkSuite",
    "ComparativeAnalysis",
]
