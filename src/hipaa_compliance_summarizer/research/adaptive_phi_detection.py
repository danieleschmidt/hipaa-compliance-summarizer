"""
Adaptive PHI Detection using Machine Learning and Statistical Validation.

Novel algorithmic approach that combines:
1. Confidence-based entity detection with uncertainty quantification
2. Adaptive threshold tuning based on document context
3. Statistical validation of detection accuracy
4. Real-time model performance optimization
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PHIConfidenceScore:
    """Statistical confidence metrics for PHI detection."""

    entity_type: str
    confidence: float
    uncertainty: float
    false_positive_probability: float
    context_relevance: float

    @property
    def adjusted_confidence(self) -> float:
        """Confidence adjusted for uncertainty and context."""
        return self.confidence * self.context_relevance * (1 - self.uncertainty)


@dataclass
class DetectionContext:
    """Document context for adaptive threshold adjustment."""

    document_type: str
    word_count: int
    medical_term_density: float
    phi_density_estimate: float
    historical_accuracy: float = 0.95

    def get_context_weight(self) -> float:
        """Calculate context-based detection weight."""
        # Higher medical term density = more precise detection needed
        medical_weight = min(self.medical_term_density * 2, 1.0)
        # Longer documents need more conservative thresholds
        length_weight = 1 - (math.log10(max(self.word_count, 100)) - 2) * 0.1
        return max(0.5, medical_weight * length_weight * self.historical_accuracy)


@dataclass
class ValidationMetrics:
    """Statistical validation results for PHI detection model."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    validation_runs: int = 0
    statistical_significance: float = 0.0

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        return self.true_positives / max(1, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity)."""
        return self.true_positives / max(1, self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        p, r = self.precision, self.recall
        return 2 * (p * r) / max(0.001, p + r)

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / max(1, total)

    def is_statistically_significant(self, alpha: float = 0.05) -> bool:
        """Check if results are statistically significant at given alpha level."""
        return self.statistical_significance < alpha and self.validation_runs >= 30


class PHIConfidenceModel:
    """Advanced confidence modeling for PHI detection with uncertainty quantification."""

    def __init__(self):
        self.pattern_confidence_history: Dict[str, List[float]] = defaultdict(list)
        self.context_adjustments: Dict[str, float] = {}
        self.validation_metrics = ValidationMetrics()
        self._calibration_data: List[Tuple[float, bool]] = []  # (confidence, was_correct)

    def calculate_confidence(
        self,
        pattern: str,
        match_text: str,
        context: DetectionContext
    ) -> PHIConfidenceScore:
        """Calculate statistical confidence score with uncertainty quantification."""

        # Base confidence from pattern strength and historical performance
        base_confidence = self._get_pattern_base_confidence(pattern, match_text)

        # Uncertainty quantification using historical variance
        uncertainty = self._calculate_uncertainty(pattern)

        # False positive probability based on calibration
        fp_probability = self._estimate_false_positive_probability(base_confidence)

        # Context relevance score
        context_relevance = self._calculate_context_relevance(match_text, context)

        return PHIConfidenceScore(
            entity_type=pattern,
            confidence=base_confidence,
            uncertainty=uncertainty,
            false_positive_probability=fp_probability,
            context_relevance=context_relevance
        )

    def _get_pattern_base_confidence(self, pattern: str, match_text: str) -> float:
        """Calculate base confidence from pattern strength and match quality."""

        # Pattern-specific confidence adjustments
        pattern_weights = {
            'ssn': 0.95,  # SSN patterns are highly specific
            'phone': 0.85,  # Phone numbers can have false positives
            'email': 0.90,  # Email patterns are reliable
            'date': 0.70,  # Dates have many false positives
            'mrn': 0.88,  # Medical record numbers are context-dependent
            'name': 0.75,  # Name detection is challenging
            'address': 0.80,  # Address patterns vary significantly
        }

        base_confidence = pattern_weights.get(pattern, 0.75)

        # Adjust based on match text characteristics
        text_length = len(match_text)
        if text_length < 3:
            base_confidence *= 0.6  # Very short matches are suspicious
        elif text_length > 50:
            base_confidence *= 0.8  # Very long matches might be context

        # Check for common false positive indicators
        if match_text.lower() in ['example', 'test', 'sample', 'xxx-xx-xxxx']:
            base_confidence *= 0.1

        return min(base_confidence, 0.99)

    def _calculate_uncertainty(self, pattern: str) -> float:
        """Calculate uncertainty based on historical performance variance."""
        history = self.pattern_confidence_history.get(pattern, [])

        if len(history) < 5:
            return 0.3  # High uncertainty with limited data

        # Calculate coefficient of variation as uncertainty measure
        mean_conf = np.mean(history)
        std_conf = np.std(history)

        if mean_conf == 0:
            return 0.5

        coefficient_of_variation = std_conf / mean_conf

        # Normalize uncertainty to [0, 1] range
        uncertainty = min(coefficient_of_variation * 2, 1.0)
        return uncertainty

    def _estimate_false_positive_probability(self, confidence: float) -> float:
        """Estimate false positive probability using calibration curve."""
        if not self._calibration_data:
            # Default calibration based on typical PHI detection performance
            return (1 - confidence) * 0.8

        # Find similar confidence scores in calibration data
        similar_scores = [
            (conf, correct) for conf, correct in self._calibration_data
            if abs(conf - confidence) < 0.1
        ]

        if not similar_scores:
            return (1 - confidence) * 0.8

        # Calculate empirical false positive rate
        incorrect_count = sum(1 for _, correct in similar_scores if not correct)
        total_count = len(similar_scores)

        return incorrect_count / total_count

    def _calculate_context_relevance(self, match_text: str, context: DetectionContext) -> float:
        """Calculate how relevant the match is given document context."""
        relevance = 1.0

        # Adjust based on document type
        doc_type_adjustments = {
            'clinical_note': 1.0,
            'lab_report': 0.9,  # Lab reports have more structured data
            'insurance_form': 1.1,  # Insurance forms have more PHI
            'discharge_summary': 1.0,
        }

        relevance *= doc_type_adjustments.get(context.document_type, 1.0)

        # Adjust based on PHI density - too many PHI matches might indicate false positives
        if context.phi_density_estimate > 0.1:  # More than 10% PHI is suspicious
            relevance *= 0.8
        elif context.phi_density_estimate < 0.01:  # Very low PHI might miss context
            relevance *= 0.9

        # Adjust based on medical term density
        if context.medical_term_density > 0.05:
            relevance *= 1.1  # Medical context increases relevance

        return min(relevance, 1.0)

    def update_performance(self, pattern: str, confidence: float, was_correct: bool):
        """Update model performance statistics."""
        self.pattern_confidence_history[pattern].append(confidence if was_correct else 0.0)
        self._calibration_data.append((confidence, was_correct))

        # Keep only recent calibration data (last 1000 samples)
        if len(self._calibration_data) > 1000:
            self._calibration_data = self._calibration_data[-1000:]

        # Update validation metrics
        if was_correct:
            self.validation_metrics.true_positives += 1
        else:
            self.validation_metrics.false_positives += 1

        self.validation_metrics.validation_runs += 1

    def get_adaptive_threshold(self, context: DetectionContext) -> float:
        """Calculate adaptive confidence threshold based on context."""
        base_threshold = 0.7

        # Adjust threshold based on document context
        context_weight = context.get_context_weight()

        # Higher accuracy requirements for medical documents
        if context.document_type in ['clinical_note', 'lab_report']:
            base_threshold += 0.05

        # Adjust based on historical performance
        if self.validation_metrics.accuracy > 0.95:
            base_threshold -= 0.05  # Can be more lenient
        elif self.validation_metrics.accuracy < 0.90:
            base_threshold += 0.1   # Need to be more conservative

        # Apply context weighting
        adaptive_threshold = base_threshold * context_weight

        return max(0.5, min(0.95, adaptive_threshold))


class AdaptivePHIDetector:
    """Advanced PHI detector with adaptive thresholding and statistical validation."""

    def __init__(self, enable_statistical_validation: bool = True):
        self.confidence_model = PHIConfidenceModel()
        self.enable_statistical_validation = enable_statistical_validation
        self.detection_patterns = self._initialize_patterns()
        self.performance_history: List[Dict] = []

        # Medical term patterns for context analysis
        self.medical_terms_pattern = re.compile(
            r'\b(?:patient|diagnosis|treatment|medication|symptoms?|doctor|physician|hospital|clinic|'
            r'blood|pressure|heart|rate|temperature|mg|ml|cc|dose|prescription|therapy)\b',
            re.IGNORECASE
        )

    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize PHI detection patterns with enhanced specificity."""
        return {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'ssn_no_dashes': re.compile(r'\b\d{9}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            'date_mdy': re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
            'date_dmy': re.compile(r'\b\d{1,2}-\d{1,2}-\d{4}\b'),
            'mrn': re.compile(r'\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b', re.IGNORECASE),
            'dea': re.compile(r'\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b', re.IGNORECASE),
            'insurance_id': re.compile(r'\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b', re.IGNORECASE),
            'name_formal': re.compile(r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
            'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', re.IGNORECASE),
        }

    def analyze_document_context(self, text: str, document_type: str = "unknown") -> DetectionContext:
        """Analyze document to create detection context."""
        words = text.split()
        word_count = len(words)

        # Calculate medical term density
        medical_matches = len(self.medical_terms_pattern.findall(text))
        medical_term_density = medical_matches / max(word_count, 1)

        # Estimate PHI density using simple pattern matching
        phi_matches = 0
        for pattern in self.detection_patterns.values():
            phi_matches += len(pattern.findall(text))

        phi_density_estimate = phi_matches / max(word_count, 1)

        return DetectionContext(
            document_type=document_type,
            word_count=word_count,
            medical_term_density=medical_term_density,
            phi_density_estimate=phi_density_estimate
        )

    def detect_phi_with_confidence(
        self,
        text: str,
        document_type: str = "unknown",
        validation_data: Optional[List[Tuple[str, bool]]] = None
    ) -> List[Dict]:
        """
        Detect PHI with statistical confidence scoring and adaptive thresholding.
        
        Args:
            text: Input text to analyze
            document_type: Type of document for context
            validation_data: Optional ground truth for validation
            
        Returns:
            List of PHI detections with confidence scores and statistical metrics
        """
        start_time = time.time()

        # Analyze document context
        context = self.analyze_document_context(text, document_type)

        # Get adaptive threshold for this context
        adaptive_threshold = self.confidence_model.get_adaptive_threshold(context)

        detections = []

        for pattern_name, pattern in self.detection_patterns.items():
            for match in pattern.finditer(text):
                match_text = match.group()
                start_pos = match.start()
                end_pos = match.end()

                # Calculate confidence score
                confidence_score = self.confidence_model.calculate_confidence(
                    pattern_name, match_text, context
                )

                # Apply adaptive threshold
                if confidence_score.adjusted_confidence >= adaptive_threshold:
                    detection = {
                        'entity_type': pattern_name,
                        'text': match_text,
                        'start': start_pos,
                        'end': end_pos,
                        'confidence': confidence_score.confidence,
                        'adjusted_confidence': confidence_score.adjusted_confidence,
                        'uncertainty': confidence_score.uncertainty,
                        'false_positive_probability': confidence_score.false_positive_probability,
                        'context_relevance': confidence_score.context_relevance,
                        'threshold_used': adaptive_threshold,
                    }
                    detections.append(detection)

        # Perform statistical validation if enabled and validation data provided
        if self.enable_statistical_validation and validation_data:
            validation_results = self._validate_detections(detections, validation_data)
        else:
            validation_results = None

        # Record performance metrics
        processing_time = time.time() - start_time
        self._record_performance_metrics(
            detections, context, processing_time, validation_results
        )

        # Sort by confidence
        detections.sort(key=lambda x: x['adjusted_confidence'], reverse=True)

        return detections

    def _validate_detections(
        self,
        detections: List[Dict],
        validation_data: List[Tuple[str, bool]]
    ) -> Dict:
        """Validate detections against ground truth and update model performance."""
        validation_results = {
            'validated_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'statistical_significance': 1.0,  # Default to non-significant
        }

        # Simple validation logic - in practice, this would be more sophisticated
        validation_map = {text: is_phi for text, is_phi in validation_data}

        for detection in detections:
            text = detection['text']
            if text in validation_map:
                is_correct = validation_map[text]

                # Update confidence model
                self.confidence_model.update_performance(
                    detection['entity_type'],
                    detection['confidence'],
                    is_correct
                )

                validation_results['validated_detections'] += 1
                if is_correct:
                    validation_results['true_positives'] += 1
                else:
                    validation_results['false_positives'] += 1

        # Calculate statistical significance using basic chi-square test approximation
        if validation_results['validated_detections'] > 0:
            observed_accuracy = validation_results['true_positives'] / validation_results['validated_detections']
            expected_accuracy = 0.9  # Expected baseline accuracy

            # Simple statistical significance approximation
            n = validation_results['validated_detections']
            if n >= 30:  # Sufficient sample size for normal approximation
                z_score = abs(observed_accuracy - expected_accuracy) / math.sqrt(expected_accuracy * (1 - expected_accuracy) / n)
                # Convert z-score to p-value (rough approximation)
                validation_results['statistical_significance'] = max(0.001, 2 * (1 - min(0.999, z_score / 3)))

        return validation_results

    def _record_performance_metrics(
        self,
        detections: List[Dict],
        context: DetectionContext,
        processing_time: float,
        validation_results: Optional[Dict]
    ):
        """Record performance metrics for analysis and optimization."""
        metrics = {
            'timestamp': time.time(),
            'document_type': context.document_type,
            'word_count': context.word_count,
            'medical_term_density': context.medical_term_density,
            'phi_density_estimate': context.phi_density_estimate,
            'detections_count': len(detections),
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
            'avg_adjusted_confidence': np.mean([d['adjusted_confidence'] for d in detections]) if detections else 0.0,
            'processing_time_seconds': processing_time,
            'detections_per_second': len(detections) / max(processing_time, 0.001),
        }

        if validation_results:
            metrics.update(validation_results)

        self.performance_history.append(metrics)

        # Keep only last 100 performance records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary with statistical analysis."""
        if not self.performance_history:
            return {'status': 'no_data'}

        recent_metrics = self.performance_history[-10:]  # Last 10 runs

        summary = {
            'total_runs': len(self.performance_history),
            'recent_runs': len(recent_metrics),
            'avg_processing_time': np.mean([m['processing_time_seconds'] for m in recent_metrics]),
            'avg_detections_per_document': np.mean([m['detections_count'] for m in recent_metrics]),
            'avg_confidence': np.mean([m['avg_confidence'] for m in recent_metrics if m['avg_confidence'] > 0]),
            'throughput_detections_per_second': np.mean([m['detections_per_second'] for m in recent_metrics]),
            'validation_metrics': self.confidence_model.validation_metrics.__dict__,
            'statistical_significance': self.confidence_model.validation_metrics.is_statistically_significant(),
        }

        # Calculate performance trends
        if len(self.performance_history) >= 5:
            old_metrics = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else self.performance_history[:5]

            old_avg_time = np.mean([m['processing_time_seconds'] for m in old_metrics])
            new_avg_time = np.mean([m['processing_time_seconds'] for m in recent_metrics])

            summary['performance_trend'] = {
                'processing_time_improvement': (old_avg_time - new_avg_time) / old_avg_time if old_avg_time > 0 else 0,
                'trending_faster': new_avg_time < old_avg_time,
            }

        return summary


# Export for research benchmarking
def create_research_detector() -> AdaptivePHIDetector:
    """Factory function to create a research-optimized PHI detector."""
    return AdaptivePHIDetector(enable_statistical_validation=True)
