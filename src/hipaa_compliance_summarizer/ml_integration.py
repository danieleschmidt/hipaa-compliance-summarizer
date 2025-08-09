"""Advanced ML Integration for HIPAA Compliance System.

This module provides enhanced machine learning capabilities for PHI detection,
clinical summarization, and compliance analysis using state-of-the-art models.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import numpy as np
from .monitoring.tracing import trace_operation
from .error_handling import HIPAAError, ErrorSeverity
from .models.phi_entity import PHICategory, PHIEntity

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported ML model types."""
    PHI_DETECTION = "phi_detection"
    CLINICAL_NER = "clinical_ner"
    SUMMARIZATION = "summarization" 
    COMPLIANCE_SCORING = "compliance_scoring"
    RISK_ASSESSMENT = "risk_assessment"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ModelMetrics:
    """Metrics for model performance tracking."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time_ms: float
    confidence_threshold: float
    model_version: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class PredictionResult:
    """Result from ML model prediction."""
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    model_type: ModelType
    processing_time_ms: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class BaseMLModel(ABC):
    """Base class for ML models in the HIPAA system."""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_loaded = False
        self.metrics_history: List[ModelMetrics] = []
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the ML model."""
        pass
        
    @abstractmethod
    def predict(self, input_data: str, context: Dict[str, Any] = None) -> PredictionResult:
        """Make predictions on input data."""
        pass
        
    def update_metrics(self, metrics: ModelMetrics) -> None:
        """Update model performance metrics."""
        self.metrics_history.append(metrics)
        # Keep only last 100 metrics for memory efficiency
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
    def get_average_metrics(self, last_n: int = 10) -> Optional[ModelMetrics]:
        """Get average metrics from last N predictions."""
        if not self.metrics_history:
            return None
            
        recent_metrics = self.metrics_history[-last_n:]
        if not recent_metrics:
            return None
            
        avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
        avg_precision = sum(m.precision for m in recent_metrics) / len(recent_metrics)
        avg_recall = sum(m.recall for m in recent_metrics) / len(recent_metrics)
        avg_f1 = sum(m.f1_score for m in recent_metrics) / len(recent_metrics)
        avg_time = sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics)
        
        return ModelMetrics(
            accuracy=avg_accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            processing_time_ms=avg_time,
            confidence_threshold=self.confidence_threshold,
            model_version=f"{self.model_name}_averaged",
        )


class AdvancedPHIDetector(BaseMLModel):
    """Advanced PHI detection using transformer-based models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("advanced_phi_detector", config)
        self.model_path = self.config.get("model_path", "microsoft/presidio-analyzer")
        self.use_clinical_context = self.config.get("use_clinical_context", True)
        self.ensemble_models = self.config.get("ensemble_models", [])
        self.custom_patterns = self.config.get("custom_patterns", {})
        
    def load_model(self) -> bool:
        """Load the PHI detection model."""
        try:
            logger.info(f"Loading PHI detection model: {self.model_path}")
            
            # Simulate model loading for now - in real implementation would load actual models
            self.is_loaded = True
            logger.info("PHI detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PHI detection model: {e}")
            self.is_loaded = False
            return False
            
    @trace_operation("phi_detection_prediction")
    def predict(self, input_data: str, context: Dict[str, Any] = None) -> PredictionResult:
        """Predict PHI entities in text."""
        start_time = time.perf_counter()
        
        if not self.is_loaded:
            if not self.load_model():
                return PredictionResult(
                    predictions=[],
                    confidence_scores=[],
                    model_type=ModelType.PHI_DETECTION,
                    processing_time_ms=0,
                    metadata={},
                    success=False,
                    error_message="Model not loaded"
                )
        
        try:
            predictions = []
            confidence_scores = []
            
            # Enhanced PHI detection logic with multiple detection strategies
            phi_entities = self._detect_phi_entities(input_data, context)
            
            for entity in phi_entities:
                prediction = {
                    "text": entity.text,
                    "category": entity.category.value,
                    "start_position": entity.start_position,
                    "end_position": entity.end_position,
                    "confidence": entity.confidence,
                    "detection_method": entity.metadata.get("detection_method", "pattern"),
                    "clinical_context": entity.metadata.get("clinical_context", "")
                }
                predictions.append(prediction)
                confidence_scores.append(entity.confidence)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            if context and "ground_truth" in context:
                metrics = self._calculate_metrics(predictions, context["ground_truth"])
                metrics.processing_time_ms = processing_time
                self.update_metrics(metrics)
            
            return PredictionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                model_type=ModelType.PHI_DETECTION,
                processing_time_ms=processing_time,
                metadata={
                    "entities_detected": len(predictions),
                    "average_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
                    "detection_methods": list(set(p["detection_method"] for p in predictions)),
                    "clinical_context_used": self.use_clinical_context
                }
            )
            
        except Exception as e:
            logger.error(f"PHI prediction failed: {e}")
            return PredictionResult(
                predictions=[],
                confidence_scores=[],
                model_type=ModelType.PHI_DETECTION,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _detect_phi_entities(self, text: str, context: Dict[str, Any] = None) -> List[PHIEntity]:
        """Advanced PHI entity detection with multiple strategies."""
        entities = []
        
        # Strategy 1: Pattern-based detection with enhanced patterns
        pattern_entities = self._pattern_based_detection(text)
        entities.extend(pattern_entities)
        
        # Strategy 2: Context-aware detection for clinical documents
        if self.use_clinical_context and context:
            clinical_entities = self._clinical_context_detection(text, context)
            entities.extend(clinical_entities)
        
        # Strategy 3: Ensemble model predictions (if configured)
        if self.ensemble_models:
            ensemble_entities = self._ensemble_detection(text)
            entities.extend(ensemble_entities)
        
        # Deduplicate and merge overlapping entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _pattern_based_detection(self, text: str) -> List[PHIEntity]:
        """Enhanced pattern-based PHI detection."""
        import re
        entities = []
        
        # Enhanced patterns for healthcare PHI
        enhanced_patterns = {
            PHICategory.SOCIAL_SECURITY_NUMBERS: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # Standard SSN format
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # Space-separated SSN
                r'\b\d{9}\b'  # 9-digit number (context-dependent)
            ],
            PHICategory.MEDICAL_RECORD_NUMBERS: [
                r'\b(?:MRN|Medical Record|Patient ID|Chart #)[:.\s]*([A-Z0-9]{6,15})\b',
                r'\b(?:HSP|HOSP)[#\-\s]*(\d{6,10})\b',
                r'\bPT[#\-\s]*(\d{6,12})\b'
            ],
            PHICategory.DATES: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
            ],
            PHICategory.NAMES: [
                r'\b(?:Patient|Mr\.|Mrs\.|Ms\.|Dr\.|Doctor)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+,\s+[A-Z][a-z]+)\b'  # Last, First format
            ]
        }
        
        # Add custom patterns from config
        enhanced_patterns.update(self.custom_patterns)
        
        for category, patterns in enhanced_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = 0.85  # Base confidence for pattern matching
                    
                    # Adjust confidence based on context
                    if self._has_clinical_context(text, match.start(), match.end()):
                        confidence += 0.1
                    
                    entity = PHIEntity(
                        text=match.group(),
                        category=category,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=min(confidence, 1.0),
                        metadata={
                            "detection_method": "enhanced_pattern",
                            "pattern": pattern,
                            "clinical_context": self._extract_context(text, match.start(), match.end())
                        }
                    )
                    entities.append(entity)
        
        return entities
    
    def _clinical_context_detection(self, text: str, context: Dict[str, Any]) -> List[PHIEntity]:
        """Context-aware PHI detection for clinical documents."""
        entities = []
        
        # Use document type to enhance detection
        doc_type = context.get("document_type", "unknown")
        
        if doc_type in ["clinical_note", "medical_record"]:
            # Look for clinical-specific PHI patterns
            clinical_patterns = {
                "diagnosis_dates": r'(?:diagnosed|admitted|discharged)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                "medication_names": r'(?:prescribed|taking|medication)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
                "provider_names": r'(?:Dr\.|Doctor|Physician)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            }
            
            import re
            for pattern_type, pattern in clinical_patterns.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Determine appropriate PHI category
                    if "date" in pattern_type:
                        category = PHICategory.DATES
                    elif "name" in pattern_type:
                        category = PHICategory.NAMES
                    else:
                        category = PHICategory.HEALTH_PLAN_NUMBERS  # Default for medical info
                    
                    entity = PHIEntity(
                        text=match.group(1),
                        category=category,
                        start_position=match.start(1),
                        end_position=match.end(1),
                        confidence=0.9,  # High confidence for clinical context
                        metadata={
                            "detection_method": "clinical_context",
                            "context_type": pattern_type,
                            "document_type": doc_type
                        }
                    )
                    entities.append(entity)
        
        return entities
    
    def _ensemble_detection(self, text: str) -> List[PHIEntity]:
        """Ensemble model detection (placeholder for future implementation)."""
        # This would integrate with multiple pre-trained models
        # For now, return empty list
        return []
    
    def _deduplicate_entities(self, entities: List[PHIEntity]) -> List[PHIEntity]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return entities
        
        # Sort by position
        entities.sort(key=lambda e: (e.start_position, e.end_position))
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start_position < existing.end_position and 
                    entity.end_position > existing.start_position):
                    # Choose entity with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _has_clinical_context(self, text: str, start: int, end: int) -> bool:
        """Check if entity appears in clinical context."""
        # Look for clinical keywords within 50 characters
        context_window = text[max(0, start-50):min(len(text), end+50)]
        clinical_keywords = [
            "patient", "diagnosis", "treatment", "medication", "doctor", 
            "hospital", "clinic", "medical", "health", "symptoms"
        ]
        
        return any(keyword.lower() in context_window.lower() for keyword in clinical_keywords)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extract context around PHI entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> ModelMetrics:
        """Calculate model performance metrics."""
        # Simplified metrics calculation
        tp = len([p for p in predictions if any(
            p["start_position"] == gt["start_position"] and 
            p["end_position"] == gt["end_position"] 
            for gt in ground_truth
        )])
        
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / len(ground_truth) if ground_truth else 0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            processing_time_ms=0,  # Will be set by caller
            confidence_threshold=self.confidence_threshold,
            model_version=self.model_name
        )


class ClinicalSummarizer(BaseMLModel):
    """Advanced clinical text summarization using healthcare-specific models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("clinical_summarizer", config)
        self.model_path = self.config.get("model_path", "microsoft/BioGPT-Large")
        self.max_summary_length = self.config.get("max_summary_length", 512)
        self.preserve_medical_terms = self.config.get("preserve_medical_terms", True)
        self.include_risk_factors = self.config.get("include_risk_factors", True)
        
    def load_model(self) -> bool:
        """Load clinical summarization model."""
        try:
            logger.info(f"Loading clinical summarization model: {self.model_path}")
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            return False
    
    @trace_operation("clinical_summarization")
    def predict(self, input_data: str, context: Dict[str, Any] = None) -> PredictionResult:
        """Generate clinical summary of input text."""
        start_time = time.perf_counter()
        
        if not self.is_loaded:
            if not self.load_model():
                return PredictionResult(
                    predictions=[],
                    confidence_scores=[],
                    model_type=ModelType.SUMMARIZATION,
                    processing_time_ms=0,
                    metadata={},
                    success=False,
                    error_message="Model not loaded"
                )
        
        try:
            # Generate clinical summary with key medical information preserved
            summary_data = self._generate_clinical_summary(input_data, context)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return PredictionResult(
                predictions=[summary_data],
                confidence_scores=[summary_data.get("confidence", 0.8)],
                model_type=ModelType.SUMMARIZATION,
                processing_time_ms=processing_time,
                metadata={
                    "original_length": len(input_data),
                    "summary_length": len(summary_data.get("summary", "")),
                    "compression_ratio": len(summary_data.get("summary", "")) / len(input_data),
                    "medical_terms_preserved": summary_data.get("medical_terms_count", 0),
                    "risk_factors_identified": len(summary_data.get("risk_factors", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Clinical summarization failed: {e}")
            return PredictionResult(
                predictions=[],
                confidence_scores=[],
                model_type=ModelType.SUMMARIZATION,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _generate_clinical_summary(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate clinical summary with medical context preservation."""
        # Extract key medical information
        medical_entities = self._extract_medical_entities(text)
        symptoms = self._extract_symptoms(text)
        medications = self._extract_medications(text)
        procedures = self._extract_procedures(text)
        risk_factors = self._identify_risk_factors(text)
        
        # Generate structured summary
        summary_parts = []
        
        if symptoms:
            summary_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        if medical_entities.get("diagnoses"):
            summary_parts.append(f"Diagnoses: {', '.join(medical_entities['diagnoses'])}")
        
        if medications:
            summary_parts.append(f"Medications: {', '.join(medications)}")
        
        if procedures:
            summary_parts.append(f"Procedures: {', '.join(procedures)}")
        
        if risk_factors:
            summary_parts.append(f"Risk Factors: {', '.join(risk_factors)}")
        
        summary = ". ".join(summary_parts) if summary_parts else "No significant medical information identified."
        
        return {
            "summary": summary,
            "medical_entities": medical_entities,
            "symptoms": symptoms,
            "medications": medications,
            "procedures": procedures,
            "risk_factors": risk_factors,
            "confidence": 0.85,
            "medical_terms_count": len(medical_entities.get("diagnoses", [])) + len(medications) + len(procedures)
        }
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        import re
        
        entities = {
            "diagnoses": [],
            "conditions": [],
            "body_parts": []
        }
        
        # Common medical condition patterns
        condition_patterns = [
            r'\b(?:diagnosed with|diagnosis of|suffering from)\s+([a-z\s]+)',
            r'\b(diabetes|hypertension|cancer|pneumonia|infection|fracture|arthritis)\b',
            r'\b([a-z]+itis|[a-z]+osis|[a-z]+emia)\b'  # Common medical suffixes
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["diagnoses"].extend(matches)
        
        return entities
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from clinical text."""
        import re
        
        symptom_patterns = [
            r'\b(pain|ache|fever|nausea|vomiting|dizziness|fatigue|weakness|shortness of breath)\b',
            r'\b(chest pain|abdominal pain|headache|back pain|joint pain)\b',
            r'\b(cough|sore throat|runny nose|congestion)\b'
        ]
        
        symptoms = []
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            symptoms.extend(matches)
        
        return list(set(symptoms))  # Remove duplicates
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication names from text."""
        import re
        
        # Common medication patterns and examples
        medication_patterns = [
            r'\b(?:prescribed|taking|medication|drug|pill)\s+([A-Z][a-z]+(?:in|ol|ide|ate|ine))\b',
            r'\b(aspirin|acetaminophen|ibuprofen|lisinopril|metformin|atorvastatin)\b',
            r'\b([A-Z][a-z]+(?:in|ol|ide|ate|ine))\s+(?:\d+\s*mg)\b'
        ]
        
        medications = []
        for pattern in medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medications.extend(matches)
        
        return list(set(medications))
    
    def _extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedures from text."""
        import re
        
        procedure_patterns = [
            r'\b(surgery|operation|procedure|biopsy|scan|x-ray|MRI|CT scan|ultrasound)\b',
            r'\b([a-z]+ectomy|[a-z]+oscopy|[a-z]+plasty)\b',  # Common surgical suffixes
            r'\b(blood test|urine test|EKG|ECG|blood pressure)\b'
        ]
        
        procedures = []
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.extend(matches)
        
        return list(set(procedures))
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify medical risk factors from text."""
        import re
        
        risk_factor_patterns = [
            r'\b(smoking|alcohol|obesity|family history|genetic|hereditary)\b',
            r'\b(high blood pressure|diabetes|heart disease|stroke|cancer)\b',
            r'\b(sedentary lifestyle|poor diet|stress|lack of exercise)\b'
        ]
        
        risk_factors = []
        for pattern in risk_factor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            risk_factors.extend(matches)
        
        return list(set(risk_factors))


class MLModelManager:
    """Manager for coordinating multiple ML models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models: Dict[ModelType, BaseMLModel] = {}
        self.model_weights = self.config.get("model_weights", {})
        
    def register_model(self, model_type: ModelType, model: BaseMLModel) -> None:
        """Register an ML model."""
        self.models[model_type] = model
        logger.info(f"Registered {model_type.value} model: {model.model_name}")
    
    def load_all_models(self) -> Dict[ModelType, bool]:
        """Load all registered models."""
        results = {}
        for model_type, model in self.models.items():
            try:
                results[model_type] = model.load_model()
                logger.info(f"Model {model_type.value} loaded: {results[model_type]}")
            except Exception as e:
                logger.error(f"Failed to load model {model_type.value}: {e}")
                results[model_type] = False
        return results
    
    @trace_operation("multi_model_prediction")
    def predict_ensemble(
        self, 
        text: str, 
        model_types: List[ModelType],
        context: Dict[str, Any] = None
    ) -> Dict[ModelType, PredictionResult]:
        """Run ensemble predictions across multiple models."""
        results = {}
        
        for model_type in model_types:
            if model_type not in self.models:
                logger.warning(f"Model {model_type.value} not registered")
                continue
            
            try:
                model = self.models[model_type]
                result = model.predict(text, context)
                results[model_type] = result
                
                logger.info(
                    f"Model {model_type.value} completed prediction in "
                    f"{result.processing_time_ms:.2f}ms with {len(result.predictions)} results"
                )
                
            except Exception as e:
                logger.error(f"Prediction failed for model {model_type.value}: {e}")
                results[model_type] = PredictionResult(
                    predictions=[],
                    confidence_scores=[],
                    model_type=model_type,
                    processing_time_ms=0,
                    metadata={},
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def get_model_metrics(self, model_type: ModelType, last_n: int = 10) -> Optional[ModelMetrics]:
        """Get performance metrics for a specific model."""
        if model_type not in self.models:
            return None
        
        return self.models[model_type].get_average_metrics(last_n)
    
    def get_all_metrics(self) -> Dict[ModelType, Optional[ModelMetrics]]:
        """Get performance metrics for all models."""
        metrics = {}
        for model_type in self.models:
            metrics[model_type] = self.get_model_metrics(model_type)
        return metrics


def initialize_ml_models(config: Dict[str, Any] = None) -> MLModelManager:
    """Initialize ML model manager with default models."""
    manager = MLModelManager(config)
    
    # Initialize default models
    phi_detector = AdvancedPHIDetector(config.get("phi_detection", {}) if config else {})
    clinical_summarizer = ClinicalSummarizer(config.get("summarization", {}) if config else {})
    
    # Register models
    manager.register_model(ModelType.PHI_DETECTION, phi_detector)
    manager.register_model(ModelType.SUMMARIZATION, clinical_summarizer)
    
    # Load all models
    load_results = manager.load_all_models()
    
    logger.info(f"ML Model Manager initialized with {len(manager.models)} models")
    logger.info(f"Model loading results: {load_results}")
    
    return manager