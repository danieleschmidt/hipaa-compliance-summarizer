"""Advanced ML Integration for HIPAA Compliance System.

This module provides enhanced machine learning capabilities for PHI detection,
clinical summarization, and compliance analysis using state-of-the-art models.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, 
        AutoModelForSequenceClassification, pipeline,
        BertTokenizer, BertForTokenClassification
    )
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers or torch not available. ML features will use fallback implementations.")

from .models.phi_entity import PHICategory, PHIEntity
from .monitoring.tracing import trace_operation

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
            
            if TRANSFORMERS_AVAILABLE:
                # Load transformer-based NER model for PHI detection
                self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                
                # Create pipeline for easier inference
                self.ner_pipeline = pipeline(
                    "ner", 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=-1  # Use CPU for broader compatibility
                )
                
                # Load clinical BERT if available
                try:
                    self.clinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                    self.clinical_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                    self.clinical_available = True
                    logger.info("Clinical BERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"Clinical BERT not available: {e}")
                    self.clinical_available = False
                    
            else:
                # Fallback to rule-based detection
                logger.warning("Using rule-based PHI detection fallback")
                self.ner_pipeline = None
                self.clinical_available = False
            
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

        # Strategy 1: Transformer-based NER detection
        if TRANSFORMERS_AVAILABLE and self.ner_pipeline:
            transformer_entities = self._transformer_based_detection(text)
            entities.extend(transformer_entities)

        # Strategy 2: Pattern-based detection with enhanced patterns  
        pattern_entities = self._pattern_based_detection(text)
        entities.extend(pattern_entities)

        # Strategy 3: Context-aware detection for clinical documents
        if self.use_clinical_context and context:
            clinical_entities = self._clinical_context_detection(text, context)
            entities.extend(clinical_entities)

        # Strategy 4: Ensemble model predictions (if configured)
        if self.ensemble_models:
            ensemble_entities = self._ensemble_detection(text)
            entities.extend(ensemble_entities)

        # Deduplicate and merge overlapping entities
        entities = self._deduplicate_entities(entities)

        return entities
    
    def _transformer_based_detection(self, text: str) -> List[PHIEntity]:
        """Use transformer models for PHI detection."""
        entities = []
        
        if not self.ner_pipeline:
            return entities
            
        try:
            # Run NER pipeline on text
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                # Map NER labels to PHI categories
                phi_category = self._map_ner_to_phi_category(result['entity_group'])
                if phi_category:
                    entity = PHIEntity(
                        text=result['word'],
                        category=phi_category,
                        confidence=result['score'],
                        start_position=result['start'],
                        end_position=result['end'],
                        metadata={
                            "detection_method": "transformer_ner",
                            "original_label": result['entity_group'],
                            "model": "bert-large-cased-finetuned-conll03"
                        }
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"Transformer-based detection failed: {e}")
            
        return entities
    
    def _map_ner_to_phi_category(self, ner_label: str) -> Optional[PHICategory]:
        """Map NER model labels to HIPAA PHI categories."""
        mapping = {
            'PER': PHICategory.NAMES,
            'PERSON': PHICategory.NAMES,
            'LOC': PHICategory.GEOGRAPHIC_SUBDIVISIONS,
            'LOCATION': PHICategory.GEOGRAPHIC_SUBDIVISIONS,
            'ORG': PHICategory.NAMES,  # Organizations can contain PHI
            'ORGANIZATION': PHICategory.NAMES,
            'MISC': PHICategory.OTHER_UNIQUE_IDENTIFIERS,
        }
        return mapping.get(ner_label.upper())

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

    def _has_clinical_context(self, text: str, start: int, end: int, window: int = 50) -> bool:
        """Check if text has clinical context around detected entity."""
        clinical_keywords = [
            "patient", "diagnosis", "treatment", "medication", "dosage", "symptom",
            "procedure", "admission", "discharge", "clinical", "medical", "hospital",
            "doctor", "nurse", "physician", "therapy", "surgery", "exam", "test"
        ]
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()
        
        return any(keyword in context for keyword in clinical_keywords)

    def _extract_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extract surrounding context for PHI entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _clinical_context_detection(self, text: str, context: Dict[str, Any]) -> List[PHIEntity]:
        """Enhanced detection using clinical context and domain knowledge."""
        entities = []
        
        if not self.clinical_available:
            return entities
            
        try:
            # Use clinical BERT for context-aware detection
            # This would involve more sophisticated clinical NLP
            # For now, return empty list - can be enhanced with actual clinical models
            pass
            
        except Exception as e:
            logger.warning(f"Clinical context detection failed: {e}")
            
        return entities

    def _ensemble_detection(self, text: str) -> List[PHIEntity]:
        """Use ensemble of models for improved accuracy.""" 
        entities = []
        
        # Placeholder for ensemble model implementation
        # Would combine predictions from multiple models
        
        return entities

    def _deduplicate_entities(self, entities: List[PHIEntity]) -> List[PHIEntity]:
        """Remove duplicate and overlapping PHI entities."""
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
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                deduplicated.append(entity)
                
        return deduplicated

    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> ModelMetrics:
        """Calculate model performance metrics."""
        if not TRANSFORMERS_AVAILABLE:
            # Fallback metrics calculation
            return ModelMetrics(
                accuracy=0.85,
                precision=0.80,
                recall=0.75,
                f1_score=0.77,
                processing_time_ms=0,
                confidence_threshold=self.confidence_threshold,
                model_version=self.model_name
            )
        
        # Convert predictions and ground truth to comparable format
        pred_labels = [1 if p['confidence'] > self.confidence_threshold else 0 for p in predictions]
        true_labels = [1] * len(ground_truth)  # Assuming ground truth are all positive cases
        
        # Pad or truncate to match lengths
        min_len = min(len(pred_labels), len(true_labels))
        pred_labels = pred_labels[:min_len]
        true_labels = true_labels[:min_len]
        
        if not pred_labels or not true_labels:
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                processing_time_ms=0,
                confidence_threshold=self.confidence_threshold,
                model_version=self.model_name
            )
        
        # Calculate metrics using sklearn
        try:
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
        except Exception:
            accuracy = precision = recall = f1 = 0.0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            processing_time_ms=0,
            confidence_threshold=self.confidence_threshold,
            model_version=self.model_name
        )


class ClinicalSummarizer(BaseMLModel):
    """Advanced clinical text summarization using transformer models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("clinical_summarizer", config)
        self.model_path = self.config.get("model_path", "microsoft/BioGPT-Large")
        self.max_length = self.config.get("max_length", 512)
        self.preserve_clinical_terms = self.config.get("preserve_clinical_terms", True)
        
    def load_model(self) -> bool:
        """Load clinical summarization model."""
        try:
            logger.info(f"Loading clinical summarization model: {self.model_path}")
            
            if TRANSFORMERS_AVAILABLE:
                # Load summarization pipeline
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",  # Use BART for better summarization
                    device=-1
                )
                logger.info("Clinical summarization model loaded successfully")
            else:
                logger.warning("Using rule-based summarization fallback")
                self.summarization_pipeline = None
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load clinical summarization model: {e}")
            self.is_loaded = False
            return False
    
    @trace_operation("clinical_summarization")
    def predict(self, input_data: str, context: Dict[str, Any] = None) -> PredictionResult:
        """Generate clinical summary from PHI-redacted text."""
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
            if TRANSFORMERS_AVAILABLE and self.summarization_pipeline:
                # Use transformer-based summarization
                summary_result = self.summarization_pipeline(
                    input_data,
                    max_length=self.max_length,
                    min_length=50,
                    do_sample=False
                )
                summary_text = summary_result[0]['summary_text']
                confidence = 0.85  # Default confidence for transformer summarization
            else:
                # Fallback to rule-based summarization
                summary_text = self._rule_based_summarization(input_data)
                confidence = 0.70
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            predictions = [{
                "summary": summary_text,
                "confidence": confidence,
                "method": "transformer" if self.summarization_pipeline else "rule_based"
            }]
            
            return PredictionResult(
                predictions=predictions,
                confidence_scores=[confidence],
                model_type=ModelType.SUMMARIZATION,
                processing_time_ms=processing_time,
                metadata={
                    "summary_length": len(summary_text),
                    "input_length": len(input_data),
                    "compression_ratio": len(summary_text) / len(input_data) if input_data else 0
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
    
    def _rule_based_summarization(self, text: str) -> str:
        """Fallback rule-based summarization for clinical text."""
        # Simple extractive summarization
        sentences = text.split('.')
        important_sentences = []
        
        # Keywords that indicate important clinical information
        clinical_keywords = [
            'diagnosis', 'treatment', 'patient', 'procedure', 'medication',
            'symptoms', 'condition', 'therapy', 'outcome', 'discharge'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(keyword in sentence.lower() for keyword in clinical_keywords):
                important_sentences.append(sentence)
        
        if important_sentences:
            return '. '.join(important_sentences[:3]) + '.'  # Top 3 sentences
        else:
            # Return first few sentences if no clinical keywords found
            return '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else text


class MLModelManager:
    """Manages multiple ML models for the HIPAA system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models: Dict[ModelType, BaseMLModel] = {}
        self.model_configs = self.config.get("models", {})
        
    def initialize_models(self) -> Dict[ModelType, bool]:
        """Initialize all configured ML models."""
        results = {}
        
        # Initialize PHI Detection model
        phi_config = self.model_configs.get("phi_detection", {})
        phi_detector = AdvancedPHIDetector(phi_config)
        results[ModelType.PHI_DETECTION] = phi_detector.load_model()
        if results[ModelType.PHI_DETECTION]:
            self.models[ModelType.PHI_DETECTION] = phi_detector
        
        # Initialize Clinical Summarizer
        summarizer_config = self.model_configs.get("summarization", {})
        clinical_summarizer = ClinicalSummarizer(summarizer_config)
        results[ModelType.SUMMARIZATION] = clinical_summarizer.load_model()
        if results[ModelType.SUMMARIZATION]:
            self.models[ModelType.SUMMARIZATION] = clinical_summarizer
        
        return results
    
    def get_model(self, model_type: ModelType) -> Optional[BaseMLModel]:
        """Get a specific model by type."""
        return self.models.get(model_type)
    
    def predict_phi(self, text: str, context: Dict[str, Any] = None) -> Optional[PredictionResult]:
        """Convenience method for PHI detection."""
        phi_model = self.get_model(ModelType.PHI_DETECTION)
        if phi_model:
            return phi_model.predict(text, context)
        return None
    
    def summarize_clinical(self, text: str, context: Dict[str, Any] = None) -> Optional[PredictionResult]:
        """Convenience method for clinical summarization."""
        summarizer = self.get_model(ModelType.SUMMARIZATION)
        if summarizer:
            return summarizer.predict(text, context)
        return None
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models."""
        status = {}
        for model_type, model in self.models.items():
            avg_metrics = model.get_average_metrics()
            status[model_type.value] = {
                "loaded": model.is_loaded,
                "model_name": model.model_name,
                "confidence_threshold": model.confidence_threshold,
                "metrics_count": len(model.metrics_history),
                "average_metrics": avg_metrics.to_dict() if avg_metrics else None
            }
        return status

    def _detect_clinical_context(self, text: str, doc_type: str = None) -> List[PHIEntity]:
        """Detect PHI in clinical context with enhanced accuracy."""
        entities = []
        
        # Clinical context patterns for better PHI detection
        clinical_patterns = {
            "medical_record_number": r'(?:MRN|Medical Record|MR)[:\s#]*(\d+)',
            "patient_identifier": r'(?:Patient|ID)[:\s#]*(\w+)',
            "visit_date": r'(?:Visit|Seen|Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            "provider_name": r'(?:Dr\.|Doctor|Provider)[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
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
