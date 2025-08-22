"""Advanced PHI detection service with ML capabilities."""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models.audit_log import AuditAction, AuditEvent
from ..models.phi_entity import PHICategory, PHIEntity, RedactionMethod
from ..phi import PHIRedactor

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of PHI detection analysis."""

    entities: List[PHIEntity]
    confidence_scores: Dict[str, float]
    processing_time_ms: float
    detection_method: str
    model_version: Optional[str] = None


class PHIDetectionService:
    """Advanced PHI detection service with multiple detection strategies."""

    def __init__(self, enable_ml_models: bool = False):
        """Initialize PHI detection service.
        
        Args:
            enable_ml_models: Whether to enable ML-based detection (requires additional models)
        """
        self.redactor = PHIRedactor()
        self.enable_ml_models = enable_ml_models
        self.detection_stats = {
            "total_detections": 0,
            "by_category": {},
            "by_method": {},
        }

        # Initialize ML models if enabled
        if enable_ml_models:
            self._initialize_ml_models()

    def _initialize_ml_models(self):
        """Initialize ML models for advanced PHI detection."""
        # Placeholder for ML model initialization
        # In production, this would load BioBERT, ClinicalBERT, etc.
        logger.info("ML models would be initialized here for production deployment")
        self.ml_models = {
            "clinical_bert": None,  # Would load actual model
            "bio_bert": None,       # Would load actual model
            "custom_phi_model": None  # Would load custom trained model
        }

    def detect_phi_entities(self, text: str,
                           detection_method: str = "hybrid",
                           confidence_threshold: float = 0.8) -> DetectionResult:
        """Detect PHI entities using specified method.
        
        Args:
            text: Text content to analyze
            detection_method: Method to use (pattern, ml, hybrid)
            confidence_threshold: Minimum confidence score
            
        Returns:
            DetectionResult with detected entities and metadata
        """
        start_time = datetime.now()

        if detection_method == "pattern":
            entities = self._detect_with_patterns(text, confidence_threshold)
        elif detection_method == "ml" and self.enable_ml_models:
            entities = self._detect_with_ml(text, confidence_threshold)
        elif detection_method == "hybrid":
            entities = self._detect_hybrid(text, confidence_threshold)
        else:
            # Default to pattern-based detection
            entities = self._detect_with_patterns(text, confidence_threshold)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate confidence scores by category
        confidence_scores = self._calculate_category_confidence(entities)

        # Update statistics
        self._update_detection_stats(entities, detection_method)

        return DetectionResult(
            entities=entities,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time,
            detection_method=detection_method,
            model_version="1.0.0"
        )

    def _detect_with_patterns(self, text: str, confidence_threshold: float) -> List[PHIEntity]:
        """Detect PHI using pattern-based approach."""
        # Use existing PHI redactor for pattern detection
        basic_entities = self.redactor.detect(text)

        # Convert to PHIEntity objects with enhanced metadata
        phi_entities = []
        for i, entity in enumerate(basic_entities):
            # Map entity types to PHI categories
            category = self._map_entity_type_to_category(entity.type)

            # Calculate confidence based on pattern strength
            confidence = self._calculate_pattern_confidence(entity.type, entity.value)

            if confidence >= confidence_threshold:
                phi_entity = PHIEntity(
                    entity_id=f"phi_{hashlib.sha256(f'{entity.value}_{entity.start}_{entity.end}'.encode()).hexdigest()[:8]}",
                    category=category,
                    value=entity.value,
                    confidence_score=confidence,
                    start_position=entity.start,
                    end_position=entity.end,
                    detection_method="pattern_matching",
                    redaction_method=RedactionMethod.MASKING,
                    risk_level=self._assess_risk_level(category, confidence)
                )
                phi_entities.append(phi_entity)

        return phi_entities

    def _detect_with_ml(self, text: str, confidence_threshold: float) -> List[PHIEntity]:
        """Detect PHI using ML models (placeholder implementation)."""
        # Placeholder for ML-based detection
        # In production, this would use actual ML models
        logger.info("ML-based detection would run here with trained models")

        # For now, fall back to pattern detection
        return self._detect_with_patterns(text, confidence_threshold)

    def _detect_hybrid(self, text: str, confidence_threshold: float) -> List[PHIEntity]:
        """Detect PHI using hybrid approach combining patterns and ML."""
        # Start with pattern-based detection
        pattern_entities = self._detect_with_patterns(text, confidence_threshold * 0.8)

        # If ML models are available, enhance with ML detection
        if self.enable_ml_models:
            ml_entities = self._detect_with_ml(text, confidence_threshold * 0.9)

            # Merge and deduplicate entities
            all_entities = pattern_entities + ml_entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Boost confidence for entities detected by multiple methods
            for entity in unique_entities:
                if self._detected_by_multiple_methods(entity, pattern_entities, ml_entities):
                    entity.confidence_score = min(1.0, entity.confidence_score * 1.2)

            return unique_entities
        else:
            return pattern_entities

    def _map_entity_type_to_category(self, entity_type: str) -> PHICategory:
        """Map entity type string to PHI category enum."""
        type_mapping = {
            "name": PHICategory.NAMES,
            "names": PHICategory.NAMES,
            "person": PHICategory.NAMES,
            "date": PHICategory.DATES,
            "dates": PHICategory.DATES,
            "phone": PHICategory.TELEPHONE_NUMBERS,
            "telephone": PHICategory.TELEPHONE_NUMBERS,
            "phone_number": PHICategory.TELEPHONE_NUMBERS,
            "ssn": PHICategory.SOCIAL_SECURITY_NUMBERS,
            "social_security": PHICategory.SOCIAL_SECURITY_NUMBERS,
            "email": PHICategory.EMAIL_ADDRESSES,
            "email_address": PHICategory.EMAIL_ADDRESSES,
            "mrn": PHICategory.MEDICAL_RECORD_NUMBERS,
            "medical_record": PHICategory.MEDICAL_RECORD_NUMBERS,
            "account": PHICategory.ACCOUNT_NUMBERS,
            "account_number": PHICategory.ACCOUNT_NUMBERS,
            "address": PHICategory.GEOGRAPHIC_SUBDIVISIONS,
            "location": PHICategory.GEOGRAPHIC_SUBDIVISIONS,
            "url": PHICategory.WEB_URLS,
            "ip": PHICategory.IP_ADDRESSES,
            "ip_address": PHICategory.IP_ADDRESSES,
        }

        # Try to match the entity type
        entity_type_lower = entity_type.lower()
        return type_mapping.get(entity_type_lower, PHICategory.OTHER_UNIQUE_IDENTIFYING_NUMBERS)

    def _calculate_pattern_confidence(self, entity_type: str, value: str) -> float:
        """Calculate confidence score for pattern-based detection."""
        # Base confidence scores by pattern type
        base_confidence = {
            "ssn": 0.95,
            "phone": 0.90,
            "email": 0.95,
            "date": 0.85,
            "name": 0.75,
            "mrn": 0.90,
            "account": 0.80,
        }

        confidence = base_confidence.get(entity_type.lower(), 0.70)

        # Adjust based on value characteristics
        if len(value) < 3:
            confidence *= 0.7  # Very short values are less reliable
        elif re.match(r'^\d+$', value):
            confidence *= 1.1  # Numeric patterns are often more reliable
        elif re.match(r'^[A-Z][a-z]+$', value):
            confidence *= 0.9  # Capitalized words (names) are slightly less certain

        return min(1.0, confidence)

    def _assess_risk_level(self, category: PHICategory, confidence: float) -> str:
        """Assess risk level based on PHI category and confidence."""
        high_risk_categories = {
            PHICategory.SOCIAL_SECURITY_NUMBERS,
            PHICategory.MEDICAL_RECORD_NUMBERS,
            PHICategory.BIOMETRIC_IDENTIFIERS,
            PHICategory.FULL_FACE_PHOTOS
        }

        medium_risk_categories = {
            PHICategory.NAMES,
            PHICategory.TELEPHONE_NUMBERS,
            PHICategory.EMAIL_ADDRESSES,
            PHICategory.ACCOUNT_NUMBERS
        }

        if category in high_risk_categories or confidence >= 0.95:
            return "high"
        elif category in medium_risk_categories or confidence >= 0.80:
            return "medium"
        else:
            return "low"

    def _calculate_category_confidence(self, entities: List[PHIEntity]) -> Dict[str, float]:
        """Calculate average confidence scores by PHI category."""
        category_scores = {}
        category_counts = {}

        for entity in entities:
            category = entity.category.value
            if category not in category_scores:
                category_scores[category] = 0.0
                category_counts[category] = 0

            category_scores[category] += entity.confidence_score
            category_counts[category] += 1

        # Calculate averages
        return {
            category: score / category_counts[category]
            for category, score in category_scores.items()
        }

    def _deduplicate_entities(self, entities: List[PHIEntity]) -> List[PHIEntity]:
        """Remove duplicate entities based on position and value."""
        unique_entities = []
        seen_positions = set()

        # Sort by start position for consistent processing
        sorted_entities = sorted(entities, key=lambda e: e.start_position)

        for entity in sorted_entities:
            position_key = (entity.start_position, entity.end_position, entity.value)
            if position_key not in seen_positions:
                unique_entities.append(entity)
                seen_positions.add(position_key)

        return unique_entities

    def _detected_by_multiple_methods(self, entity: PHIEntity,
                                     pattern_entities: List[PHIEntity],
                                     ml_entities: List[PHIEntity]) -> bool:
        """Check if an entity was detected by multiple methods."""
        # This is a simplified check - in production would be more sophisticated
        pattern_detected = any(
            e.start_position == entity.start_position and e.end_position == entity.end_position
            for e in pattern_entities
        )
        ml_detected = any(
            e.start_position == entity.start_position and e.end_position == entity.end_position
            for e in ml_entities
        )

        return pattern_detected and ml_detected

    def _update_detection_stats(self, entities: List[PHIEntity], method: str):
        """Update internal detection statistics."""
        self.detection_stats["total_detections"] += len(entities)

        if method not in self.detection_stats["by_method"]:
            self.detection_stats["by_method"][method] = 0
        self.detection_stats["by_method"][method] += len(entities)

        for entity in entities:
            category = entity.category.value
            if category not in self.detection_stats["by_category"]:
                self.detection_stats["by_category"][category] = 0
            self.detection_stats["by_category"][category] += 1

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get current detection statistics."""
        return self.detection_stats.copy()

    def validate_phi_entity(self, entity: PHIEntity) -> Tuple[bool, List[str]]:
        """Validate a PHI entity for compliance and accuracy.
        
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []

        # Check confidence threshold
        if entity.confidence_score < 0.5:
            errors.append("Confidence score too low for reliable detection")

        # Check position validity
        if entity.start_position >= entity.end_position:
            errors.append("Invalid position range")

        # Check category-specific validation
        if entity.category == PHICategory.SOCIAL_SECURITY_NUMBERS:
            if not re.match(r'^\d{3}-?\d{2}-?\d{4}$', entity.value):
                errors.append("Invalid SSN format")

        elif entity.category == PHICategory.TELEPHONE_NUMBERS:
            if not re.match(r'^[\+]?[\d\-\(\)\s\.]{10,15}$', entity.value):
                errors.append("Invalid phone number format")

        elif entity.category == PHICategory.EMAIL_ADDRESSES:
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', entity.value):
                errors.append("Invalid email format")

        return len(errors) == 0, errors

    def create_audit_event(self, entities: List[PHIEntity], document_id: str = None) -> AuditEvent:
        """Create audit event for PHI detection."""
        return AuditEvent(
            action=AuditAction.PHI_DETECTED,
            description=f"Detected {len(entities)} PHI entities",
            resource_type="document",
            resource_id=document_id,
            details={
                "entity_count": len(entities),
                "categories_detected": list(set(e.category.value for e in entities)),
                "detection_method": "phi_detection_service",
                "total_confidence": sum(e.confidence_score for e in entities) / len(entities) if entities else 0
            },
            compliance_relevant=True,
            security_level="sensitive"
        )
