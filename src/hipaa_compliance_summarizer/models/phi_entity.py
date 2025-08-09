"""PHI Entity models for HIPAA compliance tracking."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class PHICategory(str, Enum):
    """HIPAA Safe Harbor PHI categories."""

    # Direct identifiers (18 categories per HIPAA Safe Harbor rule)
    NAMES = "names"
    GEOGRAPHIC_SUBDIVISIONS = "geographic_subdivisions"
    DATES = "dates"
    TELEPHONE_NUMBERS = "telephone_numbers"
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"
    DEVICE_IDENTIFIERS = "device_identifiers"
    WEB_URLS = "web_urls"
    IP_ADDRESSES = "ip_addresses"
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"
    FULL_FACE_PHOTOS = "full_face_photos"
    ACCOUNT_NUMBERS = "account_numbers"
    CERTIFICATE_NUMBERS = "certificate_numbers"
    SOCIAL_SECURITY_NUMBERS = "social_security_numbers"
    MEDICAL_RECORD_NUMBERS = "medical_record_numbers"
    HEALTH_PLAN_NUMBERS = "health_plan_numbers"
    FAX_NUMBERS = "fax_numbers"
    EMAIL_ADDRESSES = "email_addresses"
    OTHER_UNIQUE_IDENTIFYING_NUMBERS = "other_unique_identifying_numbers"


class RedactionMethod(str, Enum):
    """Methods used for PHI redaction."""

    REMOVAL = "removal"          # Complete removal of PHI
    MASKING = "masking"          # Replace with asterisks or placeholders
    SYNTHETIC = "synthetic"      # Replace with synthetic but realistic data
    TOKENIZATION = "tokenization"  # Replace with reversible tokens
    ENCRYPTION = "encryption"    # Encrypt in place with key management


@dataclass
class PHIEntity:
    """Represents a detected PHI entity with full metadata."""

    # Core identification
    entity_id: str
    category: PHICategory
    value: str
    confidence_score: float

    # Location within document
    start_position: int
    end_position: int
    line_number: Optional[int] = None

    # Processing metadata
    detection_method: str = "pattern_matching"  # pattern_matching, ml_model, manual
    redaction_method: RedactionMethod = RedactionMethod.MASKING
    redacted_value: Optional[str] = None

    # Compliance tracking
    risk_level: str = "medium"  # low, medium, high, critical
    requires_audit: bool = True
    compliance_notes: Optional[str] = None

    # Timestamps
    detected_at: datetime = None
    processed_at: Optional[datetime] = None

    def __post_init__(self):
        """Set default timestamps and validate data."""
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()

        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")

        # Validate positions
        if self.start_position < 0 or self.end_position < 0:
            raise ValueError("Positions must be non-negative")
        if self.start_position >= self.end_position:
            raise ValueError("Start position must be less than end position")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entity_id": self.entity_id,
            "category": self.category.value,
            "value": self.value,
            "confidence_score": self.confidence_score,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "line_number": self.line_number,
            "detection_method": self.detection_method,
            "redaction_method": self.redaction_method.value,
            "redacted_value": self.redacted_value,
            "risk_level": self.risk_level,
            "requires_audit": self.requires_audit,
            "compliance_notes": self.compliance_notes,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PHIEntity":
        """Create PHIEntity from dictionary."""
        # Convert string enums back to enum instances
        category = PHICategory(data["category"])
        redaction_method = RedactionMethod(data["redaction_method"])

        # Parse timestamps
        detected_at = datetime.fromisoformat(data["detected_at"]) if data.get("detected_at") else None
        processed_at = datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None

        return cls(
            entity_id=data["entity_id"],
            category=category,
            value=data["value"],
            confidence_score=data["confidence_score"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            line_number=data.get("line_number"),
            detection_method=data.get("detection_method", "pattern_matching"),
            redaction_method=redaction_method,
            redacted_value=data.get("redacted_value"),
            risk_level=data.get("risk_level", "medium"),
            requires_audit=data.get("requires_audit", True),
            compliance_notes=data.get("compliance_notes"),
            detected_at=detected_at,
            processed_at=processed_at,
        )

    def calculate_risk_score(self) -> float:
        """Calculate numerical risk score based on category and confidence."""
        # Base risk scores by category
        category_risk = {
            PHICategory.SOCIAL_SECURITY_NUMBERS: 1.0,
            PHICategory.MEDICAL_RECORD_NUMBERS: 0.9,
            PHICategory.ACCOUNT_NUMBERS: 0.8,
            PHICategory.FULL_FACE_PHOTOS: 0.9,
            PHICategory.BIOMETRIC_IDENTIFIERS: 1.0,
            PHICategory.NAMES: 0.7,
            PHICategory.DATES: 0.5,
            PHICategory.TELEPHONE_NUMBERS: 0.6,
            PHICategory.EMAIL_ADDRESSES: 0.6,
            PHICategory.IP_ADDRESSES: 0.4,
            PHICategory.WEB_URLS: 0.3,
            PHICategory.GEOGRAPHIC_SUBDIVISIONS: 0.4,
        }

        base_risk = category_risk.get(self.category, 0.5)  # Default medium risk

        # Adjust by confidence score - higher confidence = higher risk
        confidence_multiplier = 0.5 + (self.confidence_score * 0.5)

        return min(1.0, base_risk * confidence_multiplier)

    def needs_special_handling(self) -> bool:
        """Determine if this entity requires special compliance handling."""
        high_risk_categories = {
            PHICategory.SOCIAL_SECURITY_NUMBERS,
            PHICategory.MEDICAL_RECORD_NUMBERS,
            PHICategory.BIOMETRIC_IDENTIFIERS,
            PHICategory.FULL_FACE_PHOTOS,
        }

        return (
            self.category in high_risk_categories or
            self.confidence_score >= 0.95 or
            self.risk_level in ["high", "critical"]
        )
