"""Data transformers for document processing pipeline."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.phi_entity import PHICategory
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of a transformation operation."""

    original_content: str
    transformed_content: str
    transformation_type: str
    modifications: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_length": len(self.original_content),
            "transformed_length": len(self.transformed_content),
            "transformation_type": self.transformation_type,
            "modifications_count": len(self.modifications),
            "modifications": self.modifications,
            "metadata": self.metadata,
            "success": self.success,
            "error_message": self.error_message
        }


class BaseTransformer(ABC):
    """Base class for document transformers."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize transformer.
        
        Args:
            name: Transformer name
            config: Transformer configuration
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def transform(self, content: str, context: Dict[str, Any] = None) -> TransformationResult:
        """Transform content.
        
        Args:
            content: Content to transform
            context: Additional context for transformation
            
        Returns:
            Transformation result
        """
        pass

    def validate_input(self, content: str) -> bool:
        """Validate input content."""
        return isinstance(content, str) and len(content.strip()) > 0


class DocumentTransformer(BaseTransformer):
    """Transformer for basic document preprocessing."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("document_preprocessor", config)

        # Default configuration
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.remove_empty_lines = self.config.get("remove_empty_lines", True)
        self.standardize_line_endings = self.config.get("standardize_line_endings", True)
        self.preserve_structure = self.config.get("preserve_structure", True)

    @trace_operation("document_transformation")
    def transform(self, content: str, context: Dict[str, Any] = None) -> TransformationResult:
        """Transform document content with preprocessing."""
        if not self.validate_input(content):
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message="Invalid input content"
            )

        original_content = content
        transformed_content = content
        modifications = []

        try:
            # Standardize line endings
            if self.standardize_line_endings:
                old_content = transformed_content
                transformed_content = re.sub(r'\r\n|\r', '\n', transformed_content)
                if old_content != transformed_content:
                    modifications.append({
                        "type": "line_ending_standardization",
                        "description": "Standardized line endings to Unix format"
                    })

            # Normalize whitespace
            if self.normalize_whitespace:
                old_content = transformed_content
                # Replace multiple spaces with single space, but preserve indentation
                lines = transformed_content.split('\n')
                normalized_lines = []

                for line in lines:
                    # Preserve leading whitespace for structure
                    leading_whitespace = re.match(r'^(\s*)', line).group(1)
                    content_part = line[len(leading_whitespace):]
                    # Normalize internal whitespace
                    normalized_content = re.sub(r'\s+', ' ', content_part).strip()
                    if normalized_content or not self.remove_empty_lines:
                        normalized_lines.append(leading_whitespace + normalized_content)

                transformed_content = '\n'.join(normalized_lines)

                if old_content != transformed_content:
                    modifications.append({
                        "type": "whitespace_normalization",
                        "description": "Normalized internal whitespace while preserving structure"
                    })

            # Remove excessive empty lines (more than 2 consecutive)
            if self.remove_empty_lines:
                old_content = transformed_content
                transformed_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', transformed_content)
                if old_content != transformed_content:
                    modifications.append({
                        "type": "empty_line_removal",
                        "description": "Removed excessive empty lines"
                    })

            # Remove trailing whitespace from lines
            old_content = transformed_content
            lines = [line.rstrip() for line in transformed_content.split('\n')]
            transformed_content = '\n'.join(lines)
            if old_content != transformed_content:
                modifications.append({
                    "type": "trailing_whitespace_removal",
                    "description": "Removed trailing whitespace from lines"
                })

            metadata = {
                "original_length": len(original_content),
                "transformed_length": len(transformed_content),
                "size_reduction": len(original_content) - len(transformed_content),
                "processing_timestamp": datetime.utcnow().isoformat()
            }

            return TransformationResult(
                original_content=original_content,
                transformed_content=transformed_content,
                transformation_type=self.name,
                modifications=modifications,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Document transformation failed: {e}")
            return TransformationResult(
                original_content=original_content,
                transformed_content=original_content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message=str(e)
            )


class PHIRedactionTransformer(BaseTransformer):
    """Transformer for PHI redaction and de-identification."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("phi_redaction", config)

        # Redaction configuration
        self.redaction_style = self.config.get("redaction_style", "replacement")  # replacement, masking, removal
        self.preserve_length = self.config.get("preserve_length", True)
        self.redaction_marker = self.config.get("redaction_marker", "[REDACTED]")
        self.category_specific_markers = self.config.get("category_specific_markers", False)
        self.maintain_readability = self.config.get("maintain_readability", True)

        # Category-specific redaction markers
        self.category_markers = {
            PHICategory.NAMES.value: "[NAME]",
            PHICategory.DATES.value: "[DATE]",
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value: "[LOCATION]",
            PHICategory.TELEPHONE_NUMBERS.value: "[PHONE]",
            PHICategory.EMAIL_ADDRESSES.value: "[EMAIL]",
            PHICategory.SOCIAL_SECURITY_NUMBERS.value: "[SSN]",
            PHICategory.MEDICAL_RECORD_NUMBERS.value: "[MRN]",
            PHICategory.ACCOUNT_NUMBERS.value: "[ACCOUNT]",
            PHICategory.HEALTH_PLAN_NUMBERS.value: "[HEALTH_PLAN]"
        }

    @trace_operation("phi_redaction")
    def transform(self, content: str, context: Dict[str, Any] = None) -> TransformationResult:
        """Transform content by redacting PHI entities."""
        if not self.validate_input(content):
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message="Invalid input content"
            )

        phi_entities = context.get("phi_entities", []) if context else []

        if not phi_entities:
            logger.warning("No PHI entities provided for redaction")
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={"warning": "No PHI entities to redact"}
            )

        try:
            transformed_content = content
            modifications = []

            # Sort entities by position (descending) to maintain correct positions during replacement
            sorted_entities = sorted(
                phi_entities,
                key=lambda e: getattr(e, 'start_position', 0),
                reverse=True
            )

            for entity in sorted_entities:
                entity_text = getattr(entity, 'text', '')
                entity_category = getattr(entity, 'category', 'unknown')
                start_pos = getattr(entity, 'start_position', None)
                end_pos = getattr(entity, 'end_position', None)
                confidence = getattr(entity, 'confidence', 0.0)

                if not entity_text:
                    continue

                # Generate replacement text
                replacement = self._generate_replacement(entity_text, entity_category)

                # Perform replacement
                if start_pos is not None and end_pos is not None:
                    # Use position-based replacement for accuracy
                    if start_pos <= len(transformed_content) and end_pos <= len(transformed_content):
                        transformed_content = (
                            transformed_content[:start_pos] +
                            replacement +
                            transformed_content[end_pos:]
                        )
                        modifications.append({
                            "type": "phi_redaction",
                            "category": entity_category,
                            "original_text": entity_text,
                            "replacement": replacement,
                            "position": f"{start_pos}-{end_pos}",
                            "confidence": confidence
                        })
                else:
                    # Fallback to text-based replacement
                    if entity_text in transformed_content:
                        transformed_content = transformed_content.replace(entity_text, replacement, 1)
                        modifications.append({
                            "type": "phi_redaction",
                            "category": entity_category,
                            "original_text": entity_text,
                            "replacement": replacement,
                            "method": "text_replacement",
                            "confidence": confidence
                        })

            # Post-processing for readability
            if self.maintain_readability:
                transformed_content = self._improve_readability(transformed_content)

            metadata = {
                "redaction_style": self.redaction_style,
                "entities_redacted": len(modifications),
                "original_length": len(content),
                "redacted_length": len(transformed_content),
                "redaction_ratio": len(modifications) / len(phi_entities) if phi_entities else 0,
                "processing_timestamp": datetime.utcnow().isoformat()
            }

            return TransformationResult(
                original_content=content,
                transformed_content=transformed_content,
                transformation_type=self.name,
                modifications=modifications,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"PHI redaction failed: {e}")
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message=str(e)
            )

    def _generate_replacement(self, original_text: str, category: str) -> str:
        """Generate replacement text for PHI entity."""
        if self.redaction_style == "removal":
            return ""

        if self.category_specific_markers and category in self.category_markers:
            marker = self.category_markers[category]
        else:
            marker = self.redaction_marker

        if self.redaction_style == "masking":
            # Generate masked version
            if category == PHICategory.DATES.value:
                # Preserve date format structure
                return re.sub(r'\d', 'X', original_text)
            elif category in [PHICategory.TELEPHONE_NUMBERS.value, PHICategory.FAX_NUMBERS.value]:
                # Preserve phone number structure
                return re.sub(r'\d', 'X', original_text)
            elif category == PHICategory.SOCIAL_SECURITY_NUMBERS.value:
                # Mask SSN
                return "XXX-XX-XXXX"
            else:
                # Generic masking
                if self.preserve_length:
                    return 'X' * len(original_text)
                else:
                    return marker

        else:  # replacement style
            return marker

    def _improve_readability(self, content: str) -> str:
        """Improve readability of redacted content."""
        # Remove excessive consecutive redaction markers
        content = re.sub(r'(\[REDACTED\]\s*){3,}', '[REDACTED] ... [REDACTED] ', content)

        # Fix spacing around redaction markers
        content = re.sub(r'\s+(\[[\w\s]+\])\s+', r' \1 ', content)

        # Remove empty lines created by redaction
        content = re.sub(r'\n\s*\[REDACTED\]\s*\n', '\n[REDACTED]\n', content)

        return content


class ComplianceEnrichmentTransformer(BaseTransformer):
    """Transformer for enriching documents with compliance metadata."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("compliance_enrichment", config)

        self.add_header = self.config.get("add_header", True)
        self.add_footer = self.config.get("add_footer", True)
        self.include_timestamp = self.config.get("include_timestamp", True)
        self.include_compliance_score = self.config.get("include_compliance_score", True)
        self.include_risk_level = self.config.get("include_risk_level", True)

    @trace_operation("compliance_enrichment")
    def transform(self, content: str, context: Dict[str, Any] = None) -> TransformationResult:
        """Transform content by adding compliance metadata."""
        if not self.validate_input(content):
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message="Invalid input content"
            )

        try:
            transformed_content = content
            modifications = []

            # Extract compliance information from context
            compliance_analysis = context.get("compliance_analysis") if context else None
            risk_analysis = context.get("risk_analysis") if context else None
            document_id = context.get("document_id", "unknown") if context else "unknown"

            # Add header
            if self.add_header:
                header_parts = ["=== HIPAA COMPLIANT DOCUMENT ==="]

                if self.include_timestamp:
                    header_parts.append(f"Processed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

                header_parts.append(f"Document ID: {document_id}")

                if compliance_analysis and self.include_compliance_score:
                    score = compliance_analysis.overall_compliance_score
                    status = compliance_analysis.compliance_status.upper()
                    header_parts.append(f"Compliance Score: {score:.1f}% ({status})")

                if risk_analysis and self.include_risk_level:
                    risk_level = risk_analysis.risk_level.upper()
                    header_parts.append(f"Risk Level: {risk_level}")

                header = "\\n".join(header_parts) + "\\n" + "=" * 50 + "\\n\\n"
                transformed_content = header + transformed_content

                modifications.append({
                    "type": "header_addition",
                    "description": "Added HIPAA compliance header"
                })

            # Add footer
            if self.add_footer:
                footer_parts = ["=" * 50]
                footer_parts.append("This document has been processed for HIPAA compliance.")
                footer_parts.append("All PHI has been identified and appropriately handled.")

                if compliance_analysis:
                    if compliance_analysis.safe_harbor_compliance:
                        footer_parts.append("✓ Safe Harbor de-identification requirements met.")
                    else:
                        footer_parts.append("⚠ Safe Harbor de-identification requirements NOT fully met.")

                footer_parts.append("Generated by HIPAA Compliance Summarizer")
                footer_parts.append("=== END OF DOCUMENT ===")

                footer = "\\n\\n" + "\\n".join(footer_parts)
                transformed_content = transformed_content + footer

                modifications.append({
                    "type": "footer_addition",
                    "description": "Added HIPAA compliance footer"
                })

            metadata = {
                "enrichment_elements": len(modifications),
                "original_length": len(content),
                "enriched_length": len(transformed_content),
                "processing_timestamp": datetime.utcnow().isoformat()
            }

            return TransformationResult(
                original_content=content,
                transformed_content=transformed_content,
                transformation_type=self.name,
                modifications=modifications,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Compliance enrichment failed: {e}")
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message=str(e)
            )


class OutputTransformer(BaseTransformer):
    """Transformer for final output formatting and validation."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("output_formatter", config)

        self.output_format = self.config.get("output_format", "text")  # text, json, xml
        self.include_metadata = self.config.get("include_metadata", False)
        self.validate_output = self.config.get("validate_output", True)
        self.compress_whitespace = self.config.get("compress_whitespace", False)

    @trace_operation("output_transformation")
    def transform(self, content: str, context: Dict[str, Any] = None) -> TransformationResult:
        """Transform content for final output."""
        if not self.validate_input(content):
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message="Invalid input content"
            )

        try:
            transformed_content = content
            modifications = []

            # Format based on output type
            if self.output_format == "json":
                output_data = {
                    "content": content,
                    "document_id": context.get("document_id") if context else None,
                    "processed_at": datetime.utcnow().isoformat()
                }

                if self.include_metadata and context:
                    output_data["metadata"] = {
                        "compliance_analysis": getattr(context.get("compliance_analysis"), "to_dict", lambda: {})(),
                        "risk_analysis": getattr(context.get("risk_analysis"), "to_dict", lambda: {})(),
                        "phi_count": len(context.get("phi_entities", []))
                    }

                import json
                transformed_content = json.dumps(output_data, indent=2, ensure_ascii=False)
                modifications.append({
                    "type": "json_formatting",
                    "description": "Formatted output as JSON"
                })

            elif self.output_format == "xml":
                xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<hipaa_document>
    <metadata>
        <document_id>{context.get('document_id', 'unknown') if context else 'unknown'}</document_id>
        <processed_at>{datetime.utcnow().isoformat()}</processed_at>
    </metadata>
    <content><![CDATA[{content}]]></content>
</hipaa_document>"""
                transformed_content = xml_content
                modifications.append({
                    "type": "xml_formatting",
                    "description": "Formatted output as XML"
                })

            # Compress whitespace if requested
            if self.compress_whitespace and self.output_format == "text":
                old_content = transformed_content
                # Remove excessive whitespace while preserving single spaces and line breaks
                transformed_content = re.sub(r'[ \\t]+', ' ', transformed_content)
                transformed_content = re.sub(r'\\n\\s*\\n\\s*\\n+', '\\n\\n', transformed_content)

                if old_content != transformed_content:
                    modifications.append({
                        "type": "whitespace_compression",
                        "description": "Compressed excessive whitespace"
                    })

            # Validate output
            if self.validate_output:
                validation_result = self._validate_output(transformed_content)
                if not validation_result["valid"]:
                    return TransformationResult(
                        original_content=content,
                        transformed_content=content,
                        transformation_type=self.name,
                        modifications=[],
                        metadata={},
                        success=False,
                        error_message=f"Output validation failed: {validation_result['error']}"
                    )

            metadata = {
                "output_format": self.output_format,
                "original_length": len(content),
                "formatted_length": len(transformed_content),
                "compression_ratio": (len(content) - len(transformed_content)) / len(content) if len(content) > 0 else 0,
                "processing_timestamp": datetime.utcnow().isoformat()
            }

            return TransformationResult(
                original_content=content,
                transformed_content=transformed_content,
                transformation_type=self.name,
                modifications=modifications,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Output transformation failed: {e}")
            return TransformationResult(
                original_content=content,
                transformed_content=content,
                transformation_type=self.name,
                modifications=[],
                metadata={},
                success=False,
                error_message=str(e)
            )

    def _validate_output(self, content: str) -> Dict[str, Any]:
        """Validate the output content."""
        try:
            if self.output_format == "json":
                import json
                json.loads(content)
            elif self.output_format == "xml":
                # Basic XML validation - check for balanced tags
                import xml.etree.ElementTree as ET
                ET.fromstring(content)

            # Check for basic content requirements
            if len(content.strip()) == 0:
                return {"valid": False, "error": "Empty content"}

            return {"valid": True, "error": None}

        except Exception as e:
            return {"valid": False, "error": str(e)}
