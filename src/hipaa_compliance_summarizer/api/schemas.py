"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# Base schemas
class BaseResponse(BaseModel):
    """Base response model with common fields."""

    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# Document schemas
class DocumentUploadResponse(BaseResponse):
    """Response for document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status (uploaded, duplicate, failed)")
    message: str = Field(..., description="Status message")


class DocumentProcessRequest(BaseModel):
    """Request for document processing."""

    compliance_level: str = Field(
        default="standard",
        description="Compliance processing level",
        regex="^(strict|standard|minimal)$"
    )
    detection_method: str = Field(
        default="hybrid",
        description="PHI detection method",
        regex="^(pattern|ml|hybrid)$"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for PHI detection"
    )
    generate_summary: bool = Field(
        default=True,
        description="Whether to generate document summary"
    )
    include_redacted_content: bool = Field(
        default=False,
        description="Whether to include redacted content in response"
    )


class DocumentProcessResponse(BaseResponse):
    """Response for document processing."""

    document_id: str = Field(..., description="Document identifier")
    processing_status: str = Field(..., description="Processing status")
    compliance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall compliance score (0.0-1.0)"
    )
    phi_entities_detected: int = Field(
        ...,
        ge=0,
        description="Number of PHI entities detected"
    )
    risk_level: str = Field(..., description="Overall risk level assessment")
    summary: Optional[str] = Field(None, description="Document summary")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores by PHI category"
    )
    redacted_content: Optional[str] = Field(
        None,
        description="Redacted document content (if requested)"
    )


# PHI Detection schemas
class PHIDetectionResponse(BaseModel):
    """Response for PHI detection details."""

    detection_id: str = Field(..., description="Detection identifier")
    document_id: str = Field(..., description="Associated document ID")
    entity_type: str = Field(..., description="Type of PHI entity")
    entity_category: str = Field(..., description="HIPAA PHI category")
    redacted_value: str = Field(..., description="Redacted replacement value")
    start_position: int = Field(..., ge=0, description="Start position in text")
    end_position: int = Field(..., ge=0, description="End position in text")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score"
    )
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    detection_method: str = Field(..., description="Detection method used")
    requires_manual_review: bool = Field(..., description="Whether manual review is required")

    @validator('end_position')
    def end_greater_than_start(cls, v, values):
        """Validate that end position is greater than start position."""
        if 'start_position' in values and v <= values['start_position']:
            raise ValueError('end_position must be greater than start_position')
        return v


# Batch Processing schemas
class BatchProcessRequest(BaseModel):
    """Request for batch processing."""

    document_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of document IDs to process"
    )
    compliance_level: str = Field(
        default="standard",
        description="Compliance processing level",
        regex="^(strict|standard|minimal)$"
    )
    generate_summaries: bool = Field(
        default=True,
        description="Whether to generate document summaries"
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show processing progress"
    )


class BatchProcessResponse(BaseResponse):
    """Response for batch processing."""

    batch_id: str = Field(..., description="Batch processing identifier")
    status: str = Field(..., description="Batch processing status")
    documents_processed: int = Field(
        ...,
        ge=0,
        description="Number of documents successfully processed"
    )
    documents_failed: int = Field(
        ...,
        ge=0,
        description="Number of documents that failed processing"
    )
    avg_compliance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average compliance score across all documents"
    )
    total_phi_detected: int = Field(
        ...,
        ge=0,
        description="Total PHI entities detected across all documents"
    )
    processing_time_seconds: Optional[float] = Field(
        None,
        description="Total processing time in seconds"
    )
    failed_document_ids: Optional[List[str]] = Field(
        None,
        description="List of document IDs that failed processing"
    )


# Compliance Reporting schemas
class ComplianceMetrics(BaseModel):
    """Compliance metrics model."""

    total_documents: int = Field(..., ge=0, description="Total documents processed")
    completed_documents: int = Field(..., ge=0, description="Successfully completed documents")
    failed_documents: int = Field(..., ge=0, description="Failed document processing")
    avg_compliance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average compliance score"
    )
    total_phi_detected: int = Field(..., ge=0, description="Total PHI entities detected")
    high_risk_documents: int = Field(..., ge=0, description="High-risk documents count")
    compliance_violations: int = Field(..., ge=0, description="Compliance violations detected")


class ComplianceReportResponse(BaseResponse):
    """Response for compliance report generation."""

    report_id: str = Field(..., description="Report identifier")
    report_type: str = Field(..., description="Type of report generated")
    period_start: datetime = Field(..., description="Report period start date")
    period_end: datetime = Field(..., description="Report period end date")
    generated_at: datetime = Field(..., description="Report generation timestamp")

    # Metrics
    total_documents: int = Field(..., ge=0, description="Total documents in period")
    completed_documents: int = Field(..., ge=0, description="Completed documents")
    failed_documents: int = Field(..., ge=0, description="Failed documents")
    avg_compliance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average compliance score"
    )
    total_phi_detected: int = Field(..., ge=0, description="Total PHI detected")
    high_risk_documents: int = Field(..., ge=0, description="High-risk documents")
    compliance_violations: int = Field(..., ge=0, description="Compliance violations")

    # Additional details
    phi_breakdown: Optional[Dict[str, int]] = Field(
        None,
        description="Breakdown of PHI types detected"
    )
    risk_breakdown: Optional[Dict[str, int]] = Field(
        None,
        description="Breakdown by risk level"
    )
    recommendations: Optional[List[str]] = Field(
        None,
        description="Compliance improvement recommendations"
    )

    @validator('period_end')
    def end_after_start(cls, v, values):
        """Validate that end date is after start date."""
        if 'period_start' in values and v <= values['period_start']:
            raise ValueError('period_end must be after period_start')
        return v


# System schemas
class SystemHealthResponse(BaseModel):
    """System health status response."""

    status: str = Field(..., description="Overall system health status")
    database: str = Field(..., description="Database connection status")
    version: str = Field(..., description="System version")
    uptime: Optional[str] = Field(None, description="System uptime")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")


class SystemStatsResponse(BaseModel):
    """System statistics response."""

    documents: ComplianceMetrics = Field(..., description="Document processing metrics")
    phi_detection: Dict[str, Any] = Field(..., description="PHI detection statistics")
    system: Dict[str, Any] = Field(..., description="System information")


# Error schemas
class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response model."""

    error_code: str = "VALIDATION_ERROR"
    validation_errors: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed validation error information"
    )


# Audit schemas
class AuditEventResponse(BaseModel):
    """Audit event response model."""

    event_id: str = Field(..., description="Event identifier")
    event_type: str = Field(..., description="Type of auditable event")
    event_action: str = Field(..., description="Action performed")
    event_description: str = Field(..., description="Event description")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    timestamp: datetime = Field(..., description="Event timestamp")
    success: bool = Field(..., description="Whether the action was successful")
    security_level: str = Field(..., description="Security classification")
    compliance_relevant: bool = Field(..., description="Whether event is compliance-relevant")


# Document metadata schemas
class DocumentMetadata(BaseModel):
    """Document metadata model."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    document_type: str = Field(..., description="Type of healthcare document")
    content_type: str = Field(..., description="MIME content type")
    upload_date: datetime = Field(..., description="Upload timestamp")
    processing_status: str = Field(..., description="Current processing status")
    compliance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Compliance score if processed"
    )
    phi_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of PHI entities if processed"
    )
    risk_level: Optional[str] = Field(None, description="Risk level if processed")


class DocumentListResponse(BaseResponse):
    """Response for document listing."""

    documents: List[DocumentMetadata] = Field(..., description="List of documents")
    total_count: int = Field(..., ge=0, description="Total number of documents")
    page_size: int = Field(..., ge=1, description="Number of documents per page")
    page_number: int = Field(..., ge=1, description="Current page number")
    has_more: bool = Field(..., description="Whether there are more documents")


# Configuration schemas
class ConfigurationResponse(BaseModel):
    """System configuration response."""

    phi_detection_threshold: float = Field(..., description="PHI detection confidence threshold")
    default_compliance_level: str = Field(..., description="Default compliance level")
    max_file_size_mb: int = Field(..., description="Maximum file size in MB")
    supported_file_types: List[str] = Field(..., description="Supported file types")
    retention_period_days: int = Field(..., description="Data retention period in days")
    rate_limits: Dict[str, int] = Field(..., description="API rate limits")


# Schema registry for OpenAPI documentation
DocumentSchema = DocumentMetadata
PHIDetectionSchema = PHIDetectionResponse
ComplianceReportSchema = ComplianceReportResponse
