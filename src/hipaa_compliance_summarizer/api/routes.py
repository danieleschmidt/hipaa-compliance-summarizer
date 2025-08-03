"""API routes for HIPAA compliance processing."""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import hashlib

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..processor import HIPAAProcessor, ProcessingResult, ComplianceLevel
from ..batch import BatchProcessor, BatchDashboard
from ..services.phi_detection_service import PHIDetectionService
from ..database import get_db_connection
from ..database.repositories import DocumentRepository, PHIDetectionRepository, AuditRepository
from ..models.audit_log import AuditEvent, AuditAction
from .schemas import (
    DocumentUploadResponse, 
    DocumentProcessRequest, 
    DocumentProcessResponse,
    PHIDetectionResponse,
    ComplianceReportResponse,
    BatchProcessRequest,
    BatchProcessResponse,
)

logger = logging.getLogger(__name__)

# Create API router
api_blueprint = APIRouter(
    prefix="",
    tags=["HIPAA Compliance API"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)


# Dependency injection
def get_document_repository() -> DocumentRepository:
    """Get document repository instance."""
    return DocumentRepository()


def get_phi_repository() -> PHIDetectionRepository:
    """Get PHI detection repository instance."""
    return PHIDetectionRepository()


def get_audit_repository() -> AuditRepository:
    """Get audit repository instance."""
    return AuditRepository()


def get_hipaa_processor() -> HIPAAProcessor:
    """Get HIPAA processor instance."""
    return HIPAAProcessor()


def get_phi_detection_service() -> PHIDetectionService:
    """Get PHI detection service instance."""
    return PHIDetectionService(enable_ml_models=True)


# Document Processing Endpoints
@api_blueprint.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a healthcare document",
    description="Upload a healthcare document for PHI detection and compliance processing."
)
async def upload_document(
    file: UploadFile = File(..., description="Healthcare document file"),
    document_type: str = Query("unknown", description="Type of document (clinical_note, lab_report, etc.)"),
    compliance_level: str = Query("standard", description="Compliance processing level"),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    audit_repo: AuditRepository = Depends(get_audit_repository),
) -> DocumentUploadResponse:
    """Upload and register a healthcare document."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Calculate file hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Check for duplicate
        existing_doc = doc_repo.get_by_hash(file_hash)
        if existing_doc:
            return DocumentUploadResponse(
                document_id=existing_doc.id,
                filename=existing_doc.filename,
                status="duplicate",
                message="Document already exists"
            )
        
        # Create document record
        from ..database.models import Document
        document = Document(
            id=str(uuid.uuid4()),
            filename=file.filename,
            file_path=f"/uploads/{file.filename}",  # Would be actual storage path
            file_size=len(content),
            file_hash=file_hash,
            document_type=document_type,
            content_type=file.content_type or "application/octet-stream",
            original_content=content.decode('utf-8', errors='ignore'),
            uploaded_by="api_user",  # Would come from auth context
            created_at=datetime.utcnow(),
        )
        
        # Save to database
        doc_repo.create(document)
        
        # Create audit record
        from ..database.models import AuditRecord
        audit_record = AuditRecord(
            id=str(uuid.uuid4()),
            event_type="document_upload",
            event_action="CREATE",
            event_description=f"Document uploaded: {file.filename}",
            resource_type="document",
            resource_id=document.id,
            user_id="api_user",
            created_at=datetime.utcnow(),
        )
        audit_repo.create(audit_record)
        
        logger.info(f"Document uploaded successfully: {document.id}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=file.filename,
            status="uploaded",
            message="Document uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail="Document upload failed")


@api_blueprint.post(
    "/documents/{document_id}/process",
    response_model=DocumentProcessResponse,
    summary="Process document for PHI detection",
    description="Process an uploaded document for PHI detection, redaction, and compliance analysis."
)
async def process_document(
    document_id: str,
    request: DocumentProcessRequest,
    doc_repo: DocumentRepository = Depends(get_document_repository),
    phi_repo: PHIDetectionRepository = Depends(get_phi_repository),
    processor: HIPAAProcessor = Depends(get_hipaa_processor),
    phi_service: PHIDetectionService = Depends(get_phi_detection_service),
) -> DocumentProcessResponse:
    """Process a document for PHI detection and compliance analysis."""
    try:
        # Get document
        document = doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document.processing_status == "completed":
            raise HTTPException(status_code=409, detail="Document already processed")
        
        # Update status to processing
        doc_repo.update_processing_status(document_id, "processing", "api_processor")
        
        # Process document
        processor.compliance_level = ComplianceLevel(request.compliance_level)
        result = processor.process_document(document.original_content)
        
        # Enhanced PHI detection
        phi_result = phi_service.detect_phi_entities(
            document.original_content,
            detection_method=request.detection_method,
            confidence_threshold=request.confidence_threshold
        )
        
        # Save PHI detections
        from ..database.models import PHIDetection
        for entity in phi_result.entities:
            detection = PHIDetection(
                id=str(uuid.uuid4()),
                document_id=document_id,
                entity_type=entity.category.value,
                entity_category=entity.category.value,
                entity_value=entity.value,  # Would be encrypted in production
                redacted_value=f"[{entity.category.value.upper()}]",
                start_position=entity.start_position,
                end_position=entity.end_position,
                detection_method=entity.detection_method,
                confidence_score=entity.confidence_score,
                risk_level=entity.risk_level,
                requires_manual_review=entity.needs_special_handling(),
                created_at=datetime.utcnow(),
            )
            phi_repo.create(detection)
        
        # Update document with results
        doc_repo.update_compliance_metrics(
            document_id=document_id,
            compliance_score=result.compliance_score,
            phi_count=len(phi_result.entities),
            risk_level=max([e.risk_level for e in phi_result.entities], default="low", 
                          key=lambda x: ["low", "medium", "high", "critical"].index(x)),
            redacted_content=result.redacted.text,
            summary=result.summary
        )
        
        # Update status to completed
        doc_repo.update_processing_status(document_id, "completed")
        
        logger.info(f"Document processed successfully: {document_id}")
        
        return DocumentProcessResponse(
            document_id=document_id,
            processing_status="completed",
            compliance_score=result.compliance_score,
            phi_entities_detected=len(phi_result.entities),
            risk_level=max([e.risk_level for e in phi_result.entities], default="low",
                          key=lambda x: ["low", "medium", "high", "critical"].index(x)),
            summary=result.summary,
            processing_time_ms=phi_result.processing_time_ms,
            confidence_scores=phi_result.confidence_scores,
        )
        
    except HTTPException:
        # Update status to failed
        doc_repo.update_processing_status(document_id, "failed")
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        doc_repo.update_processing_status(document_id, "failed")
        raise HTTPException(status_code=500, detail="Document processing failed")


@api_blueprint.get(
    "/documents/{document_id}",
    response_model=DocumentProcessResponse,
    summary="Get document processing results",
    description="Retrieve the processing results and compliance analysis for a document."
)
async def get_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(get_document_repository),
    phi_repo: PHIDetectionRepository = Depends(get_phi_repository),
) -> DocumentProcessResponse:
    """Get document processing results."""
    try:
        # Get document
        document = doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get PHI detections
        phi_detections = phi_repo.get_by_document_id(document_id)
        
        # Calculate confidence scores by category
        confidence_scores = {}
        if phi_detections:
            category_scores = {}
            category_counts = {}
            
            for detection in phi_detections:
                category = detection.entity_category
                if category not in category_scores:
                    category_scores[category] = 0.0
                    category_counts[category] = 0
                
                category_scores[category] += detection.confidence_score
                category_counts[category] += 1
            
            confidence_scores = {
                category: score / category_counts[category]
                for category, score in category_scores.items()
            }
        
        return DocumentProcessResponse(
            document_id=document.id,
            processing_status=document.processing_status,
            compliance_score=document.compliance_score,
            phi_entities_detected=len(phi_detections),
            risk_level=document.risk_level,
            summary=document.summary,
            processing_time_ms=0.0,  # Not stored
            confidence_scores=confidence_scores,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


# PHI Detection Endpoints
@api_blueprint.get(
    "/documents/{document_id}/phi-detections",
    response_model=List[PHIDetectionResponse],
    summary="Get PHI detections for document",
    description="Retrieve all PHI entities detected in a specific document."
)
async def get_phi_detections(
    document_id: str,
    phi_repo: PHIDetectionRepository = Depends(get_phi_repository),
) -> List[PHIDetectionResponse]:
    """Get PHI detections for a document."""
    try:
        detections = phi_repo.get_by_document_id(document_id)
        
        return [
            PHIDetectionResponse(
                detection_id=d.id,
                document_id=d.document_id,
                entity_type=d.entity_type,
                entity_category=d.entity_category,
                redacted_value=d.redacted_value,
                start_position=d.start_position,
                end_position=d.end_position,
                confidence_score=d.confidence_score,
                risk_level=d.risk_level,
                detection_method=d.detection_method,
                requires_manual_review=d.requires_manual_review,
            )
            for d in detections
        ]
        
    except Exception as e:
        logger.error(f"Failed to get PHI detections: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PHI detections")


# Batch Processing Endpoints
@api_blueprint.post(
    "/batch/process",
    response_model=BatchProcessResponse,
    summary="Start batch processing",
    description="Start batch processing of multiple documents for PHI detection and compliance analysis."
)
async def start_batch_process(
    request: BatchProcessRequest,
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> BatchProcessResponse:
    """Start batch processing of documents."""
    try:
        # Get documents to process
        documents = []
        for doc_id in request.document_ids:
            doc = doc_repo.get_by_id(doc_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        # Create batch processor
        batch_processor = BatchProcessor()
        
        # Process documents (simplified for API - would be async in production)
        results = []
        for doc in documents:
            try:
                processor = HIPAAProcessor(compliance_level=ComplianceLevel(request.compliance_level))
                result = processor.process_document(doc.original_content)
                results.append(result)
                
                # Update document status
                doc_repo.update_compliance_metrics(
                    doc.id,
                    result.compliance_score,
                    result.phi_detected_count,
                    "medium",  # Simplified
                    result.redacted.text,
                    result.summary
                )
                doc_repo.update_processing_status(doc.id, "completed", "batch_processor")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc.id}: {e}")
                doc_repo.update_processing_status(doc.id, "failed")
        
        # Generate dashboard
        successful_results = [r for r in results if hasattr(r, 'compliance_score')]
        dashboard = BatchDashboard(
            documents_processed=len(successful_results),
            avg_compliance_score=sum(r.compliance_score for r in successful_results) / len(successful_results) if successful_results else 0.0,
            total_phi_detected=sum(r.phi_detected_count for r in successful_results),
        )
        
        batch_id = str(uuid.uuid4())
        logger.info(f"Batch processing completed: {batch_id}")
        
        return BatchProcessResponse(
            batch_id=batch_id,
            status="completed",
            documents_processed=len(successful_results),
            documents_failed=len(documents) - len(successful_results),
            avg_compliance_score=dashboard.avg_compliance_score,
            total_phi_detected=dashboard.total_phi_detected,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")


# Compliance Reporting Endpoints
@api_blueprint.get(
    "/compliance/report",
    response_model=ComplianceReportResponse,
    summary="Generate compliance report",
    description="Generate a compliance report for a specified time period."
)
async def generate_compliance_report(
    period_start: datetime = Query(..., description="Report period start date"),
    period_end: datetime = Query(..., description="Report period end date"),
    report_type: str = Query("summary", description="Type of report (summary, detailed, audit)"),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    phi_repo: PHIDetectionRepository = Depends(get_phi_repository),
) -> ComplianceReportResponse:
    """Generate a compliance report."""
    try:
        # Get processing summary
        processing_summary = doc_repo.get_processing_summary()
        
        # Get PHI statistics
        phi_stats = phi_repo.get_statistics()
        
        # Calculate compliance metrics
        total_documents = processing_summary.get("total_documents", 0)
        completed_documents = processing_summary.get("completed", 0)
        avg_compliance_score = processing_summary.get("avg_compliance_score", 0.0)
        total_phi_entities = processing_summary.get("total_phi_entities", 0)
        high_risk_count = processing_summary.get("high_risk_count", 0)
        
        # Generate report
        report_id = str(uuid.uuid4())
        
        return ComplianceReportResponse(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            total_documents=total_documents,
            completed_documents=completed_documents,
            failed_documents=processing_summary.get("failed", 0),
            avg_compliance_score=avg_compliance_score,
            total_phi_detected=total_phi_entities,
            high_risk_documents=high_risk_count,
            compliance_violations=0,  # Would calculate from actual violations
            generated_at=datetime.utcnow(),
        )
        
    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")


# System Information Endpoints
@api_blueprint.get(
    "/system/stats",
    summary="Get system statistics",
    description="Get system processing statistics and health metrics."
)
async def get_system_stats(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    phi_repo: PHIDetectionRepository = Depends(get_phi_repository),
):
    """Get system statistics."""
    try:
        processing_summary = doc_repo.get_processing_summary()
        phi_stats = phi_repo.get_statistics()
        
        return {
            "documents": processing_summary,
            "phi_detection": phi_stats,
            "system": {
                "version": "1.2.0",
                "uptime": "placeholder",
                "last_updated": datetime.utcnow().isoformat(),
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")