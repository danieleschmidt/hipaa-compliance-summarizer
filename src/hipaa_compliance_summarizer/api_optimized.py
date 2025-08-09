"""Optimized API implementation for HIPAA compliance system."""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .batch import BatchProcessor
from .ml_integration_lite import get_ml_processor
from .performance_enhanced import get_performance_optimizer, performance_monitor
from .processor import HIPAAProcessor, ProcessingResult
from .reporting import ComplianceReporter
from .security_enhanced import get_security_manager

logger = logging.getLogger(__name__)
security = HTTPBearer()


# Pydantic models for API
class DocumentRequest(BaseModel):
    content: str = Field(..., description="Document content to process")
    compliance_level: str = Field(default="standard", description="Compliance level")
    generate_summary: bool = Field(default=True, description="Generate summary")


class BatchRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document contents")
    compliance_level: str = Field(default="standard", description="Compliance level")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


class ProcessingResponse(BaseModel):
    document_id: str
    summary: str
    compliance_score: float
    phi_detected_count: int
    processing_time_ms: float
    timestamp: datetime


class BatchResponse(BaseModel):
    batch_id: str
    total_documents: int
    processed_documents: int
    average_compliance_score: float
    total_phi_detected: int
    processing_time_ms: float
    individual_results: List[ProcessingResponse]


class ComplianceReportResponse(BaseModel):
    report_id: str
    period: str
    documents_processed: int
    average_compliance_score: float
    violations_detected: int
    recommendations: List[str]
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    system_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]


# Global application state
class AppState:
    def __init__(self):
        self.hipaa_processor = HIPAAProcessor()
        self.batch_processor = BatchProcessor()
        self.compliance_reporter = ComplianceReporter()
        self.security_manager = get_security_manager()
        self.performance_optimizer = get_performance_optimizer()
        self.ml_processor = get_ml_processor()
        self.start_time = time.time()


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting HIPAA Compliance API...")
    app_state.security_manager.log_security_event("api_startup", "LOW")

    yield

    # Shutdown
    logger.info("Shutting down HIPAA Compliance API...")
    app_state.security_manager.log_security_event("api_shutdown", "LOW")
    app_state.performance_optimizer.shutdown()


app = FastAPI(
    title="HIPAA Compliance API",
    description="Advanced HIPAA-compliant document processing API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Security dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate user credentials."""
    token = credentials.credentials

    # Validate session token
    if not app_state.security_manager.validate_session_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return {"user_id": "api_user"}  # In production, extract from token


# Middleware for request logging and rate limiting
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security and logging middleware."""
    start_time = time.time()
    client_ip = request.client.host

    # Track access attempt
    success = True
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            success = False
    except Exception as e:
        success = False
        logger.error(f"Request processing error: {e}")
        response = Response("Internal server error", status_code=500)

    # Log security event
    processing_time = (time.time() - start_time) * 1000
    app_state.security_manager.track_access_attempt(client_ip, success)
    app_state.security_manager.log_security_event(
        "api_request",
        "LOW" if success else "MEDIUM",
        source_ip=client_ip,
        method=request.method,
        url=str(request.url),
        status_code=getattr(response, 'status_code', 500),
        processing_time_ms=processing_time
    )

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_state.start_time

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        system_metrics={
            "active_processors": 1,
            "cache_size": len(app_state.performance_optimizer.cache._cache),
            "metrics_collected": len(app_state.performance_optimizer.metrics_history)
        },
        cache_stats=app_state.performance_optimizer.cache.get_stats()
    )


@app.post("/process", response_model=ProcessingResponse)
@performance_monitor("api_process_document")
async def process_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Process a single document for PHI detection and redaction."""

    with app_state.security_manager.security_context(
        "document_processing",
        current_user.get("user_id")
    ):
        start_time = time.time()

        try:
            # Process document
            result = app_state.hipaa_processor.process_document(request.content)
            processing_time = (time.time() - start_time) * 1000

            # Generate unique document ID
            import uuid
            document_id = str(uuid.uuid4())

            response = ProcessingResponse(
                document_id=document_id,
                summary=result.summary,
                compliance_score=result.compliance_score,
                phi_detected_count=result.phi_detected_count,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )

            # Log successful processing
            background_tasks.add_task(
                app_state.security_manager.log_security_event,
                "document_processed",
                "LOW",
                user_id=current_user.get("user_id"),
                document_id=document_id,
                compliance_score=result.compliance_score,
                phi_detected=result.phi_detected_count
            )

            return response

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(e)}"
            )


@app.post("/batch-process", response_model=BatchResponse)
@performance_monitor("api_batch_process")
async def batch_process_documents(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Process multiple documents in batch."""

    with app_state.security_manager.security_context(
        "batch_processing",
        current_user.get("user_id")
    ):
        start_time = time.time()

        try:
            # Generate batch ID
            import uuid
            batch_id = str(uuid.uuid4())

            # Process documents
            individual_results = []
            total_phi = 0
            total_compliance_score = 0

            if request.parallel_processing:
                # Use performance optimizer for parallel processing
                def process_single_doc(content: str) -> ProcessingResult:
                    return app_state.hipaa_processor.process_document(content)

                # Execute in parallel
                futures = app_state.performance_optimizer.execute_parallel(
                    [lambda: process_single_doc(doc) for doc in request.documents],
                    use_processes=False
                )

                # Collect results
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        doc_response = ProcessingResponse(
                            document_id=f"{batch_id}_{i}",
                            summary=result.summary,
                            compliance_score=result.compliance_score,
                            phi_detected_count=result.phi_detected_count,
                            processing_time_ms=0,  # Individual timing not tracked in batch
                            timestamp=datetime.utcnow()
                        )
                        individual_results.append(doc_response)
                        total_phi += result.phi_detected_count
                        total_compliance_score += result.compliance_score

                    except Exception as e:
                        logger.error(f"Error processing document {i}: {e}")
                        # Continue with other documents
            else:
                # Sequential processing
                for i, content in enumerate(request.documents):
                    try:
                        result = app_state.hipaa_processor.process_document(content)
                        doc_response = ProcessingResponse(
                            document_id=f"{batch_id}_{i}",
                            summary=result.summary,
                            compliance_score=result.compliance_score,
                            phi_detected_count=result.phi_detected_count,
                            processing_time_ms=0,
                            timestamp=datetime.utcnow()
                        )
                        individual_results.append(doc_response)
                        total_phi += result.phi_detected_count
                        total_compliance_score += result.compliance_score

                    except Exception as e:
                        logger.error(f"Error processing document {i}: {e}")
                        continue

            processing_time = (time.time() - start_time) * 1000
            processed_count = len(individual_results)
            avg_compliance = total_compliance_score / max(processed_count, 1)

            response = BatchResponse(
                batch_id=batch_id,
                total_documents=len(request.documents),
                processed_documents=processed_count,
                average_compliance_score=avg_compliance,
                total_phi_detected=total_phi,
                processing_time_ms=processing_time,
                individual_results=individual_results
            )

            # Log batch processing
            background_tasks.add_task(
                app_state.security_manager.log_security_event,
                "batch_processed",
                "LOW",
                user_id=current_user.get("user_id"),
                batch_id=batch_id,
                documents_processed=processed_count,
                avg_compliance_score=avg_compliance
            )

            return response

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch processing failed: {str(e)}"
            )


@app.get("/compliance-report", response_model=ComplianceReportResponse)
@performance_monitor("api_compliance_report")
async def generate_compliance_report(
    period: str = "2024-Q1",
    documents_processed: int = 0,
    include_recommendations: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Generate compliance report."""

    with app_state.security_manager.security_context(
        "compliance_reporting",
        current_user.get("user_id")
    ):
        try:
            # Generate report
            report = app_state.compliance_reporter.generate_report(
                period=period,
                documents_processed=documents_processed,
                include_recommendations=include_recommendations
            )

            import uuid
            report_id = str(uuid.uuid4())

            return ComplianceReportResponse(
                report_id=report_id,
                period=period,
                documents_processed=documents_processed,
                average_compliance_score=report.overall_compliance,
                violations_detected=report.violations_detected,
                recommendations=report.recommendations or [],
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Compliance report error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {str(e)}"
            )


@app.get("/security-dashboard")
async def get_security_dashboard(current_user: dict = Depends(get_current_user)):
    """Get security dashboard data."""
    return app_state.security_manager.get_security_dashboard()


@app.get("/performance-report")
async def get_performance_report(
    hours: int = 24,
    current_user: dict = Depends(get_current_user)
):
    """Get performance report."""
    return app_state.performance_optimizer.get_performance_report(hours)


@app.post("/auth/login")
async def login(username: str, password: str):
    """Login endpoint (simplified for demo)."""
    # In production, implement proper authentication
    if username == "admin" and password == "secure_password":
        token = app_state.security_manager.generate_session_token(username)
        return {"access_token": token, "token_type": "bearer"}

    raise HTTPException(
        status_code=401,
        detail="Invalid credentials"
    )


def run_api(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the API server."""
    config = uvicorn.Config(
        "hipaa_compliance_summarizer.api_optimized:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    run_api()
