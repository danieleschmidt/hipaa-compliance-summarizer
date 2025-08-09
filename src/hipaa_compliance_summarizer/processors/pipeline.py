"""Main processing pipeline for HIPAA compliance workflow."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..monitoring.metrics import MetricsCollector
from ..monitoring.tracing import get_tracer, trace_operation

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for processing pipeline."""

    enable_parallel_processing: bool = True
    max_concurrent_stages: int = 3
    timeout_seconds: int = 300
    retry_failed_stages: bool = True
    max_retries: int = 2
    enable_checkpointing: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = True
    stage_specific_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    pipeline_id: str
    status: PipelineStatus
    input_document: Dict[str, Any]
    processed_document: Optional[Dict[str, Any]] = None
    stage_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "input_document": self.input_document,
            "processed_document": self.processed_document,
            "stage_results": self.stage_results,
            "errors": self.errors,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


class PipelineStage:
    """Base class for pipeline processing stages."""

    def __init__(self, name: str, required_inputs: List[str] = None,
                 produces_outputs: List[str] = None,
                 config: Dict[str, Any] = None):
        """Initialize pipeline stage.
        
        Args:
            name: Stage name
            required_inputs: List of required input keys
            produces_outputs: List of output keys this stage produces
            config: Stage-specific configuration
        """
        self.name = name
        self.required_inputs = required_inputs or []
        self.produces_outputs = produces_outputs or []
        self.config = config or {}
        self.metrics_collector = None
        self.tracer = None

    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for the stage."""
        self.metrics_collector = collector

    def set_tracer(self, tracer):
        """Set tracer for the stage."""
        self.tracer = tracer

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate that required inputs are present."""
        missing_inputs = [inp for inp in self.required_inputs if inp not in context]
        if missing_inputs:
            logger.error(f"Stage {self.name} missing required inputs: {missing_inputs}")
            return False
        return True

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stage processing.
        
        Args:
            context: Processing context containing inputs and intermediate results
            
        Returns:
            Updated context with stage outputs
        """
        if not self.validate_inputs(context):
            raise ValueError(f"Stage {self.name} validation failed")

        start_time = datetime.utcnow()

        try:
            # Execute stage with tracing if available
            if self.tracer:
                with self.tracer.trace(f"pipeline_stage_{self.name}") as span:
                    if span:
                        span.add_tag("stage.name", self.name)
                        span.add_tag("stage.inputs", str(self.required_inputs))

                    result = await self.process(context)
            else:
                result = await self.process(context)

            # Record metrics if available
            if self.metrics_collector:
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.metrics_collector.record_histogram(
                    "pipeline_stage_duration_ms",
                    processing_time,
                    {"stage": self.name}
                )
                self.metrics_collector.record_counter(
                    "pipeline_stage_executions_total",
                    1.0,
                    {"stage": self.name, "status": "success"}
                )

            return result

        except Exception as e:
            # Record error metrics
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "pipeline_stage_executions_total",
                    1.0,
                    {"stage": self.name, "status": "error"}
                )

            logger.error(f"Stage {self.name} failed: {e}")
            raise

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method to implement stage-specific processing."""
        raise NotImplementedError(f"Stage {self.name} must implement process method")


class DocumentIngestionStage(PipelineStage):
    """Stage for document ingestion and preprocessing."""

    def __init__(self):
        super().__init__(
            name="document_ingestion",
            required_inputs=["document"],
            produces_outputs=["content", "metadata", "document_id"]
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process document ingestion."""
        document = context["document"]

        # Extract content based on document type
        if isinstance(document, dict):
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            document_id = document.get("id", str(uuid.uuid4()))
        elif isinstance(document, str):
            content = document
            metadata = {}
            document_id = str(uuid.uuid4())
        else:
            raise ValueError("Invalid document format")

        # Basic preprocessing
        content = content.strip()
        if not content:
            raise ValueError("Document content is empty")

        # Update context
        context.update({
            "content": content,
            "metadata": metadata,
            "document_id": document_id,
            "word_count": len(content.split()),
            "character_count": len(content)
        })

        logger.info(f"Ingested document {document_id}: {len(content)} characters")
        return context


class PHIDetectionStage(PipelineStage):
    """Stage for PHI detection."""

    def __init__(self, phi_service):
        super().__init__(
            name="phi_detection",
            required_inputs=["content", "document_id"],
            produces_outputs=["phi_entities", "phi_detection_result"]
        )
        self.phi_service = phi_service

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process PHI detection."""
        content = context["content"]

        # Configure detection based on stage config
        detection_method = self.config.get("detection_method", "hybrid")
        confidence_threshold = self.config.get("confidence_threshold", 0.8)

        # Run PHI detection
        detection_result = self.phi_service.detect_phi_entities(
            text=content,
            detection_method=detection_method,
            confidence_threshold=confidence_threshold
        )

        # Update context
        context.update({
            "phi_entities": detection_result.entities,
            "phi_detection_result": detection_result
        })

        logger.info(f"Detected {len(detection_result.entities)} PHI entities")
        return context


class DocumentAnalysisStage(PipelineStage):
    """Stage for document analysis."""

    def __init__(self, document_analyzer):
        super().__init__(
            name="document_analysis",
            required_inputs=["content", "document_id"],
            produces_outputs=["document_analysis"]
        )
        self.document_analyzer = document_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process document analysis."""
        content = context["content"]
        document_id = context["document_id"]
        metadata = context.get("metadata", {})

        # Run document analysis
        analysis_result = self.document_analyzer.analyze_document(
            content=content,
            document_id=document_id,
            metadata=metadata
        )

        # Update context
        context["document_analysis"] = analysis_result

        logger.info(f"Completed document analysis: {analysis_result.document_type}")
        return context


class PHIAnalysisStage(PipelineStage):
    """Stage for PHI-specific analysis."""

    def __init__(self, phi_analyzer):
        super().__init__(
            name="phi_analysis",
            required_inputs=["phi_entities", "content", "document_id"],
            produces_outputs=["phi_analysis"]
        )
        self.phi_analyzer = phi_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process PHI analysis."""
        phi_entities = context["phi_entities"]
        content = context["content"]
        document_id = context["document_id"]

        # Run PHI analysis
        analysis_result = self.phi_analyzer.analyze_phi_distribution(
            phi_entities=phi_entities,
            content=content,
            document_id=document_id
        )

        # Update context
        context["phi_analysis"] = analysis_result

        logger.info(f"Completed PHI analysis: {analysis_result.privacy_risk_score:.2f} risk score")
        return context


class ComplianceAnalysisStage(PipelineStage):
    """Stage for compliance analysis."""

    def __init__(self, compliance_analyzer):
        super().__init__(
            name="compliance_analysis",
            required_inputs=["phi_entities", "content", "document_id"],
            produces_outputs=["compliance_analysis"]
        )
        self.compliance_analyzer = compliance_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process compliance analysis."""
        phi_entities = context["phi_entities"]
        content = context["content"]
        document_id = context["document_id"]
        redacted_content = context.get("redacted_content")

        # Run compliance analysis
        analysis_result = self.compliance_analyzer.analyze_compliance(
            phi_entities=phi_entities,
            document_content=content,
            document_id=document_id,
            redacted_content=redacted_content
        )

        # Update context
        context["compliance_analysis"] = analysis_result

        logger.info(f"Completed compliance analysis: {analysis_result.compliance_status}")
        return context


class RiskAnalysisStage(PipelineStage):
    """Stage for risk analysis."""

    def __init__(self, risk_analyzer):
        super().__init__(
            name="risk_analysis",
            required_inputs=["phi_entities", "content", "document_id"],
            produces_outputs=["risk_analysis"]
        )
        self.risk_analyzer = risk_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process risk analysis."""
        phi_entities = context["phi_entities"]
        content = context["content"]
        document_id = context["document_id"]
        metadata = context.get("metadata", {})

        # Run risk analysis
        analysis_result = self.risk_analyzer.analyze_risk(
            phi_entities=phi_entities,
            document_content=content,
            document_id=document_id,
            context=metadata
        )

        # Update context
        context["risk_analysis"] = analysis_result

        logger.info(f"Completed risk analysis: {analysis_result.risk_level} risk")
        return context


class ProcessingPipeline:
    """Main processing pipeline orchestrator."""

    def __init__(self, config: PipelineConfig = None):
        """Initialize processing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.stages: List[PipelineStage] = []
        self.metrics_collector = None
        self.tracer = None

        # Pipeline state
        self.active_pipelines: Dict[str, PipelineResult] = {}

        if self.config.enable_tracing:
            self.tracer = get_tracer()

    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for pipeline and all stages."""
        self.metrics_collector = collector
        for stage in self.stages:
            stage.set_metrics_collector(collector)

    def add_stage(self, stage: PipelineStage):
        """Add a processing stage to the pipeline."""
        if self.metrics_collector:
            stage.set_metrics_collector(self.metrics_collector)
        if self.tracer:
            stage.set_tracer(self.tracer)

        self.stages.append(stage)
        logger.info(f"Added stage: {stage.name}")

    def remove_stage(self, stage_name: str):
        """Remove a processing stage from the pipeline."""
        self.stages = [s for s in self.stages if s.name != stage_name]
        logger.info(f"Removed stage: {stage_name}")

    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    @trace_operation("pipeline_execution")
    async def process_document(self, document: Union[str, Dict[str, Any]],
                             pipeline_id: str = None) -> PipelineResult:
        """Process a document through the entire pipeline.
        
        Args:
            document: Document to process (string content or dict with content/metadata)
            pipeline_id: Optional pipeline ID for tracking
            
        Returns:
            Pipeline result
        """
        pipeline_id = pipeline_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Create pipeline result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            input_document=document if isinstance(document, dict) else {"content": document}
        )

        self.active_pipelines[pipeline_id] = result

        try:
            logger.info(f"Starting pipeline {pipeline_id}")

            # Initialize processing context
            context = {
                "document": document,
                "pipeline_id": pipeline_id,
                "pipeline_config": self.config
            }

            # Execute stages
            for i, stage in enumerate(self.stages):
                try:
                    logger.debug(f"Executing stage {i+1}/{len(self.stages)}: {stage.name}")

                    # Apply stage-specific configuration
                    if stage.name in self.config.stage_specific_config:
                        stage.config.update(self.config.stage_specific_config[stage.name])

                    # Execute stage
                    context = await stage.execute(context)

                    # Store stage result
                    result.stage_results[stage.name] = {
                        "status": "completed",
                        "outputs": stage.produces_outputs,
                        "execution_order": i + 1
                    }

                    # Checkpoint if enabled
                    if self.config.enable_checkpointing:
                        result.metadata[f"checkpoint_{stage.name}"] = datetime.utcnow().isoformat()

                except Exception as e:
                    error = {
                        "stage": stage.name,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    result.errors.append(error)
                    result.stage_results[stage.name] = {
                        "status": "failed",
                        "error": str(e),
                        "execution_order": i + 1
                    }

                    if self.config.retry_failed_stages and len(result.errors) <= self.config.max_retries:
                        logger.warning(f"Retrying stage {stage.name} (attempt {len(result.errors)})")
                        continue
                    else:
                        logger.error(f"Stage {stage.name} failed permanently: {e}")
                        result.status = PipelineStatus.FAILED
                        break

            # Finalize result
            if result.status != PipelineStatus.FAILED:
                result.status = PipelineStatus.COMPLETED
                result.processed_document = self._extract_final_document(context)

            result.completed_at = datetime.utcnow()
            result.processing_time_ms = (result.completed_at - start_time).total_seconds() * 1000

            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_histogram(
                    "pipeline_duration_ms",
                    result.processing_time_ms,
                    {"status": result.status.value}
                )
                self.metrics_collector.record_counter(
                    "pipeline_executions_total",
                    1.0,
                    {"status": result.status.value}
                )

            logger.info(f"Pipeline {pipeline_id} completed: {result.status.value}")
            return result

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.errors.append({
                "stage": "pipeline",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            result.completed_at = datetime.utcnow()
            result.processing_time_ms = (result.completed_at - start_time).total_seconds() * 1000

            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            return result

        finally:
            # Clean up active pipeline
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]

    def _extract_final_document(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final processed document from context."""
        return {
            "document_id": context.get("document_id"),
            "content": context.get("content"),
            "redacted_content": context.get("redacted_content"),
            "metadata": context.get("metadata", {}),
            "document_analysis": context.get("document_analysis"),
            "phi_analysis": context.get("phi_analysis"),
            "compliance_analysis": context.get("compliance_analysis"),
            "risk_analysis": context.get("risk_analysis"),
            "phi_entities": context.get("phi_entities", []),
            "processing_summary": {
                "total_phi_entities": len(context.get("phi_entities", [])),
                "compliance_score": getattr(context.get("compliance_analysis"), "overall_compliance_score", 0),
                "risk_level": getattr(context.get("risk_analysis"), "risk_level", "unknown"),
                "document_type": getattr(context.get("document_analysis"), "document_type", "unknown")
            }
        }

    async def process_batch(self, documents: List[Union[str, Dict[str, Any]]],
                           max_concurrent: int = None) -> List[PipelineResult]:
        """Process multiple documents concurrently.
        
        Args:
            documents: List of documents to process
            max_concurrent: Maximum concurrent pipelines (defaults to config value)
            
        Returns:
            List of pipeline results
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_stages

        async def process_with_semaphore(doc, semaphore):
            async with semaphore:
                return await self.process_document(doc)

        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [process_with_semaphore(doc, semaphore) for doc in documents]

        logger.info(f"Processing batch of {len(documents)} documents with max concurrency {max_concurrent}")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in batch
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed result for exception
                failed_result = PipelineResult(
                    pipeline_id=str(uuid.uuid4()),
                    status=PipelineStatus.FAILED,
                    input_document=documents[i] if isinstance(documents[i], dict) else {"content": documents[i]},
                    errors=[{
                        "stage": "batch_processing",
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)

        logger.info(f"Batch processing completed: {len(processed_results)} results")
        return processed_results

    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineResult]:
        """Get status of an active pipeline."""
        return self.active_pipelines.get(pipeline_id)

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel an active pipeline."""
        if pipeline_id in self.active_pipelines:
            self.active_pipelines[pipeline_id].status = PipelineStatus.CANCELLED
            del self.active_pipelines[pipeline_id]
            logger.info(f"Cancelled pipeline {pipeline_id}")
            return True
        return False

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        return {
            "active_pipelines": len(self.active_pipelines),
            "total_stages": len(self.stages),
            "stage_names": [s.name for s in self.stages],
            "config": {
                "max_concurrent_stages": self.config.max_concurrent_stages,
                "timeout_seconds": self.config.timeout_seconds,
                "retry_enabled": self.config.retry_failed_stages
            }
        }
