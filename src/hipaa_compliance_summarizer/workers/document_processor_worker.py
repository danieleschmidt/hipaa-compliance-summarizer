"""Worker for document processing jobs."""

import logging
from typing import Any, Dict

from ..monitoring.tracing import trace_operation
from ..processors.pipeline import ProcessingPipeline
from .queue_manager import Job

logger = logging.getLogger(__name__)


class DocumentProcessorWorker:
    """Worker for processing documents through the HIPAA compliance pipeline."""

    def __init__(self, pipeline: ProcessingPipeline):
        """Initialize document processor worker.
        
        Args:
            pipeline: Processing pipeline instance
        """
        self.pipeline = pipeline
        self.processed_count = 0
        self.error_count = 0

    @trace_operation("document_processor_worker")
    async def process_document_job(self, job: Job) -> Dict[str, Any]:
        """Process a document processing job.
        
        Args:
            job: Job containing document to process
            
        Returns:
            Processing result
        """
        logger.info(f"Processing document job {job.job_id}")

        try:
            # Extract document data from job
            document_data = job.data.get("document")
            if not document_data:
                raise ValueError("No document data provided in job")

            # Get processing options
            options = job.data.get("options", {})

            # Process document through pipeline
            result = await self.pipeline.process_document(
                document=document_data,
                pipeline_id=f"job_{job.job_id}"
            )

            # Update statistics
            self.processed_count += 1

            # Prepare result
            processing_result = {
                "pipeline_result": result.to_dict(),
                "processing_time_ms": result.processing_time_ms,
                "status": result.status.value,
                "document_id": result.processed_document.get("document_id") if result.processed_document else None,
                "errors": result.errors
            }

            # Add summary information
            if result.processed_document:
                summary = result.processed_document.get("processing_summary", {})
                processing_result["summary"] = summary

            logger.info(f"Document job {job.job_id} completed: {result.status.value}")
            return processing_result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Document job {job.job_id} failed: {e}")
            raise

    @trace_operation("batch_document_processor_worker")
    async def process_batch_job(self, job: Job) -> Dict[str, Any]:
        """Process a batch document processing job.
        
        Args:
            job: Job containing batch of documents
            
        Returns:
            Batch processing result
        """
        logger.info(f"Processing batch job {job.job_id}")

        try:
            # Extract batch data
            documents = job.data.get("documents", [])
            if not documents:
                raise ValueError("No documents provided in batch job")

            max_concurrent = job.data.get("max_concurrent", 3)

            # Process batch
            results = await self.pipeline.process_batch(
                documents=documents,
                max_concurrent=max_concurrent
            )

            # Analyze batch results
            successful = [r for r in results if r.status.value == "completed"]
            failed = [r for r in results if r.status.value == "failed"]

            # Update statistics
            self.processed_count += len(successful)
            self.error_count += len(failed)

            # Prepare batch result
            batch_result = {
                "total_documents": len(documents),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(documents) * 100 if documents else 0,
                "results": [r.to_dict() for r in results],
                "summary": {
                    "total_phi_entities": sum(
                        r.processed_document.get("processing_summary", {}).get("total_phi_entities", 0)
                        for r in successful if r.processed_document
                    ),
                    "avg_compliance_score": (
                        sum(
                            r.processed_document.get("processing_summary", {}).get("compliance_score", 0)
                            for r in successful if r.processed_document
                        ) / len(successful) if successful else 0
                    ),
                    "high_risk_documents": sum(
                        1 for r in successful
                        if r.processed_document and
                           r.processed_document.get("processing_summary", {}).get("risk_level") in ["high", "critical"]
                    )
                }
            }

            logger.info(f"Batch job {job.job_id} completed: {len(successful)}/{len(documents)} successful")
            return batch_result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Batch job {job.job_id} failed: {e}")
            raise

    async def process_redaction_job(self, job: Job) -> Dict[str, Any]:
        """Process a redaction-only job.
        
        Args:
            job: Job containing document for redaction
            
        Returns:
            Redaction result
        """
        logger.info(f"Processing redaction job {job.job_id}")

        try:
            # Extract job data
            content = job.data.get("content")
            phi_entities = job.data.get("phi_entities", [])
            redaction_config = job.data.get("redaction_config", {})

            if not content:
                raise ValueError("No content provided for redaction")

            # Import redaction transformer
            from ..processors.transformers import PHIRedactionTransformer

            # Create redaction transformer
            redactor = PHIRedactionTransformer(redaction_config)

            # Perform redaction
            context = {"phi_entities": phi_entities}
            redaction_result = redactor.transform(content, context)

            # Prepare result
            result = {
                "original_content": redaction_result.original_content,
                "redacted_content": redaction_result.transformed_content,
                "redaction_successful": redaction_result.success,
                "modifications": redaction_result.modifications,
                "metadata": redaction_result.metadata,
                "error_message": redaction_result.error_message
            }

            logger.info(f"Redaction job {job.job_id} completed: {len(redaction_result.modifications)} modifications")
            return result

        except Exception as e:
            logger.error(f"Redaction job {job.job_id} failed: {e}")
            raise

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "processed_documents": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.processed_count / (self.processed_count + self.error_count)) * 100
                if (self.processed_count + self.error_count) > 0 else 0
            ),
            "worker_type": "document_processor"
        }
