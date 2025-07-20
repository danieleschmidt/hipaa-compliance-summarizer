from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import multiprocessing

from .documents import Document, detect_document_type
from .processor import HIPAAProcessor, ProcessingResult, ComplianceLevel
from .phi import PHIRedactor


@dataclass
class ErrorResult:
    """Represents a failed processing result."""
    
    file_path: str
    error: str
    error_type: str = "ProcessingError"
    
    # Add compatibility properties for dashboard generation
    @property
    def compliance_score(self) -> float:
        """Return 0.0 for failed processing."""
        return 0.0
    
    @property
    def phi_detected_count(self) -> int:
        """Return 0 for failed processing."""
        return 0


@dataclass
class BatchDashboard:
    """Simple summary of batch processing results."""

    documents_processed: int
    avg_compliance_score: float
    total_phi_detected: int

    def __str__(self) -> str:
        return (
            f"Documents processed: {self.documents_processed}\n"
            f"Average compliance score: {self.avg_compliance_score}\n"
            f"Total PHI detected: {self.total_phi_detected}"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return a dictionary representation of the dashboard."""
        return {
            "documents_processed": self.documents_processed,
            "avg_compliance_score": self.avg_compliance_score,
            "total_phi_detected": self.total_phi_detected,
        }

    def to_json(self) -> str:
        """Return a JSON string representation of the dashboard."""
        import json

        return json.dumps(self.to_dict(), indent=2)


logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process directories of healthcare documents."""

    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STANDARD) -> None:
        self.processor = HIPAAProcessor(compliance_level=compliance_level)

    def process_directory(
        self,
        input_dir: str,
        *,
        output_dir: Optional[str] = None,
        compliance_level: Optional[str] = None,
        generate_summaries: bool = False,
        show_progress: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[Union[ProcessingResult, ErrorResult]]:
        """Process all files in ``input_dir`` and optionally write outputs.
        
        Args:
            input_dir: Directory containing files to process
            output_dir: Optional output directory for processed files
            compliance_level: HIPAA compliance level (strict/standard/minimal)
            generate_summaries: Whether to generate summary files
            show_progress: Whether to display progress information
            max_workers: Number of worker threads (auto-detected if None)
        
        Returns:
            List of processing results or error results
        """
        # Auto-detect optimal worker count if not specified
        if max_workers is None:
            max_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
            logger.info("Auto-detected optimal workers: %d", max_workers)
        
        # Validate input parameters
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if compliance_level is not None:
            try:
                # Validate compliance level
                self.processor.compliance_level = ComplianceLevel(compliance_level)
            except ValueError as e:
                raise ValueError(f"Invalid compliance level: {e}")

        results: List[Union[ProcessingResult, ErrorResult]] = []
        
        # Validate input directory
        try:
            in_path = Path(input_dir)
            if not in_path.exists():
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            if not in_path.is_dir():
                raise ValueError(f"Input path is not a directory: {input_dir}")
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Permission denied accessing input directory: {e}")
        
        # Setup output directory if specified
        out_path = None
        if output_dir:
            try:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise PermissionError(f"Cannot create output directory: {e}")

        # Get list of files to process
        try:
            files = [f for f in in_path.iterdir() if f.is_file()]
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Permission denied accessing input directory: {e}")
        
        total = len(files)
        if total == 0:
            logger.warning("No files found in input directory: %s", input_dir)
            return results

        # Adaptive worker scaling based on file count
        if total < max_workers:
            max_workers = max(1, total)
            logger.info("Reducing workers to %d for %d files", max_workers, total)
        
        logger.info("Processing %d files with %d workers", total, max_workers)

        def handle(file: Path) -> Union[ProcessingResult, ErrorResult]:
            """Process a single file with comprehensive error handling."""
            try:
                logger.info("Processing file %s", file)
                
                # Detect document type
                try:
                    doc_type = detect_document_type(file.name)
                except Exception as e:
                    return ErrorResult(
                        file_path=str(file),
                        error=f"Failed to detect document type: {e}",
                        error_type="DocumentTypeError"
                    )
                
                # Create document object and process
                try:
                    doc = Document(str(file), doc_type)
                    result = self.processor.process_document(doc)
                except (IOError, OSError) as e:
                    return ErrorResult(
                        file_path=str(file),
                        error=f"File read error: {e}",
                        error_type="FileReadError"
                    )
                except MemoryError as e:
                    return ErrorResult(
                        file_path=str(file),
                        error=f"Out of memory: {e}",
                        error_type="MemoryError"
                    )
                except Exception as e:
                    return ErrorResult(
                        file_path=str(file),
                        error=f"Processing error: {e}",
                        error_type="ProcessingError"
                    )
                
                # Write output files if output directory specified
                if out_path:
                    try:
                        out_file = out_path / file.name
                        out_file.write_text(result.redacted.text)
                        
                        if generate_summaries:
                            summary_file = out_file.with_suffix(out_file.suffix + ".summary.txt")
                            summary_file.write_text(result.summary)
                    except (IOError, OSError, PermissionError) as e:
                        return ErrorResult(
                            file_path=str(file),
                            error=f"Cannot write file: {e}",
                            error_type="FileWriteError"
                        )
                
                return result
                
            except Exception as e:
                # Catch-all for any unexpected errors
                return ErrorResult(
                    file_path=str(file),
                    error=f"Unexpected error: {e}",
                    error_type="UnexpectedError"
                )

        # Process files with thread pool and error handling
        processed = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and track futures
                future_to_file = {executor.submit(handle, file): file for file in files}
                
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        processed += 1
                        
                        if show_progress:
                            status = "✓" if isinstance(result, ProcessingResult) else "✗"
                            print(f"[{processed}/{total}] {status} {file.name}")
                            
                    except Exception as e:
                        # Handle exceptions from the future
                        error_result = ErrorResult(
                            file_path=str(file),
                            error=f"Thread execution error: {e}",
                            error_type="ThreadExecutionError"
                        )
                        results.append(error_result)
                        processed += 1
                        
                        if show_progress:
                            print(f"[{processed}/{total}] ✗ {file.name}")
                            
        except Exception as e:
            raise RuntimeError(f"Thread pool execution failed: {e}")

        # Log summary
        successful_count = sum(1 for r in results if isinstance(r, ProcessingResult))
        error_count = len(results) - successful_count
        
        if error_count > 0:
            logger.warning("Processed %d files: %d successful, %d errors", 
                         len(results), successful_count, error_count)
        else:
            logger.info("Successfully processed %d files", successful_count)

        return results

    def generate_dashboard(self, results: Sequence[Union[ProcessingResult, ErrorResult]]) -> BatchDashboard:
        """Create a dashboard summary for ``results``."""
        if results:
            # Calculate metrics safely, handling both ProcessingResult and ErrorResult
            total_score = 0.0
            total_phi = 0
            valid_results = 0
            
            for r in results:
                try:
                    # Use safe attribute access
                    score = getattr(r, 'compliance_score', 0.0)
                    phi_count = getattr(r, 'phi_detected_count', 0)
                    
                    if score is not None and phi_count is not None:
                        total_score += float(score)
                        total_phi += int(phi_count)
                        valid_results += 1
                except (ValueError, TypeError, AttributeError):
                    # Skip invalid results
                    continue
            
            avg_score = total_score / valid_results if valid_results > 0 else 1.0
        else:
            avg_score = 1.0
            total_phi = 0
            
        return BatchDashboard(
            documents_processed=len(results),
            avg_compliance_score=round(avg_score, 2),
            total_phi_detected=total_phi,
        )

    def save_dashboard(self, results: Sequence[Union[ProcessingResult, ErrorResult]], path: Union[str, Path]) -> None:
        """Write a dashboard summary for ``results`` to ``path`` as JSON."""
        # Convert to string for validation
        path_str = str(path)
        
        # Validate file path
        if not path_str or '\x00' in path_str:
            raise ValueError(f"Invalid file path: {path}")
        
        try:
            dash = self.generate_dashboard(results)
            dashboard_json = dash.to_json()
            
            # Write with proper error handling
            dashboard_path = Path(path_str)
            dashboard_path.write_text(dashboard_json)
            logger.info("Dashboard saved to %s", path_str)
            
        except PermissionError as e:
            raise PermissionError(f"Cannot write dashboard to {path_str}: {e}")
        except OSError as e:
            if "No space left" in str(e):
                raise OSError(f"Failed to write dashboard - disk full: {e}")
            else:
                raise OSError(f"Failed to write dashboard to {path_str}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving dashboard to {path_str}: {e}")

    def get_cache_performance(self) -> dict:
        """Get information about PHI detection cache performance."""
        try:
            cache_info = PHIRedactor.get_cache_info()
        except (AttributeError, RuntimeError) as e:
            logger.warning("Cache information not available: %s", e)
            # Return safe defaults
            return {
                "pattern_compilation": {
                    "hits": 0,
                    "misses": 0,
                    "hit_ratio": 0.0,
                    "current_size": 0,
                    "max_size": 0
                },
                "phi_detection": {
                    "hits": 0,
                    "misses": 0,
                    "hit_ratio": 0.0,
                    "current_size": 0,
                    "max_size": 0
                }
            }
        
        try:
            # Calculate cache hit ratios with safe attribute access
            pattern_cache = cache_info.get("pattern_compilation")
            phi_cache = cache_info.get("phi_detection")
            
            def safe_cache_metrics(cache_obj):
                """Safely extract cache metrics."""
                if cache_obj is None:
                    return {"hits": 0, "misses": 0, "hit_ratio": 0.0, "current_size": 0, "max_size": 0}
                
                try:
                    hits = getattr(cache_obj, 'hits', 0)
                    misses = getattr(cache_obj, 'misses', 0)
                    current_size = getattr(cache_obj, 'currsize', 0)
                    max_size = getattr(cache_obj, 'maxsize', 0)
                    
                    hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0.0
                    
                    return {
                        "hits": int(hits),
                        "misses": int(misses),
                        "hit_ratio": round(float(hit_ratio), 3),
                        "current_size": int(current_size),
                        "max_size": int(max_size) if max_size is not None else 0
                    }
                except (ValueError, TypeError, AttributeError):
                    return {"hits": 0, "misses": 0, "hit_ratio": 0.0, "current_size": 0, "max_size": 0}
            
            return {
                "pattern_compilation": safe_cache_metrics(pattern_cache),
                "phi_detection": safe_cache_metrics(phi_cache)
            }
            
        except Exception as e:
            logger.warning("Error calculating cache performance: %s", e)
            # Return safe defaults
            return {
                "pattern_compilation": {
                    "hits": 0,
                    "misses": 0,
                    "hit_ratio": 0.0,
                    "current_size": 0,
                    "max_size": 0
                },
                "phi_detection": {
                    "hits": 0,
                    "misses": 0,
                    "hit_ratio": 0.0,
                    "current_size": 0,
                    "max_size": 0
                }
            }

    def clear_cache(self) -> None:
        """Clear PHI detection caches to free memory."""
        try:
            PHIRedactor.clear_cache()
            logger.info("PHI detection caches cleared")
        except Exception as e:
            logger.error("Failed to clear cache: %s", e)
            raise RuntimeError(f"Failed to clear cache: {e}")
