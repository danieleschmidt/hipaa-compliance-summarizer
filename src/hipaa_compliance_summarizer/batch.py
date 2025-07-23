from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import multiprocessing
import mmap
import json
from typing import Iterator, Dict

from .documents import Document, detect_document_type
from .processor import HIPAAProcessor, ProcessingResult, ComplianceLevel
from .phi import PHIRedactor
from .security import validate_directory_path, validate_file_for_processing, SecurityError, sanitize_filename


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

    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STANDARD,
                 performance_monitor: Optional[object] = None) -> None:
        self.processor = HIPAAProcessor(compliance_level=compliance_level)
        self._file_content_cache = {}  # Simple LRU-style cache for file contents
        self._max_cache_size = 50  # Limit cache to 50 files to prevent memory issues
        self.performance_monitor = performance_monitor
        
        # Set up processor with monitoring if available
        if performance_monitor and hasattr(self.processor, 'phi_redactor'):
            self.processor.phi_redactor.performance_monitor = performance_monitor

    def _optimized_file_read(self, file_path: Path) -> str:
        """Optimized file reading with caching and memory mapping for large files."""
        file_key = str(file_path)
        
        # Check cache first
        if file_key in self._file_content_cache:
            return self._file_content_cache[file_key]
        
        file_size = file_path.stat().st_size
        
        # For small files (< 1MB), use regular read with caching
        if file_size < 1024 * 1024:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Manage cache size
            if len(self._file_content_cache) >= self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._file_content_cache))
                del self._file_content_cache[oldest_key]
            
            self._file_content_cache[file_key] = content
            return content
        
        # For large files (>= 1MB), use memory mapping
        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    content = mm.read().decode('utf-8', errors='ignore')
            return content
        except (OSError, ValueError):
            # Fallback to regular read if mmap fails
            return file_path.read_text(encoding='utf-8', errors='ignore')

    def _batch_file_iterator(self, files: List[Path], batch_size: int = 5) -> Iterator[List[Path]]:
        """Yield batches of files for processing."""
        for i in range(0, len(files), batch_size):
            yield files[i:i + batch_size]

    def _preload_files(self, files: List[Path]) -> None:
        """Preload small files into cache to reduce I/O during processing."""
        small_files = [f for f in files if f.stat().st_size < 512 * 1024]  # < 512KB
        
        if small_files:
            logger.info("Preloading %d small files into cache", len(small_files))
            for file_path in small_files[:self._max_cache_size]:
                try:
                    self._optimized_file_read(file_path)
                except Exception as e:
                    logger.warning("Failed to preload file %s: %s", file_path, e)

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
        # Validate parameters and setup
        max_workers = self._validate_and_setup_workers(max_workers)
        self._validate_compliance_level(compliance_level)
        
        # Validate input directory
        in_path = self._validate_input_directory(input_dir)
        
        # Setup output directory
        out_path = self._setup_output_directory(output_dir)
        
        # Collect files to process
        files = self._collect_files_to_process(in_path)
        if not files:
            logger.warning("No files found in input directory: %s", input_dir)
            return []
        
        # Optimize processing order and workers
        files, max_workers = self._optimize_processing_setup(files, max_workers)
        
        logger.info("Processing %d files with %d workers", len(files), max_workers)
        
        # Process files and return results
        return self._execute_file_processing(
            files, out_path, generate_summaries, show_progress, max_workers
        )
    
    def _validate_and_setup_workers(self, max_workers: Optional[int]) -> int:
        """Validate and setup optimal worker count."""
        if max_workers is None:
            max_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
            logger.info("Auto-detected optimal workers: %d", max_workers)
        
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        return max_workers
    
    def _validate_compliance_level(self, compliance_level: Optional[str]) -> None:
        """Validate and set compliance level."""
        if compliance_level is not None:
            try:
                self.processor.compliance_level = ComplianceLevel(compliance_level)
            except ValueError as e:
                raise ValueError(f"Invalid compliance level: {e}")
    
    def _validate_input_directory(self, input_dir: str) -> Path:
        """Validate input directory with security checks."""
        try:
            return validate_directory_path(input_dir)
        except SecurityError as e:
            error_msg = str(e)
            if "does not exist" in error_msg:
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            elif "not a directory" in error_msg:
                raise ValueError(f"Input path is not a directory: {input_dir}")
            elif "not readable" in error_msg:
                raise PermissionError(f"Permission denied accessing input directory: {input_dir}")
            else:
                raise ValueError(f"Security validation failed for input directory: {e}")
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Permission denied accessing input directory: {e}")
    
    def _setup_output_directory(self, output_dir: Optional[str]) -> Optional[Path]:
        """Setup output directory with security validation."""
        if not output_dir:
            return None
            
        try:
            # Validate the parent directory path for security
            output_parent = str(Path(output_dir).parent)
            if output_parent != '.':
                validate_directory_path(output_parent)
            
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Validate the created directory
            validate_directory_path(str(out_path))
            return out_path
            
        except SecurityError as e:
            # Only reject truly dangerous paths, allow reasonable directories to be created
            if "dangerous path pattern" in str(e):
                raise ValueError(f"Security validation failed for output directory: {e}")
            # For other security errors, try to proceed but log a warning
            logger.warning("Output directory security check: %s", e)
            return Path(output_dir)
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Cannot create output directory: {e}")
    
    def _collect_files_to_process(self, in_path: Path) -> List[Path]:
        """Collect list of files to process from input directory."""
        try:
            return [f for f in in_path.iterdir() if f.is_file()]
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Permission denied accessing input directory: {e}")
    
    def _optimize_processing_setup(self, files: List[Path], max_workers: int) -> tuple[List[Path], int]:
        """Optimize file processing order and worker count."""
        # Sort files by size for better processing order (small files first)
        files.sort(key=lambda f: f.stat().st_size)
        
        # Preload small files into cache for faster access
        self._preload_files(files)
        
        # Adaptive worker scaling based on file count
        total = len(files)
        if total < max_workers:
            max_workers = max(1, total)
            logger.info("Reducing workers to %d for %d files", max_workers, total)
        
        return files, max_workers
    
    def _execute_file_processing(
        self,
        files: List[Path],
        out_path: Optional[Path],
        generate_summaries: bool,
        show_progress: bool,
        max_workers: int
    ) -> List[Union[ProcessingResult, ErrorResult]]:
        """Execute file processing with thread pool."""
        results: List[Union[ProcessingResult, ErrorResult]] = []
        total = len(files)
        processed = 0
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and track futures
                future_to_file = {
                    executor.submit(self._process_file_with_monitoring, file, out_path, generate_summaries): file 
                    for file in files
                }
                
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
        self._log_processing_summary(results)
        return results
    
    def _process_file_with_monitoring(
        self, 
        file: Path, 
        out_path: Optional[Path], 
        generate_summaries: bool
    ) -> Union[ProcessingResult, ErrorResult]:
        """Process a single file with comprehensive monitoring and error handling."""
        document_id = str(file)
        
        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_document_processing(document_id)
        
        try:
            logger.info("Processing file %s", file)
            
            # Security validation first
            try:
                validated_file = validate_file_for_processing(str(file))
            except SecurityError as e:
                return self._create_error_result(
                    file, f"Security validation failed: {e}", "SecurityError", document_id
                )
            
            # Detect document type
            try:
                doc_type = detect_document_type(validated_file.name)
            except Exception as e:
                return self._create_error_result(
                    file, f"Failed to detect document type: {e}", "DocumentTypeError", document_id
                )
            
            # Create document object and process
            try:
                doc = Document(str(validated_file), doc_type)
                result = self.processor.process_document(doc)
            except (IOError, OSError) as e:
                return self._create_error_result(
                    file, f"File read error: {e}", "FileReadError", document_id
                )
            except MemoryError as e:
                return self._create_error_result(
                    file, f"Out of memory: {e}", "MemoryError", document_id
                )
            except Exception as e:
                return self._create_error_result(
                    file, f"Processing error: {e}", "ProcessingError", document_id
                )
            
            # Write output files if output directory specified
            if out_path:
                try:
                    self._write_output_files(validated_file, out_path, result, generate_summaries)
                except (IOError, OSError, PermissionError) as e:
                    return self._create_error_result(
                        file, f"Cannot write file: {e}", "FileWriteError", document_id
                    )
            
            # Successfully processed - record metrics
            if self.performance_monitor:
                self.performance_monitor.end_document_processing(
                    document_id, success=True, result=result
                )
            
            return result
            
        except Exception as e:
            # Catch-all for any unexpected errors
            return self._create_error_result(
                file, f"Unexpected error: {e}", "UnexpectedError", document_id
            )
    
    def _create_error_result(
        self, 
        file: Path, 
        error_msg: str, 
        error_type: str, 
        document_id: str
    ) -> ErrorResult:
        """Create an error result and handle performance monitoring."""
        error_result = ErrorResult(
            file_path=str(file),
            error=error_msg,
            error_type=error_type
        )
        
        if self.performance_monitor:
            self.performance_monitor.end_document_processing(
                document_id, success=False, error=error_result.error
            )
        
        return error_result
    
    def _write_output_files(
        self, 
        validated_file: Path, 
        out_path: Path, 
        result: ProcessingResult, 
        generate_summaries: bool
    ) -> None:
        """Write output files to the specified directory."""
        # Sanitize filename for security
        safe_filename = sanitize_filename(validated_file.name)
        out_file = out_path / safe_filename
        
        # Ensure we don't overwrite existing files
        counter = 1
        while out_file.exists():
            name, ext = os.path.splitext(safe_filename)
            out_file = out_path / f"{name}_{counter}{ext}"
            counter += 1
        
        out_file.write_text(result.redacted.text)
        
        if generate_summaries:
            summary_file = out_file.with_suffix(out_file.suffix + ".summary.txt")
            summary_file.write_text(result.summary)
    
    def _log_processing_summary(self, results: List[Union[ProcessingResult, ErrorResult]]) -> None:
        """Log summary of processing results."""
        successful_count = sum(1 for r in results if isinstance(r, ProcessingResult))
        error_count = len(results) - successful_count
        
        if error_count > 0:
            logger.warning("Processed %d files: %d successful, %d errors", 
                         len(results), successful_count, error_count)
        else:
            logger.info("Successfully processed %d files", successful_count)


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
    
    def generate_performance_dashboard(self) -> Optional[Dict]:
        """Generate enhanced performance dashboard with monitoring data."""
        if not self.performance_monitor:
            return None
        
        from .monitoring import MonitoringDashboard
        dashboard = MonitoringDashboard(self.performance_monitor)
        return dashboard.generate_dashboard_data()
    
    def save_performance_dashboard(self, path: Union[str, Path]) -> None:
        """Save enhanced performance dashboard to JSON file."""
        if not self.performance_monitor:
            logger.warning("No performance monitor available - cannot save performance dashboard")
            return
        
        dashboard_data = self.generate_performance_dashboard()
        if dashboard_data:
            path_str = str(path)
            try:
                dashboard_path = Path(path_str)
                dashboard_path.write_text(json.dumps(dashboard_data, indent=2, default=str))
                logger.info("Performance dashboard saved to %s", path_str)
            except Exception as e:
                logger.error(f"Failed to save performance dashboard: {e}")
                raise

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
        """Clear PHI detection caches and file content cache to free memory."""
        try:
            PHIRedactor.clear_cache()
            self._file_content_cache.clear()
            logger.info("PHI detection and file content caches cleared")
        except Exception as e:
            logger.error("Failed to clear cache: %s", e)
            raise RuntimeError(f"Failed to clear cache: {e}")

    def get_file_cache_info(self) -> dict:
        """Get information about file content cache performance."""
        return {
            "file_cache_size": len(self._file_content_cache),
            "file_cache_max_size": self._max_cache_size,
            "file_cache_usage_ratio": len(self._file_content_cache) / self._max_cache_size if self._max_cache_size > 0 else 0.0
        }
