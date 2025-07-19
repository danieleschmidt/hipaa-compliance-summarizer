from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from .documents import Document, detect_document_type
from .processor import HIPAAProcessor, ProcessingResult, ComplianceLevel
from .phi import PHIRedactor


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
        max_workers: int = 1,
    ) -> List[ProcessingResult]:
        """Process all files in ``input_dir`` and optionally write outputs."""
        if compliance_level is not None:
            # allow caller to override compliance level
            self.processor.compliance_level = ComplianceLevel(compliance_level)

        results: List[ProcessingResult] = []
        in_path = Path(input_dir)
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        files = [f for f in in_path.iterdir() if f.is_file()]
        total = len(files)

        def handle(file: Path) -> ProcessingResult:
            logger.info("Processing file %s", file)
            doc_type = detect_document_type(file.name)
            doc = Document(str(file), doc_type)
            result = self.processor.process_document(doc)
            if out_path:
                out_file = out_path / file.name
                out_file.write_text(result.redacted.text)
                if generate_summaries:
                    summary_file = out_file.with_suffix(out_file.suffix + ".summary.txt")
                    summary_file.write_text(result.summary)
            return result

        processed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(handle, files):
                processed += 1
                results.append(result)
                if show_progress:
                    print(f"[{processed}/{total}] {files[processed-1].name}")

        return results

    def generate_dashboard(self, results: Sequence[ProcessingResult]) -> BatchDashboard:
        """Create a dashboard summary for ``results``."""
        if results:
            avg_score = sum(r.compliance_score for r in results) / len(results)
            total_phi = sum(r.phi_detected_count for r in results)
        else:
            avg_score = 1.0
            total_phi = 0
        return BatchDashboard(
            documents_processed=len(results),
            avg_compliance_score=round(avg_score, 2),
            total_phi_detected=total_phi,
        )

    def save_dashboard(self, results: Sequence[ProcessingResult], path: str) -> None:
        """Write a dashboard summary for ``results`` to ``path`` as JSON."""
        dash = self.generate_dashboard(results)
        Path(path).write_text(dash.to_json())
        logger.info("Dashboard saved to %s", path)

    def get_cache_performance(self) -> dict:
        """Get information about PHI detection cache performance."""
        cache_info = PHIRedactor.get_cache_info()
        
        # Calculate cache hit ratios
        pattern_cache = cache_info["pattern_compilation"]
        phi_cache = cache_info["phi_detection"]
        
        pattern_hit_ratio = pattern_cache.hits / (pattern_cache.hits + pattern_cache.misses) if (pattern_cache.hits + pattern_cache.misses) > 0 else 0
        phi_hit_ratio = phi_cache.hits / (phi_cache.hits + phi_cache.misses) if (phi_cache.hits + phi_cache.misses) > 0 else 0
        
        return {
            "pattern_compilation": {
                "hits": pattern_cache.hits,
                "misses": pattern_cache.misses,
                "hit_ratio": round(pattern_hit_ratio, 3),
                "current_size": pattern_cache.currsize,
                "max_size": pattern_cache.maxsize
            },
            "phi_detection": {
                "hits": phi_cache.hits,
                "misses": phi_cache.misses,
                "hit_ratio": round(phi_hit_ratio, 3),
                "current_size": phi_cache.currsize,
                "max_size": phi_cache.maxsize
            }
        }

    def clear_cache(self) -> None:
        """Clear PHI detection caches to free memory."""
        PHIRedactor.clear_cache()
        logger.info("PHI detection caches cleared")
