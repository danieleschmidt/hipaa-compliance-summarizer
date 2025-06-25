from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .processor import ProcessingResult


@dataclass
class ComplianceReport:
    """Summary of HIPAA compliance metrics for a set of documents."""

    period: str
    documents_processed: int
    overall_compliance: float
    violations_detected: int
    recommendations: List[str] = field(default_factory=list)


class ComplianceReporter:
    """Generate compliance reports and audit information."""

    def __init__(self, *, violation_threshold: float = 0.8) -> None:
        self.violation_threshold = violation_threshold

    def generate_report(
        self,
        *,
        period: str,
        documents_processed: int,
        results: Optional[Sequence[ProcessingResult]] = None,
        include_recommendations: bool = False,
    ) -> ComplianceReport:
        """Create a :class:`ComplianceReport` summarizing processing results."""
        results = list(results) if results is not None else []

        if results:
            avg_score = sum(r.compliance_score for r in results) / len(results)
            violations = sum(
                1 for r in results if r.compliance_score < self.violation_threshold
            )
        else:
            avg_score = 1.0
            violations = 0

        recommendations: List[str] = []
        if include_recommendations:
            if avg_score < 0.95:
                recommendations.append(
                    "Implement additional staff training on PHI handling"
                )
            if violations:
                recommendations.append(
                    "Review data retention policies for imaging files"
                )

        return ComplianceReport(
            period=period,
            documents_processed=documents_processed,
            overall_compliance=round(avg_score, 2),
            violations_detected=violations,
            recommendations=recommendations,
        )
