"""
ComplianceAuditor — timestamped audit log of PHI detection and redaction.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


class ComplianceAuditor:
    """Records every PHI detection and redaction event with timestamps."""

    def __init__(self) -> None:
        self.log: List[Dict[str, Any]] = []

    def record(
        self,
        document_id: str,
        findings: List[Dict[str, Any]],
        redacted: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add audit entries for a document's PHI findings.

        Parameters
        ----------
        document_id : str
            Identifier for the source document / text snippet.
        findings : list
            Output of ``PHIDetector.detect()``.
        redacted : bool
            Whether the PHI was subsequently redacted.
        metadata : dict, optional
            Any extra context to store alongside the entry.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for finding in findings:
            entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "document_id": document_id,
                "phi_type": finding["type"],
                "phi_value": finding["value"],
                "start_offset": finding["start"],
                "end_offset": finding["end"],
                "redacted": redacted,
            }
            if metadata:
                entry["metadata"] = metadata
            self.log.append(entry)

    def summary_by_type(self) -> Dict[str, int]:
        """Return count of each PHI type found across all logged entries."""
        counts: Dict[str, int] = {}
        for entry in self.log:
            phi_type = entry["phi_type"]
            counts[phi_type] = counts.get(phi_type, 0) + 1
        return counts

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full audit log to a JSON string."""
        return json.dumps(self.log, indent=indent, ensure_ascii=False)

    def clear(self) -> None:
        """Reset the audit log."""
        self.log = []
