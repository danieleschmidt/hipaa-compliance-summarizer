"""
PHIRedactor — replaces detected PHI with [PHI_TYPE] placeholders.
"""

from typing import List, Dict, Any


class PHIRedactor:
    """Replaces PHI spans with labelled placeholders."""

    def redact(self, text: str, findings: List[Dict[str, Any]]) -> str:
        """
        Replace every finding in *text* with ``[PHI_TYPE]``.

        Findings may overlap; if they do the longest match wins.
        Processing happens right-to-left so offsets stay valid.
        """
        if not findings:
            return text

        # De-duplicate / resolve overlaps: sort by start desc, then end desc
        # Keep each finding only if it doesn't overlap with one already kept.
        sorted_findings = sorted(
            findings, key=lambda x: (x["start"], -(x["end"] - x["start"]))
        )

        # Merge overlapping spans — keep the first encountered at each position
        merged: List[Dict[str, Any]] = []
        covered_until = -1
        for f in sorted_findings:
            if f["start"] >= covered_until:
                merged.append(f)
                covered_until = f["end"]

        # Replace right-to-left to preserve offsets
        result = text
        for f in sorted(merged, key=lambda x: x["start"], reverse=True):
            placeholder = f"[{f['type']}]"
            result = result[: f["start"]] + placeholder + result[f["end"] :]

        return result
