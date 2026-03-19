"""
ComplianceSummarizer — generates a human-readable compliance report.

Risk levels
-----------
HIGH   : SSN, MRN, or DATE (which acts as DOB proxy) found
MEDIUM : NAME + (DATE or GEOGRAPHIC or PHONE or EMAIL)
LOW    : any other PHI category only
"""

from typing import List, Dict, Any, Set


# PHI types that elevate risk to HIGH
HIGH_RISK_TYPES: Set[str] = {"SSN", "MRN", "DATE", "HEALTH_PLAN_BENEFICIARY"}

# PHI types whose co-occurrence with NAME elevates to MEDIUM
MEDIUM_RISK_TYPES: Set[str] = {
    "GEOGRAPHIC",
    "PHONE",
    "FAX",
    "EMAIL",
    "AGE_OVER_89",
    "ACCOUNT_NUMBER",
}


class ComplianceSummarizer:
    """Produces a compliance report from an audit log."""

    def summarize(
        self,
        audit_log: List[Dict[str, Any]],
        original_text: str,
        redacted_text: str,
    ) -> Dict[str, Any]:
        """
        Build a compliance report.

        Parameters
        ----------
        audit_log : list
            Entries produced by ``ComplianceAuditor.log``.
        original_text : str
            The unredacted source text.
        redacted_text : str
            The text after ``PHIRedactor.redact()``.

        Returns
        -------
        dict with keys:
            risk_level          : "HIGH" | "MEDIUM" | "LOW" | "NONE"
            phi_categories      : list of PHI types found
            phi_count           : total PHI instances detected
            redaction_score     : float (0–100 %)
            redacted_count      : int
            detected_count      : int
            summary             : human-readable string
        """
        if not audit_log:
            return {
                "risk_level": "NONE",
                "phi_categories": [],
                "phi_count": 0,
                "redaction_score": 100.0,
                "redacted_count": 0,
                "detected_count": 0,
                "summary": "No PHI detected. Document appears clean.",
            }

        detected_count = len(audit_log)
        redacted_count = sum(1 for e in audit_log if e.get("redacted", False))
        redaction_score = (redacted_count / detected_count) * 100.0 if detected_count else 100.0

        phi_types_found: Set[str] = {e["phi_type"] for e in audit_log}

        # Determine risk level
        if phi_types_found & HIGH_RISK_TYPES:
            risk_level = "HIGH"
        elif "NAME" in phi_types_found and (phi_types_found & MEDIUM_RISK_TYPES):
            risk_level = "MEDIUM"
        elif phi_types_found:
            risk_level = "LOW"
        else:
            risk_level = "NONE"

        phi_categories = sorted(phi_types_found)

        summary_lines = [
            f"Risk Level: {risk_level}",
            f"PHI Categories Found: {', '.join(phi_categories)}",
            f"Total PHI Instances: {detected_count}",
            f"Redacted: {redacted_count}/{detected_count} "
            f"({redaction_score:.1f}% completeness)",
        ]
        if redaction_score < 100.0:
            summary_lines.append(
                "⚠️  Not all detected PHI was redacted — manual review required."
            )
        else:
            summary_lines.append("✅ All detected PHI has been redacted.")

        return {
            "risk_level": risk_level,
            "phi_categories": phi_categories,
            "phi_count": detected_count,
            "redaction_score": round(redaction_score, 2),
            "redacted_count": redacted_count,
            "detected_count": detected_count,
            "summary": "\n".join(summary_lines),
        }
