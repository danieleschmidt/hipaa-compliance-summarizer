"""
HIPAA Compliance Pipeline

A pure-Python, regex-based toolkit for PHI detection, redaction,
audit logging, and compliance reporting.
"""

from .detector import PHIDetector
from .redactor import PHIRedactor
from .auditor import ComplianceAuditor
from .summarizer import ComplianceSummarizer

__all__ = ["PHIDetector", "PHIRedactor", "ComplianceAuditor", "ComplianceSummarizer"]
__version__ = "1.0.0"
