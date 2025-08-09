#!/usr/bin/env python
"""Example script to process a single medical document."""
import logging
from argparse import ArgumentParser

from hipaa_compliance_summarizer import HIPAAProcessor


def main() -> None:
    """
    Entry point for the HIPAA document summarization CLI tool.
    
    Processes a single healthcare document for PHI detection, redaction,
    and compliance summarization. Supports configurable compliance levels
    for different use cases.
    
    Command-line arguments:
        --file: Path to the document to process (required)
        --compliance-level: Compliance strictness level
                          - strict: Maximum PHI detection and redaction
                          - standard: Balanced approach (default)
                          - minimal: Minimal redaction for internal use
    
    Outputs a redacted document summary with compliance metadata including
    PHI detection count, confidence scores, and compliance assessment.
    """
    parser = ArgumentParser(description="Summarize a medical document")
    parser.add_argument("--file", required=True, help="Path to the document")
    parser.add_argument(
        "--compliance-level",
        choices=["strict", "standard", "minimal"],
        default="standard",
        help="Compliance strictness level",
    )
    args = parser.parse_args()

    # Set up logging for CLI output
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    processor = HIPAAProcessor(compliance_level=args.compliance_level)
    result = processor.process_document(args.file)
    logger.info(result.summary)


if __name__ == "__main__":
    main()
