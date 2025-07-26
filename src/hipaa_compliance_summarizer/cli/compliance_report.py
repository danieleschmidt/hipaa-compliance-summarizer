#!/usr/bin/env python
"""Generate a simple compliance report."""
from argparse import ArgumentParser
import logging
from hipaa_compliance_summarizer import ComplianceReporter


def main() -> None:
    """
    Entry point for the HIPAA compliance reporting CLI tool.
    
    Generates detailed compliance reports for specified audit periods,
    including metrics on documents processed, PHI detection accuracy,
    and compliance recommendations.
    
    Command-line arguments:
        --audit-period: Reporting period (e.g., "2024-Q1", "2024-01")
        --documents-processed: Number of documents processed (default: 0)
        --include-recommendations: Include compliance recommendations in output
    
    Outputs a formatted compliance report to stdout with metrics and
    recommendations for improving HIPAA compliance posture.
    """
    parser = ArgumentParser(description="Generate compliance report")
    parser.add_argument("--audit-period", required=True, help="Reporting period")
    parser.add_argument("--documents-processed", type=int, default=0)
    parser.add_argument("--include-recommendations", action="store_true")
    args = parser.parse_args()

    # Set up logging for CLI output
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    reporter = ComplianceReporter()
    report = reporter.generate_report(
        period=args.audit_period,
        documents_processed=args.documents_processed,
        include_recommendations=args.include_recommendations,
    )
    logger.info(report.overall_compliance)
    logger.info(report.recommendations)


if __name__ == "__main__":
    main()
