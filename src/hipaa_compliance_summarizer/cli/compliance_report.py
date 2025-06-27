#!/usr/bin/env python
"""Generate a simple compliance report."""
from argparse import ArgumentParser
from hipaa_compliance_summarizer import ComplianceReporter


def main() -> None:
    parser = ArgumentParser(description="Generate compliance report")
    parser.add_argument("--audit-period", required=True, help="Reporting period")
    parser.add_argument("--documents-processed", type=int, default=0)
    parser.add_argument("--include-recommendations", action="store_true")
    args = parser.parse_args()

    reporter = ComplianceReporter()
    report = reporter.generate_report(
        period=args.audit_period,
        documents_processed=args.documents_processed,
        include_recommendations=args.include_recommendations,
    )
    print(report.overall_compliance)
    print(report.recommendations)


if __name__ == "__main__":
    main()
