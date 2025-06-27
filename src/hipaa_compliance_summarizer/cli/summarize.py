#!/usr/bin/env python
"""Example script to process a single medical document."""
from argparse import ArgumentParser
from hipaa_compliance_summarizer import HIPAAProcessor


def main() -> None:
    parser = ArgumentParser(description="Summarize a medical document")
    parser.add_argument("--file", required=True, help="Path to the document")
    parser.add_argument(
        "--compliance-level",
        choices=["strict", "standard", "minimal"],
        default="standard",
        help="Compliance strictness level",
    )
    args = parser.parse_args()

    processor = HIPAAProcessor(compliance_level=args.compliance_level)
    result = processor.process_document(args.file)
    print(result.summary)


if __name__ == "__main__":
    main()
