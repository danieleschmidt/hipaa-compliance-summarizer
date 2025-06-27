#!/usr/bin/env python
"""Example script to batch process healthcare documents."""
from argparse import ArgumentParser
from hipaa_compliance_summarizer import BatchProcessor


def main() -> None:
    parser = ArgumentParser(description="Batch process healthcare documents")
    parser.add_argument("--input-dir", required=True, help="Input directory of documents")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed files")
    parser.add_argument(
        "--compliance-level",
        choices=["strict", "standard", "minimal"],
        default="standard",
        help="Compliance strictness level",
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Write summary files alongside redacted text",
    )
    parser.add_argument(
        "--show-dashboard",
        action="store_true",
        help="Print batch processing summary after completion",
    )
    parser.add_argument(
        "--dashboard-json",
        help="Write dashboard summary to a JSON file",
    )
    args = parser.parse_args()

    processor = BatchProcessor()
    results = processor.process_directory(
        args.input_dir,
        output_dir=args.output_dir,
        compliance_level=args.compliance_level,
        generate_summaries=args.generate_summaries,
    )

    dash = None
    if args.show_dashboard:
        dash = processor.generate_dashboard(results)
        print(dash)

    if args.dashboard_json:
        if dash is None:
            dash = processor.generate_dashboard(results)
        processor.save_dashboard(results, args.dashboard_json)


if __name__ == "__main__":
    main()
