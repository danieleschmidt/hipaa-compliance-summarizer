#!/usr/bin/env python
"""Example script to batch process healthcare documents."""
import logging
import sys
from argparse import ArgumentParser
from hipaa_compliance_summarizer import BatchProcessor
from hipaa_compliance_summarizer.startup import validate_environment, setup_logging_with_config

logger = logging.getLogger(__name__)


def main() -> None:
    # Setup logging first
    setup_logging_with_config()
    
    # Validate environment configuration
    validation = validate_environment(require_production_secrets=False)
    if not validation["valid"]:
        logger.error("Configuration validation failed:")
        for error in validation["errors"]:
            logger.error("  - %s", error)
        sys.exit(1)
    
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)
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
    parser.add_argument(
        "--show-cache-performance",
        action="store_true",
        help="Display cache performance metrics after processing",
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

    if args.show_cache_performance:
        cache_performance = processor.get_cache_performance()
        print("\nCache Performance:")
        print(f"Pattern Compilation - Hits: {cache_performance['pattern_compilation']['hits']}, "
              f"Misses: {cache_performance['pattern_compilation']['misses']}, "
              f"Hit Ratio: {cache_performance['pattern_compilation']['hit_ratio']:.1%}")
        print(f"PHI Detection - Hits: {cache_performance['phi_detection']['hits']}, "
              f"Misses: {cache_performance['phi_detection']['misses']}, "
              f"Hit Ratio: {cache_performance['phi_detection']['hit_ratio']:.1%}")
        print(f"Cache Memory Usage - Pattern: {cache_performance['pattern_compilation']['current_size']}/{cache_performance['pattern_compilation']['max_size']}, "
              f"PHI: {cache_performance['phi_detection']['current_size']}/{cache_performance['phi_detection']['max_size']}")


if __name__ == "__main__":
    main()
