#!/usr/bin/env python
"""Example script to batch process healthcare documents."""
import logging
import sys
from argparse import ArgumentParser
from hipaa_compliance_summarizer import BatchProcessor
from hipaa_compliance_summarizer.startup import validate_environment, setup_logging_with_config
from hipaa_compliance_summarizer.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


def _setup_and_validate() -> None:
    """Setup logging and validate environment configuration."""
    setup_logging_with_config()
    
    validation = validate_environment(require_production_secrets=False)
    if not validation["valid"]:
        logger.error("Configuration validation failed:")
        for error in validation["errors"]:
            logger.error("  - %s", error)
        sys.exit(1)
    
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)


def _create_argument_parser() -> ArgumentParser:
    """Create and configure the command-line argument parser."""
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
    parser.add_argument(
        "--show-memory-stats",
        action="store_true",
        help="Display memory usage statistics after processing",
    )
    return parser


def _display_dashboard_output(processor, results, args) -> None:
    """Handle dashboard generation and display."""
    dash = None
    if args.show_dashboard:
        dash = processor.generate_dashboard(results)
        logger.info(dash)

    if args.dashboard_json:
        if dash is None:
            dash = processor.generate_dashboard(results)
        processor.save_dashboard(results, args.dashboard_json)


def _display_cache_performance(processor) -> None:
    """Display cache performance metrics."""
    cache_performance = processor.get_cache_performance()
    logger.info("\nCache Performance:")
    logger.info(f"Pattern Compilation - Hits: {cache_performance['pattern_compilation']['hits']}, "
          f"Misses: {cache_performance['pattern_compilation']['misses']}, "
          f"Hit Ratio: {cache_performance['pattern_compilation']['hit_ratio']:.1%}")
    logger.info(f"PHI Detection - Hits: {cache_performance['phi_detection']['hits']}, "
          f"Misses: {cache_performance['phi_detection']['misses']}, "
          f"Hit Ratio: {cache_performance['phi_detection']['hit_ratio']:.1%}")
    logger.info(f"Cache Memory Usage - Pattern: {cache_performance['pattern_compilation']['current_size']}/{cache_performance['pattern_compilation']['max_size']}, "
          f"PHI: {cache_performance['phi_detection']['current_size']}/{cache_performance['phi_detection']['max_size']}")


def _display_memory_stats(processor) -> None:
    """Display memory usage statistics."""
    memory_stats = processor.get_memory_stats()
    if "error" not in memory_stats:
        logger.info("\nMemory Usage Statistics:")
        logger.info(f"Current Memory Usage: {memory_stats['current_memory_mb']:.1f} MB")
        logger.info(f"Peak Memory Usage: {memory_stats['peak_memory_mb']:.1f} MB") 
        cache_info = memory_stats['cache_memory_usage']
        logger.info(f"File Cache: {cache_info['file_cache_size']}/{cache_info['file_cache_max']} files")
    else:
        logger.error(f"\nMemory stats error: {memory_stats['error']}")


def main() -> None:
    """Entry point for the HIPAA batch processing CLI tool.
    
    Processes multiple healthcare documents in batch with PHI redaction,
    compliance validation, and optional dashboard generation.
    """
    _setup_and_validate()
    
    parser = _create_argument_parser()
    args = parser.parse_args()

    monitor = PerformanceMonitor()  
    processor = BatchProcessor(performance_monitor=monitor)
    results = processor.process_directory(
        args.input_dir,
        output_dir=args.output_dir,
        compliance_level=args.compliance_level,
        generate_summaries=args.generate_summaries,
    )

    _display_dashboard_output(processor, results, args)
    
    if args.show_cache_performance:
        _display_cache_performance(processor)

    if args.show_memory_stats:
        _display_memory_stats(processor)


if __name__ == "__main__":
    main()
