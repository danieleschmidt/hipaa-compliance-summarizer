"""Application constants and configuration values.

This module centralizes hardcoded values, limits, and configuration constants
to improve maintainability and reduce magic numbers throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SecurityLimits:
    """Security-related limits and thresholds."""

    # File size limits (in bytes)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB default
    MAX_FILE_SIZE_LARGE: int = 200 * 1024 * 1024  # 200MB for large files

    # Path limits
    MAX_PATH_LENGTH: int = 4096
    MAX_FILENAME_LENGTH: int = 255

    # Content limits
    MAX_DOCUMENT_SIZE: int = 50 * 1024 * 1024  # 50MB for document processing
    MAX_TEXT_LENGTH: int = 1_000_000  # 1M characters

    @classmethod
    def from_environment(cls) -> SecurityLimits:
        """Create SecurityLimits from environment variables."""
        return cls(
            MAX_FILE_SIZE=int(os.environ.get('HIPAA_MAX_FILE_SIZE', cls.MAX_FILE_SIZE)),
            MAX_FILE_SIZE_LARGE=int(os.environ.get('HIPAA_MAX_FILE_SIZE_LARGE', cls.MAX_FILE_SIZE_LARGE)),
            MAX_PATH_LENGTH=int(os.environ.get('HIPAA_MAX_PATH_LENGTH', cls.MAX_PATH_LENGTH)),
            MAX_FILENAME_LENGTH=int(os.environ.get('HIPAA_MAX_FILENAME_LENGTH', cls.MAX_FILENAME_LENGTH)),
            MAX_DOCUMENT_SIZE=int(os.environ.get('HIPAA_MAX_DOCUMENT_SIZE', cls.MAX_DOCUMENT_SIZE)),
            MAX_TEXT_LENGTH=int(os.environ.get('HIPAA_MAX_TEXT_LENGTH', cls.MAX_TEXT_LENGTH)),
        )

    @classmethod
    def from_config(cls, config_dict: dict) -> SecurityLimits:
        """Create SecurityLimits from configuration dictionary."""
        security_limits = config_dict.get('limits', {}).get('security', {})
        return cls(
            MAX_FILE_SIZE=security_limits.get('max_file_size', cls.MAX_FILE_SIZE),
            MAX_FILE_SIZE_LARGE=security_limits.get('max_file_size_large', cls.MAX_FILE_SIZE_LARGE),
            MAX_PATH_LENGTH=security_limits.get('max_path_length', cls.MAX_PATH_LENGTH),
            MAX_FILENAME_LENGTH=security_limits.get('max_filename_length', cls.MAX_FILENAME_LENGTH),
            MAX_DOCUMENT_SIZE=security_limits.get('max_document_size', cls.MAX_DOCUMENT_SIZE),
            MAX_TEXT_LENGTH=security_limits.get('max_text_length', cls.MAX_TEXT_LENGTH),
        )


@dataclass
class PerformanceLimits:
    """Performance-related limits and thresholds."""

    # Processing limits
    MAX_CONCURRENT_JOBS: int = 4
    BATCH_SIZE: int = 100
    CHUNK_SIZE: int = 8192  # For file I/O
    DEFAULT_READ_CHUNK_SIZE: int = 4096  # Default chunk size for file reading

    # Cache limits
    CACHE_MAX_SIZE: int = 1000  # Max cached items
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # Timeout values (in seconds)
    NETWORK_TIMEOUT: int = 30
    PROCESSING_TIMEOUT: int = 300  # 5 minutes

    @classmethod
    def from_environment(cls) -> PerformanceLimits:
        """Create PerformanceLimits from environment variables."""
        return cls(
            MAX_CONCURRENT_JOBS=int(os.environ.get('HIPAA_MAX_CONCURRENT_JOBS', cls.MAX_CONCURRENT_JOBS)),
            BATCH_SIZE=int(os.environ.get('HIPAA_BATCH_SIZE', cls.BATCH_SIZE)),
            CHUNK_SIZE=int(os.environ.get('HIPAA_CHUNK_SIZE', cls.CHUNK_SIZE)),
            DEFAULT_READ_CHUNK_SIZE=int(os.environ.get('HIPAA_DEFAULT_READ_CHUNK_SIZE', cls.DEFAULT_READ_CHUNK_SIZE)),
            CACHE_MAX_SIZE=int(os.environ.get('HIPAA_CACHE_MAX_SIZE', cls.CACHE_MAX_SIZE)),
            CACHE_TTL_SECONDS=int(os.environ.get('HIPAA_CACHE_TTL_SECONDS', cls.CACHE_TTL_SECONDS)),
            NETWORK_TIMEOUT=int(os.environ.get('HIPAA_NETWORK_TIMEOUT', cls.NETWORK_TIMEOUT)),
            PROCESSING_TIMEOUT=int(os.environ.get('HIPAA_PROCESSING_TIMEOUT', cls.PROCESSING_TIMEOUT)),
        )

    @classmethod
    def from_config(cls, config_dict: dict) -> PerformanceLimits:
        """Create PerformanceLimits from configuration dictionary."""
        perf_limits = config_dict.get('limits', {}).get('performance', {})
        return cls(
            MAX_CONCURRENT_JOBS=perf_limits.get('max_concurrent_jobs', cls.MAX_CONCURRENT_JOBS),
            BATCH_SIZE=perf_limits.get('batch_size', cls.BATCH_SIZE),
            CHUNK_SIZE=perf_limits.get('chunk_size', cls.CHUNK_SIZE),
            DEFAULT_READ_CHUNK_SIZE=perf_limits.get('default_read_chunk_size', cls.DEFAULT_READ_CHUNK_SIZE),
            CACHE_MAX_SIZE=perf_limits.get('cache_max_size', cls.CACHE_MAX_SIZE),
            CACHE_TTL_SECONDS=perf_limits.get('cache_ttl_seconds', cls.CACHE_TTL_SECONDS),
            NETWORK_TIMEOUT=perf_limits.get('network_timeout', cls.NETWORK_TIMEOUT),
            PROCESSING_TIMEOUT=perf_limits.get('processing_timeout', cls.PROCESSING_TIMEOUT),
        )


@dataclass
class ProcessingConstants:
    """Constants for document processing and summarization."""

    # Text summarization widths
    SUMMARY_WIDTH_STRICT: int = 400      # Characters for strict compliance
    SUMMARY_WIDTH_STANDARD: int = 600    # Characters for standard compliance

    # Scoring constants
    SCORING_PENALTY_PER_ENTITY: float = 0.01   # Penalty per PHI entity found
    SCORING_PENALTY_CAP: float = 0.2           # Maximum penalty cap
    SCORING_STRICT_MULTIPLIER: float = 1.5     # Multiplier for strict compliance

    # Cache and file handling
    DEFAULT_CACHE_SIZE: int = 50               # Maximum files in cache
    SMALL_FILE_THRESHOLD: int = 512 * 1024    # 512KB threshold for small files
    LARGE_FILE_THRESHOLD: int = 1024 * 1024   # 1MB threshold for memory mapping

    # Processing intervals
    PROGRESS_REPORT_INTERVAL: int = 10         # Report progress every N files
    DEFAULT_WORKERS: int = 4                   # Default max workers

    # Control character thresholds
    CONTROL_CHAR_THRESHOLD: int = 32           # ASCII threshold for control chars

    @classmethod
    def from_environment(cls) -> ProcessingConstants:
        """Create ProcessingConstants from environment variables."""
        return cls(
            SUMMARY_WIDTH_STRICT=int(os.environ.get('HIPAA_SUMMARY_WIDTH_STRICT', cls.SUMMARY_WIDTH_STRICT)),
            SUMMARY_WIDTH_STANDARD=int(os.environ.get('HIPAA_SUMMARY_WIDTH_STANDARD', cls.SUMMARY_WIDTH_STANDARD)),
            SCORING_PENALTY_PER_ENTITY=float(os.environ.get('HIPAA_SCORING_PENALTY_PER_ENTITY', cls.SCORING_PENALTY_PER_ENTITY)),
            SCORING_PENALTY_CAP=float(os.environ.get('HIPAA_SCORING_PENALTY_CAP', cls.SCORING_PENALTY_CAP)),
            SCORING_STRICT_MULTIPLIER=float(os.environ.get('HIPAA_SCORING_STRICT_MULTIPLIER', cls.SCORING_STRICT_MULTIPLIER)),
            DEFAULT_CACHE_SIZE=int(os.environ.get('HIPAA_DEFAULT_CACHE_SIZE', cls.DEFAULT_CACHE_SIZE)),
            SMALL_FILE_THRESHOLD=int(os.environ.get('HIPAA_SMALL_FILE_THRESHOLD', cls.SMALL_FILE_THRESHOLD)),
            LARGE_FILE_THRESHOLD=int(os.environ.get('HIPAA_LARGE_FILE_THRESHOLD', cls.LARGE_FILE_THRESHOLD)),
            PROGRESS_REPORT_INTERVAL=int(os.environ.get('HIPAA_PROGRESS_REPORT_INTERVAL', cls.PROGRESS_REPORT_INTERVAL)),
            DEFAULT_WORKERS=int(os.environ.get('HIPAA_DEFAULT_WORKERS', cls.DEFAULT_WORKERS)),
            CONTROL_CHAR_THRESHOLD=int(os.environ.get('HIPAA_CONTROL_CHAR_THRESHOLD', cls.CONTROL_CHAR_THRESHOLD)),
        )

    @classmethod
    def from_config(cls, config_dict: dict) -> ProcessingConstants:
        """Create ProcessingConstants from configuration dictionary."""
        perf_limits = config_dict.get('limits', {}).get('performance', {})
        scoring = config_dict.get('scoring', {})
        return cls(
            SUMMARY_WIDTH_STRICT=cls.SUMMARY_WIDTH_STRICT,  # Not configurable yet
            SUMMARY_WIDTH_STANDARD=cls.SUMMARY_WIDTH_STANDARD,  # Not configurable yet
            SCORING_PENALTY_PER_ENTITY=scoring.get('penalty_per_entity', cls.SCORING_PENALTY_PER_ENTITY),
            SCORING_PENALTY_CAP=scoring.get('penalty_cap', cls.SCORING_PENALTY_CAP),
            SCORING_STRICT_MULTIPLIER=scoring.get('strict_multiplier', cls.SCORING_STRICT_MULTIPLIER),
            DEFAULT_CACHE_SIZE=perf_limits.get('default_cache_size', cls.DEFAULT_CACHE_SIZE),
            SMALL_FILE_THRESHOLD=perf_limits.get('small_file_threshold', cls.SMALL_FILE_THRESHOLD),
            LARGE_FILE_THRESHOLD=perf_limits.get('large_file_threshold', cls.LARGE_FILE_THRESHOLD),
            PROGRESS_REPORT_INTERVAL=cls.PROGRESS_REPORT_INTERVAL,  # Not configurable yet
            DEFAULT_WORKERS=cls.DEFAULT_WORKERS,  # Not configurable yet
            CONTROL_CHAR_THRESHOLD=cls.CONTROL_CHAR_THRESHOLD,  # Not configurable yet
        )


@dataclass
class TestConstants:
    """Constants used in testing."""

    # Test data patterns
    TEST_SSN_PATTERN: str = "123-45-6789"
    TEST_SSN_PATTERN_ALT: str = "987-65-4321"
    TEST_PHONE_PATTERN: str = "555-123-4567"
    TEST_EMAIL_PATTERN: str = "patient@hospital.com"
    TEST_EMAIL_PATTERN_ALT: str = "john@example.com"

    # Test database URLs (for testing only)
    TEST_POSTGRES_URL: str = "postgresql://user:pass@localhost/db"
    TEST_REDIS_URL: str = "redis://localhost:6379/0"

    # Test file sizes
    SMALL_FILE_SIZE: int = 1024  # 1KB
    MEDIUM_FILE_SIZE: int = 1024 * 1024  # 1MB
    LARGE_FILE_SIZE: int = 60000  # For generating ~1.2MB content
    TEST_LARGE_TEXT_SIZE: int = 51 * 1024 * 1024  # 51MB for testing large documents
    TEST_VERY_LARGE_SIZE: int = 105 * 1024 * 1024  # 105MB for testing size limits
    TEST_MODERATE_SIZE: int = 10 * 1024 * 1024    # 10MB for error handling tests

    # Test performance metrics
    TEST_DOCS_PROCESSED: int = 100
    TEST_PROCESSING_TIME: float = 250.0
    TEST_PHI_DETECTED: int = 1500
    TEST_MEMORY_USAGE: float = 1024.0  # MB
    TEST_DISK_IO: float = 2048.0  # MB


# Unit conversion constants
BYTES_PER_KB: int = 1024
BYTES_PER_MB: int = 1024 * 1024
BYTES_PER_GB: int = 1024 * 1024 * 1024

# Legacy constants for backward compatibility
MAX_FILE_SIZE_51MB = 51 * BYTES_PER_MB  # For test compatibility

def load_config_from_file(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, searches for config in default locations.
        
    Returns:
        Configuration dictionary, empty if file not found or error loading.
    """
    if config_path is None:
        # Search for config in default locations
        default_paths = [
            Path("config/hipaa_config.yml"),
            Path("../config/hipaa_config.yml"),
            Path("../../config/hipaa_config.yml"),
        ]

        for path in default_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError, PermissionError):
        return {}


def get_configured_constants(config_dict: Optional[dict] = None):
    """Get constants configured from file and environment.
    
    Args:
        config_dict: Configuration dictionary. If None, loads from default file locations.
        
    Returns:
        Tuple of (security_limits, performance_limits, processing_constants)
    """
    if config_dict is None:
        config_dict = load_config_from_file()

    # Create instances from config, with environment taking precedence
    security_limits = SecurityLimits.from_config(config_dict)
    security_limits = SecurityLimits.from_environment()  # Override with env vars

    performance_limits = PerformanceLimits.from_config(config_dict)
    performance_limits = PerformanceLimits.from_environment()  # Override with env vars

    processing_constants = ProcessingConstants.from_config(config_dict)
    processing_constants = ProcessingConstants.from_environment()  # Override with env vars

    return security_limits, performance_limits, processing_constants


# Global instances - can be overridden by calling get_configured_constants()
SECURITY_LIMITS = SecurityLimits.from_environment()
PERFORMANCE_LIMITS = PerformanceLimits.from_environment()
PROCESSING_CONSTANTS = ProcessingConstants.from_environment()
TEST_CONSTANTS = TestConstants()

# Legacy constants for backward compatibility
MAX_FILE_SIZE = SECURITY_LIMITS.MAX_FILE_SIZE
MAX_PATH_LENGTH = SECURITY_LIMITS.MAX_PATH_LENGTH
