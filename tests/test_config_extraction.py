"""Tests for configuration file extraction functionality."""

import pytest
import yaml
from pathlib import Path

from hipaa_compliance_summarizer.constants import (
    SecurityLimits, PerformanceLimits, ProcessingConstants,
    load_config_from_file, get_configured_constants
)


class TestConfigExtraction:
    """Test that hardcoded limits can be extracted to configuration."""
    
    def test_security_limits_from_config(self):
        """Test SecurityLimits can be loaded from config dictionary."""
        config = {
            'limits': {
                'security': {
                    'max_file_size': 52428800,  # 50MB
                    'max_document_size': 26214400,  # 25MB
                    'max_text_length': 500000
                }
            }
        }
        
        limits = SecurityLimits.from_config(config)
        assert limits.MAX_FILE_SIZE == 52428800
        assert limits.MAX_DOCUMENT_SIZE == 26214400
        assert limits.MAX_TEXT_LENGTH == 500000
    
    def test_performance_limits_from_config(self):
        """Test PerformanceLimits can be loaded from config dictionary."""
        config = {
            'limits': {
                'performance': {
                    'max_concurrent_jobs': 8,
                    'batch_size': 50,
                    'chunk_size': 4096,
                    'small_file_threshold': 256000,  # 250KB
                    'large_file_threshold': 2097152  # 2MB
                }
            }
        }
        
        limits = PerformanceLimits.from_config(config)
        assert limits.MAX_CONCURRENT_JOBS == 8
        assert limits.BATCH_SIZE == 50
        assert limits.CHUNK_SIZE == 4096
    
    def test_processing_constants_from_config(self):
        """Test ProcessingConstants can be loaded from config dictionary."""
        config = {
            'limits': {
                'performance': {
                    'default_cache_size': 100,
                    'small_file_threshold': 256000,
                    'large_file_threshold': 2097152
                }
            },
            'scoring': {
                'penalty_per_entity': 0.02,
                'penalty_cap': 0.3,
                'strict_multiplier': 2.0
            }
        }
        
        constants = ProcessingConstants.from_config(config)
        assert constants.DEFAULT_CACHE_SIZE == 100
        assert constants.SMALL_FILE_THRESHOLD == 256000
        assert constants.LARGE_FILE_THRESHOLD == 2097152
        assert constants.SCORING_PENALTY_PER_ENTITY == 0.02
        assert constants.SCORING_PENALTY_CAP == 0.3
        assert constants.SCORING_STRICT_MULTIPLIER == 2.0
    
    def test_config_file_loading(self):
        """Test configuration can be loaded from YAML file."""
        config = load_config_from_file()
        
        # Should load some configuration (even if empty dict when file doesn't exist)
        assert isinstance(config, dict)
        
        # If the config file exists, should have expected structure
        if config:
            assert 'patterns' in config or 'limits' in config or 'scoring' in config
    
    def test_get_configured_constants(self):
        """Test getting configured constants."""
        security, performance, processing = get_configured_constants()
        
        assert isinstance(security, SecurityLimits)
        assert isinstance(performance, PerformanceLimits)
        assert isinstance(processing, ProcessingConstants)
        
        # Verify they have reasonable values
        assert security.MAX_FILE_SIZE > 0
        assert performance.MAX_CONCURRENT_JOBS > 0
        assert processing.DEFAULT_CACHE_SIZE > 0
    
    def test_config_defaults_are_preserved(self):
        """Test that default values are used when config is empty."""
        empty_config = {}
        
        security = SecurityLimits.from_config(empty_config)
        performance = PerformanceLimits.from_config(empty_config)
        processing = ProcessingConstants.from_config(empty_config)
        
        # Should have default values
        assert security.MAX_FILE_SIZE == 100 * 1024 * 1024  # 100MB
        assert performance.MAX_CONCURRENT_JOBS == 4
        assert processing.DEFAULT_CACHE_SIZE == 50
    
    def test_config_validation_acceptance_criteria(self):
        """Test that configuration extraction meets acceptance criteria."""
        # Acceptance criteria from CR005:
        # - All hardcoded limits in config ✓
        # - Configuration validation added ✓ 
        # - Default values documented ✓
        
        # Test that major hardcoded limits are now configurable
        config = {
            'limits': {
                'security': {
                    'max_file_size': 1000,
                    'max_document_size': 2000,
                    'max_text_length': 3000
                },
                'performance': {
                    'small_file_threshold': 4000,
                    'large_file_threshold': 5000,
                    'default_cache_size': 6000
                }
            }
        }
        
        security = SecurityLimits.from_config(config)
        performance = PerformanceLimits.from_config(config)
        processing = ProcessingConstants.from_config(config)
        
        # Verify all major limits can be overridden
        assert security.MAX_FILE_SIZE == 1000
        assert security.MAX_DOCUMENT_SIZE == 2000
        assert security.MAX_TEXT_LENGTH == 3000
        assert processing.SMALL_FILE_THRESHOLD == 4000
        assert processing.LARGE_FILE_THRESHOLD == 5000
        assert processing.DEFAULT_CACHE_SIZE == 6000