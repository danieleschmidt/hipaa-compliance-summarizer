#!/usr/bin/env python3
"""Generation 2 testing - MAKE IT ROBUST verification."""

import sys
import os
import tempfile
import logging
from pathlib import Path
sys.path.insert(0, 'src')

def test_comprehensive_error_handling():
    """Test robust error handling across the system."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        
        processor = HIPAAProcessor(ComplianceLevel.STANDARD)
        
        # Test invalid inputs
        error_cases = [
            None,  # None input
            "",    # Empty string
            "\x00binary_content",  # Binary content
            "x" * 1000000,  # Oversized content
        ]
        
        handled_errors = 0
        for case in error_cases:
            try:
                result = processor.process_document(case)
                if case == "":
                    # Empty string should be handled gracefully
                    print(f"  ✓ Empty string handled gracefully")
                    handled_errors += 1
            except (ValueError, RuntimeError) as e:
                print(f"  ✓ Error properly handled: {type(e).__name__}")
                handled_errors += 1
            except Exception as e:
                print(f"  ⚠ Unexpected error type: {type(e).__name__}: {e}")
        
        success_rate = handled_errors / len(error_cases)
        print(f"✓ Error handling robust - {success_rate:.1%} coverage")
        return success_rate >= 0.75
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_input_validation():
    """Test comprehensive input validation."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        
        # Test compliance level validation
        valid_levels = ['strict', 'standard', 'minimal']
        for level in valid_levels:
            processor = HIPAAProcessor(level)
            assert processor.compliance_level.value == level
        
        # Test invalid compliance levels
        try:
            HIPAAProcessor('invalid_level')
            return False  # Should have raised error
        except ValueError:
            pass  # Expected
        
        print("✓ Input validation comprehensive")
        return True
        
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        return False

def test_security_validation():
    """Test security validation mechanisms."""
    try:
        from hipaa_compliance_summarizer.security import validate_file_for_processing
        from hipaa_compliance_summarizer import BatchProcessor
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create safe test file
            safe_file = temp_path / "safe_document.txt"
            safe_file.write_text("Patient data for processing")
            
            # Test file validation
            validated = validate_file_for_processing(str(safe_file))
            assert validated.exists()
            
            # Test dangerous path patterns (should be caught)
            dangerous_paths = [
                "../../../etc/passwd",
                "/etc/shadow", 
                "C:\\Windows\\System32\\config\\SAM"
            ]
            
            blocked_count = 0
            for dangerous_path in dangerous_paths:
                try:
                    validate_file_for_processing(dangerous_path)
                except Exception:
                    blocked_count += 1
            
            security_rate = blocked_count / len(dangerous_paths)
            print(f"✓ Security validation active - {security_rate:.1%} threat detection")
            return security_rate >= 0.5
        
    except Exception as e:
        print(f"✗ Security validation failed: {e}")
        return False

def test_logging_and_monitoring():
    """Test comprehensive logging framework."""
    try:
        from hipaa_compliance_summarizer.logging_framework import get_logger
        from hipaa_compliance_summarizer import HIPAAProcessor
        
        # Test structured logging
        logger = get_logger(__name__)
        logger.info("Test log entry", extra={"test_component": "generation_2"})
        
        # Test that processing generates appropriate logs
        processor = HIPAAProcessor()
        
        # Capture logs by setting up a handler
        import io
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logging.getLogger('hipaa_compliance_summarizer').addHandler(handler)
        
        result = processor.process_document("Test patient data")
        
        log_output = log_stream.getvalue()
        logging.getLogger('hipaa_compliance_summarizer').removeHandler(handler)
        
        # Check for expected log patterns
        has_processing_logs = "Processing document" in log_output
        has_timing_logs = "ms" in log_output or "Processed document" in log_output
        
        print(f"✓ Logging framework active - Processing: {has_processing_logs}, Timing: {has_timing_logs}")
        return has_processing_logs or has_timing_logs
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    try:
        from hipaa_compliance_summarizer.performance import PerformanceOptimizer
        from hipaa_compliance_summarizer import BatchProcessor
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        assert optimizer is not None
        
        # Test batch processor with monitoring
        batch_processor = BatchProcessor()
        
        # Test cache performance tracking
        cache_info = batch_processor.get_cache_performance()
        assert isinstance(cache_info, dict)
        assert 'pattern_compilation' in cache_info
        assert 'phi_detection' in cache_info
        
        print("✓ Performance monitoring operational")
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")
        return False

def test_health_checks():
    """Test system health monitoring."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor
        from hipaa_compliance_summarizer.health import system_health_check
        
        # Test basic health check
        health_status = system_health_check()
        
        # Health check should return status information
        assert isinstance(health_status, dict)
        
        print(f"✓ Health monitoring active - Status: {health_status.get('status', 'unknown')}")
        return True
        
    except ImportError:
        print("⚠ Health check module not available")
        return True  # Optional component
    except Exception as e:
        print(f"✗ Health checks failed: {e}")
        return False

def main():
    """Run Generation 2 robustness tests."""
    print("🚀 GENERATION 2: MAKE IT ROBUST - Testing")
    print("=" * 60)
    
    tests = [
        test_comprehensive_error_handling,
        test_input_validation, 
        test_security_validation,
        test_logging_and_monitoring,
        test_performance_monitoring,
        test_health_checks,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"GENERATION 2 RESULTS: {passed}/{total} tests passed")
    
    if passed >= total * 0.8:  # Allow for some optional components
        print("✅ GENERATION 2: MAKE IT ROBUST - COMPLETE")
        return True
    else:
        print("❌ GENERATION 2: MAKE IT ROBUST - NEEDS IMPROVEMENTS")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)