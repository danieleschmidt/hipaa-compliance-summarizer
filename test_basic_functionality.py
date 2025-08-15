#!/usr/bin/env python3
"""Basic functionality test for the HIPAA compliance system."""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        print("✓ Core processor imports successful")
    except Exception as e:
        print(f"✗ Core processor import failed: {e}")
        return False
    
    try:
        from hipaa_compliance_summarizer.validation import InputValidator, ValidationLevel
        print("✓ Validation module imports successful")
    except Exception as e:
        print(f"✗ Validation module import failed: {e}")
        return False
    
    try:
        from hipaa_compliance_summarizer.audit_logger import AuditLogger, AuditEventType
        print("✓ Audit logger module imports successful")
    except Exception as e:
        print(f"✗ Audit logger module import failed: {e}")
        return False
    
    try:
        from hipaa_compliance_summarizer.performance_optimized import AdaptiveCache
        print("✓ Performance optimization module imports successful")
    except Exception as e:
        print(f"✗ Performance optimization module import failed: {e}")
        return False
    
    try:
        from hipaa_compliance_summarizer.auto_scaling import AutoScaler, WorkerPool
        print("✓ Auto-scaling module imports successful")
    except Exception as e:
        print(f"✗ Auto-scaling module import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test input validation
        from hipaa_compliance_summarizer.validation import InputValidator, ValidationLevel
        
        validator = InputValidator(ValidationLevel.STANDARD)
        result = validator.validate_text_input("Valid medical text")
        assert result.is_valid
        print("✓ Input validation working")
        
        # Test adaptive cache
        from hipaa_compliance_summarizer.performance_optimized import AdaptiveCache
        
        cache = AdaptiveCache(max_size=10, ttl_seconds=60)
        cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        print("✓ Adaptive cache working")
        
        # Test worker pool
        from hipaa_compliance_summarizer.auto_scaling import WorkerPool
        
        pool = WorkerPool(min_workers=1, max_workers=2)
        assert pool.current_workers == 1
        pool.scale_workers(2)
        assert pool.current_workers == 2
        pool.shutdown()
        print("✓ Worker pool working")
        
        # Test HIPAA processor with simple text
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        
        processor = HIPAAProcessor(ComplianceLevel.STANDARD)
        result = processor.process_document("Patient John Smith, DOB: 01/01/1980")
        assert result.summary is not None
        assert result.compliance_score >= 0.0
        print("✓ HIPAA processor working")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_error_handling():
    """Test error handling capabilities."""
    print("\nTesting error handling...")
    
    try:
        from hipaa_compliance_summarizer.validation import InputValidator, ValidationLevel
        from hipaa_compliance_summarizer.error_handling import ValidationError
        
        # Test validation with invalid input
        validator = InputValidator(ValidationLevel.STRICT)
        result = validator.validate_text_input("<script>alert('xss')</script>")
        assert not result.is_valid
        print("✓ Malicious input detection working")
        
        # Test null input handling
        result = validator.validate_text_input(None)
        assert not result.is_valid
        print("✓ Null input handling working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_audit_logging():
    """Test audit logging functionality."""
    print("\nTesting audit logging...")
    
    try:
        import tempfile
        from hipaa_compliance_summarizer.audit_logger import AuditLogger, AuditEventType, AuditLevel
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            # Log a test event
            event_id = logger.log_event(
                event_type=AuditEventType.DOCUMENT_PROCESSING,
                level=AuditLevel.INFO,
                operation="test_operation"
            )
            
            assert event_id is not None
            assert logger.event_count >= 2  # Session start + test event
            print("✓ Audit logging working")
        
        return True
        
    except Exception as e:
        print(f"✗ Audit logging test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("=== HIPAA Compliance System - Basic Functionality Test ===\n")
    
    tests = [
        test_basic_imports,
        test_basic_functionality,
        test_error_handling,
        test_audit_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        return 0
    else:
        print("❌ Some tests failed. See output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())