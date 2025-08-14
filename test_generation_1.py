#!/usr/bin/env python3
"""Generation 1 testing - MAKE IT WORK verification."""

import sys
import os
sys.path.insert(0, 'src')

def test_core_imports():
    """Test that core modules import successfully."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        from hipaa_compliance_summarizer import BatchProcessor, PHIRedactor
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_processing():
    """Test basic document processing functionality."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, ComplianceLevel
        
        # Test basic processing
        processor = HIPAAProcessor(ComplianceLevel.STANDARD)
        test_text = "Patient John Doe was born on 01/01/1980 and lives at 123 Main St."
        
        result = processor.process_document(test_text)
        
        assert hasattr(result, 'summary')
        assert hasattr(result, 'compliance_score')
        assert hasattr(result, 'phi_detected_count')
        assert hasattr(result, 'redacted')
        
        print(f"✓ Basic processing successful - Score: {result.compliance_score}")
        return True
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return False

def test_batch_processing():
    """Test batch processor initialization."""
    try:
        from hipaa_compliance_summarizer import BatchProcessor, ComplianceLevel
        
        batch_processor = BatchProcessor(ComplianceLevel.STANDARD)
        assert batch_processor is not None
        
        print("✓ Batch processor initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Batch processor failed: {e}")
        return False

def test_cli_tools():
    """Test CLI tool entry points."""
    try:
        # Test that CLI modules can be imported
        from hipaa_compliance_summarizer.cli import summarize
        from hipaa_compliance_summarizer.cli import batch_process
        from hipaa_compliance_summarizer.cli import compliance_report
        
        print("✓ CLI tools import successful")
        return True
        
    except ImportError as e:
        print(f"✗ CLI import failed: {e}")
        return False

def main():
    """Run Generation 1 tests."""
    print("🚀 GENERATION 1: MAKE IT WORK - Testing")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_basic_processing,
        test_batch_processing,
        test_cli_tools,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"GENERATION 1 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ GENERATION 1: MAKE IT WORK - COMPLETE")
        return True
    else:
        print("❌ GENERATION 1: MAKE IT WORK - NEEDS FIXES")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)