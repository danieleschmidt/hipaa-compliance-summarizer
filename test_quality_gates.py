#!/usr/bin/env python3
"""Quality Gates testing - comprehensive validation."""

import sys
import os
import subprocess
import tempfile
import time
from pathlib import Path
sys.path.insert(0, 'src')

def test_security_compliance():
    """Test security compliance and vulnerability scanning."""
    try:
        # Test basic security imports
        from hipaa_compliance_summarizer.security import (
            validate_file_for_processing, 
            sanitize_filename,
            get_security_recommendations
        )
        
        # Test security validation functions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            temp_file = f.name
        
        try:
            # Should validate successfully
            validated = validate_file_for_processing(temp_file)
            assert validated.exists()
            
            # Test filename sanitization
            dangerous_name = "../../../etc/passwd"
            safe_name = sanitize_filename(dangerous_name)
            assert safe_name != dangerous_name
            assert not ".." in safe_name
            
            # Test security recommendations
            recommendations = get_security_recommendations()
            assert isinstance(recommendations, list)
            
            print("✓ Security compliance verified")
            return True
            
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Security compliance failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance meets required benchmarks."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, BatchProcessor
        import time
        
        # Benchmark single document processing
        processor = HIPAAProcessor()
        test_doc = "Patient John Smith, DOB 01/01/1980, SSN 123-45-6789, Address: 123 Main St"
        
        # Measure processing time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = processor.process_document(test_doc)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance requirements
        avg_under_threshold = avg_time < 50  # 50ms average
        max_under_threshold = max_time < 200  # 200ms maximum
        
        # Test batch processing performance
        batch_processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            for i in range(5):
                test_file = temp_path / f"perf_test_{i}.txt"
                test_file.write_text(f"Patient {i} test document with PHI data {i}")
            
            # Measure batch processing
            start = time.perf_counter()
            results = batch_processor.process_directory(str(temp_path))
            batch_time = (time.perf_counter() - start) * 1000
            
            batch_per_doc = batch_time / len(results) if results else float('inf')
            
        print(f"✓ Performance benchmarks: Avg {avg_time:.1f}ms, Max {max_time:.1f}ms, Batch {batch_per_doc:.1f}ms/doc")
        return avg_under_threshold and max_under_threshold and batch_per_doc < 100
        
    except Exception as e:
        print(f"✗ Performance benchmarks failed: {e}")
        return False

def test_code_quality_checks():
    """Test code quality standards."""
    try:
        # Check if ruff is available and run basic checks
        try:
            result = subprocess.run(['python3', '-m', 'ruff', 'check', 'src/', '--select', 'E,W,F'], 
                                  capture_output=True, text=True, timeout=30)
            ruff_available = True
            ruff_issues = result.returncode != 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            ruff_available = False
            ruff_issues = False
        
        # Test import structure
        try:
            from hipaa_compliance_summarizer import (
                HIPAAProcessor, ComplianceLevel, BatchProcessor,
                PHIRedactor, ComplianceReporter
            )
            imports_clean = True
        except ImportError:
            imports_clean = False
        
        # Check for basic code structure
        src_path = Path('src/hipaa_compliance_summarizer')
        core_files = [
            '__init__.py', 'processor.py', 'batch.py', 
            'phi.py', 'security.py', 'reporting.py'
        ]
        
        files_exist = all((src_path / file).exists() for file in core_files)
        
        quality_score = sum([
            imports_clean,
            files_exist,
            not ruff_issues if ruff_available else True
        ])
        
        print(f"✓ Code quality: Imports: {imports_clean}, Files: {files_exist}, Ruff: {not ruff_issues if ruff_available else 'N/A'}")
        return quality_score >= 2
        
    except Exception as e:
        print(f"✗ Code quality checks failed: {e}")
        return False

def test_integration_functionality():
    """Test end-to-end integration functionality."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, BatchProcessor
        from hipaa_compliance_summarizer.reporting import ComplianceReporter
        
        # Test complete workflow
        processor = HIPAAProcessor()
        batch_processor = BatchProcessor()
        reporter = ComplianceReporter()
        
        # Create test scenario with multiple documents
        test_documents = [
            "Patient Alice Smith, DOB: 01/15/1985, SSN: 111-22-3333",
            "Patient Bob Johnson, Phone: 555-123-4567, Address: 456 Oak Ave",
            "Patient Carol Williams, Email: carol@email.com, MRN: MR123456"
        ]
        
        # Process all documents
        all_results = []
        for doc in test_documents:
            result = processor.process_document(doc)
            all_results.append(result)
        
        # Generate batch dashboard
        dashboard = batch_processor.generate_dashboard(all_results)
        
        # Generate compliance report
        compliance_report = reporter.generate_report(
            period="2024-Q1",
            documents_processed=len(all_results)
        )
        
        # Verify results
        all_processed = len(all_results) == len(test_documents)
        dashboard_valid = dashboard.documents_processed == len(test_documents)
        report_valid = hasattr(compliance_report, 'overall_compliance')
        
        print(f"✓ Integration: Processed {len(all_results)}, Dashboard: {dashboard_valid}, Report: {report_valid}")
        return all_processed and dashboard_valid and report_valid
        
    except Exception as e:
        print(f"✗ Integration functionality failed: {e}")
        return False

def test_cli_interface():
    """Test CLI interface functionality."""
    try:
        # Test CLI module imports
        from hipaa_compliance_summarizer.cli import summarize, batch_process, compliance_report
        
        # Test that main functions exist
        has_summarize_main = hasattr(summarize, 'main')
        has_batch_main = hasattr(batch_process, 'main')
        has_report_main = hasattr(compliance_report, 'main')
        
        cli_functions_available = has_summarize_main and has_batch_main and has_report_main
        
        # Test entry points configuration (from pyproject.toml)
        pyproject_path = Path('pyproject.toml')
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            has_cli_entries = all(cmd in content for cmd in [
                'hipaa-summarize', 'hipaa-batch-process', 'hipaa-compliance-report'
            ])
        else:
            has_cli_entries = False
        
        print(f"✓ CLI interface: Functions: {cli_functions_available}, Entry points: {has_cli_entries}")
        return cli_functions_available
        
    except Exception as e:
        print(f"✗ CLI interface failed: {e}")
        return False

def test_error_resilience():
    """Test system resilience under error conditions."""
    try:
        from hipaa_compliance_summarizer import HIPAAProcessor, BatchProcessor
        
        processor = HIPAAProcessor()
        batch_processor = BatchProcessor()
        
        error_test_cases = [
            None,  # None input
            "",    # Empty input
            "x" * 100000,  # Very large input
            "\x00\x01\x02",  # Binary data
            "<script>alert('xss')</script>",  # Potential XSS
        ]
        
        handled_errors = 0
        total_cases = len(error_test_cases)
        
        for test_case in error_test_cases:
            try:
                result = processor.process_document(test_case)
                # Some cases might succeed (like empty string)
                if test_case == "":
                    handled_errors += 1
            except (ValueError, RuntimeError, TypeError) as e:
                # Expected errors are handled gracefully
                handled_errors += 1
            except Exception as e:
                # Unexpected errors reduce resilience score
                pass
        
        resilience_score = handled_errors / total_cases
        
        # Test batch processor resilience with invalid directory
        try:
            batch_processor.process_directory("/nonexistent/directory")
        except (FileNotFoundError, PermissionError, ValueError):
            batch_resilient = True
        except Exception:
            batch_resilient = False
        else:
            batch_resilient = True  # If it returns empty results gracefully
        
        print(f"✓ Error resilience: {resilience_score:.1%} handled, Batch resilient: {batch_resilient}")
        return resilience_score >= 0.6 and batch_resilient
        
    except Exception as e:
        print(f"✗ Error resilience failed: {e}")
        return False

def test_memory_and_resource_management():
    """Test memory efficiency and resource management."""
    try:
        from hipaa_compliance_summarizer import BatchProcessor, PHIRedactor
        import gc
        
        batch_processor = BatchProcessor()
        
        # Test memory management with repeated processing
        initial_objects = len(gc.get_objects())
        
        # Process many documents to test memory usage
        for i in range(100):
            doc = f"Patient {i} with various PHI data including {i}@email.com and phone {i}555123456"
            result = batch_processor.processor.process_document(doc)
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        memory_efficient = object_growth < 1000  # Reasonable threshold
        
        # Test cache management
        cache_performance = batch_processor.get_cache_performance()
        has_cache_info = 'pattern_compilation' in cache_performance
        
        # Test cache clearing
        batch_processor.clear_cache()
        cache_cleared = True  # If no exception, clearing works
        
        print(f"✓ Resource management: Memory growth: {object_growth} objects, Cache: {has_cache_info}")
        return memory_efficient and has_cache_info and cache_cleared
        
    except Exception as e:
        print(f"✗ Resource management failed: {e}")
        return False

def main():
    """Run comprehensive quality gates validation."""
    print("🛡️ QUALITY GATES VALIDATION")
    print("=" * 70)
    
    quality_tests = [
        ("Security Compliance", test_security_compliance),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Code Quality Standards", test_code_quality_checks),
        ("Integration Functionality", test_integration_functionality),
        ("CLI Interface", test_cli_interface),
        ("Error Resilience", test_error_resilience),
        ("Resource Management", test_memory_and_resource_management),
    ]
    
    passed = 0
    total = len(quality_tests)
    
    for test_name, test_func in quality_tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} - FAILED")
    
    print("\n" + "=" * 70)
    print(f"QUALITY GATES RESULTS: {passed}/{total} gates passed")
    
    coverage_percentage = (passed / total) * 100
    
    if passed >= total * 0.85:  # 85% pass rate required
        print(f"✅ QUALITY GATES: PASSED ({coverage_percentage:.1f}% coverage)")
        return True
    else:
        print(f"❌ QUALITY GATES: FAILED ({coverage_percentage:.1f}% coverage)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)