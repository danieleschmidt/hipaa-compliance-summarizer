#!/usr/bin/env python3
"""Performance testing for Generation 3 scalability validation."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hipaa_compliance_summarizer import HIPAAProcessor, BatchProcessor

def generate_large_medical_document(size_kb=10):
    """Generate a large medical document for testing."""
    base_text = """
    Patient Name: John Smith
    Social Security Number: 123-45-6789
    Date of Birth: 01/15/1980
    Address: 123 Main Street, Anytown, State 12345
    Phone: (555) 123-4567
    Email: john.smith@email.com
    
    CLINICAL NOTES:
    Patient presented with chest pain and shortness of breath.
    Vital signs: BP 140/90, HR 85, Temperature 98.6F, O2 Sat 98%.
    EKG shows normal sinus rhythm with no acute changes.
    Administered aspirin 325mg and nitroglycerin sublingual.
    Monitored for 4 hours in emergency department.
    Lab results: Troponin <0.01, CK-MB 2.1, BNP 145.
    Patient was discharged in stable condition with follow-up scheduled.
    
    DIAGNOSIS: Chest pain, rule out myocardial infarction
    TREATMENT: Conservative management with monitoring
    DISCHARGE: Home with cardiology follow-up in 1 week
    Medications: Lisinopril 10mg daily, Metoprolol 25mg twice daily
    """
    
    # Repeat to create desired size
    target_chars = size_kb * 1024
    repeat_count = max(1, target_chars // len(base_text))
    return base_text * repeat_count

def test_single_document_performance():
    """Test performance of single document processing."""
    print("=== Single Document Performance Test ===")
    
    processor = HIPAAProcessor(compliance_level='standard')
    
    # Test with different document sizes
    sizes = [1, 5, 10, 25, 50]  # KB
    
    for size in sizes:
        doc = generate_large_medical_document(size)
        
        start = time.time()
        result = processor.process_document(doc)
        duration = time.time() - start
        
        print(f"Size: {size}KB | Time: {duration*1000:.1f}ms | PHI: {result.phi_detected_count} | Score: {result.compliance_score:.2f}")

def test_concurrent_processing():
    """Test concurrent document processing."""
    print("\n=== Concurrent Processing Performance Test ===")
    
    processor = HIPAAProcessor(compliance_level='standard')
    doc = generate_large_medical_document(10)  # 10KB doc
    
    # Test with different thread counts
    thread_counts = [1, 2, 4, 8]
    
    for threads in thread_counts:
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(processor.process_document, doc) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        duration = time.time() - start
        throughput = len(results) / duration
        
        print(f"Threads: {threads} | Total Time: {duration:.2f}s | Throughput: {throughput:.1f} docs/sec")

def test_memory_efficiency():
    """Test memory efficiency with large batches."""
    print("\n=== Memory Efficiency Test ===")
    
    # Create multiple test files
    test_files = []
    for i in range(10):
        filename = f"/tmp/test_med_{i}.txt"
        with open(filename, 'w') as f:
            f.write(generate_large_medical_document(5))  # 5KB each
        test_files.append(filename)
    
    processor = BatchProcessor()
    
    start = time.time()
    results = []
    for file in test_files:
        result = processor.processor.process_document(file)
        results.append(result)
    
    duration = time.time() - start
    
    print(f"Processed {len(results)} files in {duration:.2f}s")
    print(f"Average processing time: {(duration/len(results))*1000:.1f}ms per file")
    
    # Cleanup
    import os
    for file in test_files:
        try:
            os.unlink(file)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Generation 3: Performance & Scalability Testing")
    print("=" * 50)
    
    test_single_document_performance()
    test_concurrent_processing()
    test_memory_efficiency()
    
    print("\n✅ Performance testing completed successfully!")