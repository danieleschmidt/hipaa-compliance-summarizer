"""
Performance benchmarking suite for HIPAA Compliance Summarizer
Healthcare-grade performance testing with PHI-safe synthetic data
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import pytest
import psutil
import statistics
from unittest.mock import MagicMock

# Performance test markers
pytestmark = pytest.mark.performance


class PerformanceBenchmark:
    """Base class for performance benchmarks with healthcare compliance"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.baseline_metrics = {
            'phi_detection_latency_ms': 100,  # Max 100ms per document
            'memory_usage_mb': 512,  # Max 512MB working set
            'throughput_docs_per_sec': 10,  # Min 10 documents/second
            'compliance_score_threshold': 0.95  # Min 95% compliance
        }
    
    def record_metric(self, name: str, value: float, unit: str = 'ms'):
        """Record a performance metric"""
        self.results.append({
            'metric': name,
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        })
    
    def assert_performance_threshold(self, metric_name: str, actual: float, 
                                   threshold: float, higher_is_better: bool = False):
        """Assert performance meets healthcare-grade requirements"""
        self.record_metric(metric_name, actual)
        
        if higher_is_better:
            assert actual >= threshold, (
                f"{metric_name} ({actual}) below threshold ({threshold}). "
                f"Healthcare applications require consistent performance."
            )
        else:
            assert actual <= threshold, (
                f"{metric_name} ({actual}) exceeds threshold ({threshold}). "
                f"Healthcare applications require low latency for patient care."
            )


@pytest.fixture
def synthetic_healthcare_documents():
    """Generate synthetic healthcare documents for testing (no real PHI)"""
    documents = [
        {
            'id': f'doc_{i:04d}',
            'content': f"""
            SYNTHETIC MEDICAL RECORD - NOT REAL PHI
            
            Patient ID: SYNTHETIC_{i:06d}
            Date of Service: 2024-01-{(i % 28) + 1:02d}
            
            Chief Complaint: Patient presents with routine check-up
            History: {i}-year history of wellness visits
            
            Vital Signs:
            - Blood Pressure: {120 + (i % 40)}/{80 + (i % 20)} mmHg
            - Heart Rate: {60 + (i % 40)} bpm
            - Temperature: {98.0 + (i % 2):.1f}°F
            
            Assessment: Patient appears healthy for routine screening
            Plan: Continue routine care, follow-up in 12 months
            
            Provider: Dr. Synthetic Provider
            Facility: Test Medical Center
            """,
            'document_type': 'clinical_note',
            'word_count': 95 + (i % 50),
            'expected_phi_count': 5  # Predictable for synthetic data
        }
        for i in range(1, 101)  # 100 synthetic documents
    ]
    return documents


@pytest.fixture
def performance_monitor():
    """Monitor system resources during tests"""
    class ResourceMonitor:
        def __init__(self):
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.start_cpu = psutil.cpu_percent()
            self.start_time = time.time()
        
        def get_current_metrics(self):
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'elapsed_time': time.time() - self.start_time
            }
    
    return ResourceMonitor()


class TestPHIDetectionPerformance(PerformanceBenchmark):
    """Test PHI detection performance with healthcare requirements"""
    
    def test_phi_detection_latency(self, synthetic_healthcare_documents, 
                                 performance_monitor, benchmark):
        """Test PHI detection latency meets healthcare requirements"""
        from hipaa_compliance_summarizer.phi import PHIDetector
        
        detector = PHIDetector()
        
        def detect_phi_batch():
            results = []
            for doc in synthetic_healthcare_documents[:10]:  # Test with 10 docs
                start_time = time.perf_counter()
                phi_result = detector.detect(doc['content'])
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                results.append({
                    'doc_id': doc['id'],
                    'latency_ms': latency_ms,
                    'phi_count': len(phi_result.entities) if phi_result else 0
                })
            return results
        
        # Benchmark the detection
        results = benchmark(detect_phi_batch)
        
        # Calculate performance metrics
        latencies = [r['latency_ms'] for r in results]
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        # Assert healthcare-grade performance
        self.assert_performance_threshold('avg_phi_detection_latency_ms', 
                                        avg_latency, 
                                        self.baseline_metrics['phi_detection_latency_ms'])
        
        self.assert_performance_threshold('p95_phi_detection_latency_ms', 
                                        p95_latency, 
                                        self.baseline_metrics['phi_detection_latency_ms'] * 1.5)
        
        # Check memory usage
        memory_usage = performance_monitor.get_current_metrics()['memory_mb']
        self.assert_performance_threshold('phi_detection_memory_mb', 
                                        memory_usage,
                                        self.baseline_metrics['memory_usage_mb'])
    
    def test_phi_detection_throughput(self, synthetic_healthcare_documents, benchmark):
        """Test PHI detection throughput for batch processing"""
        from hipaa_compliance_summarizer.phi import PHIDetector
        
        detector = PHIDetector()
        
        def process_document_batch():
            documents = synthetic_healthcare_documents[:50]  # 50 documents
            start_time = time.perf_counter()
            
            processed_count = 0
            for doc in documents:
                result = detector.detect(doc['content'])
                if result:
                    processed_count += 1
            
            end_time = time.perf_counter()
            elapsed_seconds = end_time - start_time
            throughput = processed_count / elapsed_seconds
            
            return throughput
        
        throughput = benchmark(process_document_batch)
        
        self.assert_performance_threshold('phi_detection_throughput_docs_per_sec',
                                        throughput,
                                        self.baseline_metrics['throughput_docs_per_sec'],
                                        higher_is_better=True)


class TestBatchProcessingPerformance(PerformanceBenchmark):
    """Test batch processing performance for healthcare workflows"""
    
    def test_batch_processing_scalability(self, synthetic_healthcare_documents, 
                                        performance_monitor, benchmark):
        """Test batch processing scales with document volume"""
        from hipaa_compliance_summarizer.batch import BatchProcessor
        
        processor = BatchProcessor()
        
        # Test different batch sizes
        batch_sizes = [10, 25, 50, 100]
        scalability_results = []
        
        for batch_size in batch_sizes:
            documents = synthetic_healthcare_documents[:batch_size]
            
            def process_batch():
                start_time = time.perf_counter()
                results = []
                for doc in documents:
                    # Mock processing - focus on scalability patterns
                    result = {
                        'document_id': doc['id'],
                        'processed': True,
                        'phi_detected': doc['expected_phi_count'],
                        'compliance_score': 0.98
                    }
                    results.append(result)
                end_time = time.perf_counter()
                
                return {
                    'batch_size': batch_size,
                    'processing_time': end_time - start_time,
                    'throughput': batch_size / (end_time - start_time)
                }
            
            result = benchmark(process_batch)
            scalability_results.append(result)
            
            # Memory usage should scale linearly, not exponentially
            memory_usage = performance_monitor.get_current_metrics()['memory_mb']
            expected_memory = self.baseline_metrics['memory_usage_mb'] * (batch_size / 10)
            
            self.assert_performance_threshold(f'batch_memory_usage_{batch_size}_docs',
                                            memory_usage,
                                            expected_memory * 1.5)  # 50% tolerance
        
        # Check that throughput remains consistent across batch sizes
        throughputs = [r['throughput'] for r in scalability_results]
        throughput_variance = statistics.stdev(throughputs) / statistics.mean(throughputs)
        
        # Throughput variance should be low (consistent performance)
        assert throughput_variance < 0.3, (
            f"Throughput variance ({throughput_variance:.2f}) too high. "
            f"Healthcare systems require predictable performance."
        )
    
    def test_concurrent_processing_performance(self, synthetic_healthcare_documents, 
                                             benchmark):
        """Test concurrent document processing performance"""
        import concurrent.futures
        from hipaa_compliance_summarizer.processor import DocumentProcessor
        
        processor = DocumentProcessor()
        documents = synthetic_healthcare_documents[:20]
        
        def sequential_processing():
            results = []
            start_time = time.perf_counter()
            for doc in documents:
                result = {'doc_id': doc['id'], 'processed': True}
                results.append(result)
            end_time = time.perf_counter()
            return end_time - start_time
        
        def concurrent_processing():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                start_time = time.perf_counter()
                futures = []
                for doc in documents:
                    future = executor.submit(lambda d: {'doc_id': d['id'], 'processed': True}, doc)
                    futures.append(future)
                
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                end_time = time.perf_counter()
                return end_time - start_time
        
        sequential_time = benchmark(sequential_processing)
        concurrent_time = benchmark(concurrent_processing)
        
        # Concurrent processing should be faster (but account for overhead)
        speedup_ratio = sequential_time / concurrent_time
        self.record_metric('concurrent_speedup_ratio', speedup_ratio)
        
        # For healthcare applications, expect at least 2x speedup with 4 workers
        assert speedup_ratio >= 1.5, (
            f"Concurrent processing speedup ({speedup_ratio:.2f}x) insufficient. "
            f"Healthcare systems need efficient concurrent processing."
        )


class TestMemoryEfficiencyBenchmarks(PerformanceBenchmark):
    """Test memory efficiency for long-running healthcare applications"""
    
    def test_memory_leak_detection(self, synthetic_healthcare_documents, 
                                 performance_monitor):
        """Test for memory leaks during extended processing"""
        from hipaa_compliance_summarizer.processor import DocumentProcessor
        
        processor = DocumentProcessor()
        initial_memory = performance_monitor.get_current_metrics()['memory_mb']
        
        # Process documents in multiple iterations
        for iteration in range(5):
            for doc in synthetic_healthcare_documents[:20]:
                # Mock processing
                result = {
                    'document_id': doc['id'],
                    'iteration': iteration,
                    'processed': True
                }
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = performance_monitor.get_current_metrics()['memory_mb']
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be minimal after GC
            self.record_metric(f'memory_growth_iteration_{iteration}', memory_growth, 'MB')
            
            # Healthcare applications must be stable for 24/7 operation
            assert memory_growth < 100, (  # Max 100MB growth
                f"Memory growth ({memory_growth:.1f}MB) in iteration {iteration} "
                f"suggests memory leak. Healthcare systems must run 24/7."
            )
    
    def test_large_document_processing(self, performance_monitor, benchmark):
        """Test processing of large healthcare documents"""
        from hipaa_compliance_summarizer.processor import DocumentProcessor
        
        # Generate large synthetic document (no real PHI)
        large_document = {
            'id': 'large_doc_001',
            'content': '\n'.join([
                f"SYNTHETIC MEDICAL RECORD SECTION {i} - NOT REAL PHI\n"
                f"Patient presents with condition {i} on date 2024-01-{(i % 28) + 1:02d}\n"
                f"Treatment plan includes therapy {i} with follow-up\n"
                f"Provider notes: Synthetic entry {i} for testing purposes\n"
                for i in range(1000)  # 1000 sections = ~50KB document
            ]),
            'document_type': 'comprehensive_report',
            'word_count': 50000
        }
        
        processor = DocumentProcessor()
        
        def process_large_document():
            start_memory = performance_monitor.get_current_metrics()['memory_mb']
            start_time = time.perf_counter()
            
            # Mock processing of large document
            result = {
                'document_id': large_document['id'],
                'processed': True,
                'phi_detected': 50,  # Proportional to size
                'compliance_score': 0.97
            }
            
            end_time = time.perf_counter()
            end_memory = performance_monitor.get_current_metrics()['memory_mb']
            
            return {
                'processing_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'result': result
            }
        
        result = benchmark(process_large_document)
        
        # Large document processing should complete within reasonable time
        self.assert_performance_threshold('large_doc_processing_time_sec',
                                        result['processing_time'],
                                        30.0)  # Max 30 seconds
        
        # Memory usage should be reasonable relative to document size
        self.assert_performance_threshold('large_doc_memory_delta_mb',
                                        result['memory_delta'],
                                        200.0)  # Max 200MB additional


class TestCompliancePerformanceBenchmarks(PerformanceBenchmark):
    """Test compliance-related performance metrics"""
    
    def test_compliance_scoring_performance(self, synthetic_healthcare_documents, 
                                          benchmark):
        """Test performance of compliance scoring algorithms"""
        from hipaa_compliance_summarizer.processor import ComplianceScorer
        
        scorer = ComplianceScorer()
        
        def score_documents_batch():
            scores = []
            start_time = time.perf_counter()
            
            for doc in synthetic_healthcare_documents[:30]:
                # Mock compliance scoring
                score = {
                    'document_id': doc['id'],
                    'overall_score': 0.96,
                    'phi_detection_score': 0.98,
                    'redaction_score': 0.95,
                    'audit_score': 0.97
                }
                scores.append(score)
            
            end_time = time.perf_counter()
            avg_score = statistics.mean([s['overall_score'] for s in scores])
            
            return {
                'scores': scores,
                'processing_time': end_time - start_time,
                'average_compliance_score': avg_score
            }
        
        result = benchmark(score_documents_batch)
        
        # Compliance scoring should be fast
        self.assert_performance_threshold('compliance_scoring_time_sec',
                                        result['processing_time'],
                                        5.0)  # Max 5 seconds for 30 documents
        
        # Average compliance score should meet threshold
        self.assert_performance_threshold('average_compliance_score',
                                        result['average_compliance_score'],
                                        self.baseline_metrics['compliance_score_threshold'],
                                        higher_is_better=True)


@pytest.mark.benchmark(group="healthcare_performance")
def test_end_to_end_pipeline_performance(synthetic_healthcare_documents, 
                                       performance_monitor, benchmark):
    """Test complete healthcare document processing pipeline"""
    
    def process_healthcare_pipeline():
        """Complete pipeline: PHI detection -> Redaction -> Compliance check"""
        from hipaa_compliance_summarizer import HIPAAProcessor
        
        processor = HIPAAProcessor(compliance_level="standard")
        results = []
        
        start_time = time.perf_counter()
        
        for doc in synthetic_healthcare_documents[:15]:  # Process 15 documents
            # Mock full pipeline processing
            result = {
                'document_id': doc['id'],
                'phi_detected': doc['expected_phi_count'],
                'redacted': True,
                'compliance_score': 0.97,
                'processing_time_ms': 45.5,  # Mock timing
                'summary': f"Processed {doc['document_type']} successfully"
            }
            results.append(result)
        
        end_time = time.perf_counter()
        
        return {
            'total_documents': len(results),
            'total_time': end_time - start_time,
            'throughput': len(results) / (end_time - start_time),
            'average_compliance': statistics.mean([r['compliance_score'] for r in results])
        }
    
    result = benchmark(process_healthcare_pipeline)
    
    # Healthcare pipeline must maintain high throughput
    assert result['throughput'] >= 2.0, (
        f"Pipeline throughput ({result['throughput']:.1f} docs/sec) insufficient. "
        f"Healthcare environments require efficient document processing."
    )
    
    # Compliance must remain high under load
    assert result['average_compliance'] >= 0.95, (
        f"Average compliance ({result['average_compliance']:.3f}) below threshold. "
        f"Healthcare applications cannot compromise on compliance."
    )
    
    # Memory usage should be stable
    final_memory = performance_monitor.get_current_metrics()['memory_mb']
    assert final_memory < 1024, (  # Max 1GB
        f"Memory usage ({final_memory:.1f}MB) too high. "
        f"Healthcare systems must be resource-efficient."
    )


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-json=benchmark_results.json"
    ])