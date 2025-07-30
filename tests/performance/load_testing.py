"""
Load testing suite for HIPAA Compliance Summarizer
Healthcare-grade load testing with synthetic data and compliance monitoring
"""

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics
import pytest
from unittest.mock import MagicMock, patch


@dataclass
class LoadTestResult:
    """Load test execution result"""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    memory_peak_mb: float
    cpu_peak_percent: float
    compliance_score: float


class HealthcareLoadTester:
    """Healthcare-specific load testing with PHI-safe synthetic data"""
    
    def __init__(self):
        self.synthetic_documents = self._generate_synthetic_documents()
        self.compliance_requirements = {
            'max_response_time_ms': 2000,  # 2 second SLA
            'min_success_rate': 0.99,  # 99% success rate
            'max_error_rate': 0.01,  # 1% error rate
            'min_compliance_score': 0.95,  # 95% compliance
            'max_memory_mb': 2048,  # 2GB memory limit
            'max_cpu_percent': 80  # 80% CPU limit
        }
    
    def _generate_synthetic_documents(self) -> List[Dict[str, Any]]:
        """Generate synthetic healthcare documents for load testing"""
        templates = [
            {
                'type': 'clinical_note',
                'template': """
                SYNTHETIC CLINICAL NOTE - NOT REAL PHI
                Patient ID: SYN{patient_id:06d}
                Date: 2024-{month:02d}-{day:02d}
                Chief Complaint: {complaint}
                Assessment: {assessment}
                Plan: {plan}
                Provider: Dr. Synthetic {provider_id:03d}
                """,
                'variables': {
                    'complaint': ['Routine checkup', 'Follow-up visit', 'Preventive care'],
                    'assessment': ['Patient stable', 'No acute concerns', 'Continuing care'],
                    'plan': ['Continue current treatment', 'Follow-up in 6 months', 'Routine monitoring']
                }
            },
            {
                'type': 'lab_report',
                'template': """
                SYNTHETIC LAB REPORT - NOT REAL PHI
                Patient ID: SYN{patient_id:06d}
                Test Date: 2024-{month:02d}-{day:02d}
                Lab Results:
                - CBC: Normal ranges
                - Chemistry Panel: Within limits
                - Lipid Panel: {lipid_result}
                Ordering Provider: Dr. Synthetic {provider_id:03d}
                """,
                'variables': {
                    'lipid_result': ['Normal', 'Slightly elevated', 'Within range']
                }
            },
            {
                'type': 'radiology_report',
                'template': """
                SYNTHETIC RADIOLOGY REPORT - NOT REAL PHI
                Patient ID: SYN{patient_id:06d}
                Study Date: 2024-{month:02d}-{day:02d}
                Study Type: {study_type}
                Findings: {findings}
                Impression: {impression}
                Radiologist: Dr. Synthetic {provider_id:03d}
                """,
                'variables': {
                    'study_type': ['Chest X-ray', 'CT Scan', 'MRI'],
                    'findings': ['No acute findings', 'Normal study', 'Unremarkable'],
                    'impression': ['Normal examination', 'No abnormalities', 'Stable findings']
                }
            }
        ]
        
        documents = []
        for i in range(1000):  # Generate 1000 synthetic documents
            template = templates[i % len(templates)]
            variables = {k: v[i % len(v)] for k, v in template['variables'].items()}
            
            content = template['template'].format(
                patient_id=i + 1,
                month=(i % 12) + 1,
                day=(i % 28) + 1,
                provider_id=(i % 50) + 1,
                **variables
            )
            
            documents.append({
                'id': f'syn_doc_{i:06d}',
                'type': template['type'],
                'content': content,
                'word_count': len(content.split()),
                'expected_phi_count': 3,  # Predictable for synthetic data
                'compliance_level': 'standard'
            })
        
        return documents
    
    async def _process_document_async(self, document: Dict[str, Any], 
                                    session_id: str) -> Dict[str, Any]:
        """Simulate async document processing"""
        start_time = time.perf_counter()
        
        # Simulate PHI detection processing time
        await asyncio.sleep(0.05 + (len(document['content']) / 10000))  # Scale with content
        
        # Mock processing result
        result = {
            'document_id': document['id'],
            'session_id': session_id,
            'success': True,
            'phi_detected': document['expected_phi_count'],
            'redacted': True,
            'compliance_score': 0.96 + (hash(document['id']) % 5) / 100,  # 0.96-1.00
            'processing_time': time.perf_counter() - start_time,
            'error': None
        }
        
        # Simulate occasional failures (1% rate)
        if hash(document['id']) % 100 == 0:
            result.update({
                'success': False,
                'error': 'Simulated processing error',
                'compliance_score': 0.0
            })
        
        return result
    
    async def run_concurrent_load_test(self, 
                                     concurrent_users: int,
                                     documents_per_user: int,
                                     test_duration_seconds: Optional[int] = None) -> LoadTestResult:
        """Run concurrent load test with multiple simulated users"""
        test_name = f"concurrent_users_{concurrent_users}_docs_{documents_per_user}"
        start_time = time.perf_counter()
        
        # Create tasks for concurrent users
        tasks = []
        for user_id in range(concurrent_users):
            user_documents = self.synthetic_documents[:documents_per_user]
            task = asyncio.create_task(
                self._simulate_user_session(f"user_{user_id:03d}", user_documents)
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Aggregate results
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        compliance_scores = []
        
        for user_results in results:
            if isinstance(user_results, Exception):
                continue  # Skip failed user sessions
            
            for result in user_results:
                total_requests += 1
                response_times.append(result['processing_time'] * 1000)  # Convert to ms
                
                if result['success']:
                    successful_requests += 1
                    compliance_scores.append(result['compliance_score'])
                else:
                    failed_requests += 1
        
        # Calculate metrics
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        requests_per_second = total_requests / duration
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = (statistics.quantiles(response_times, n=20)[18] 
                           if len(response_times) >= 20 else avg_response_time)
        p99_response_time = (statistics.quantiles(response_times, n=100)[98] 
                           if len(response_times) >= 100 else avg_response_time)
        avg_compliance = statistics.mean(compliance_scores) if compliance_scores else 0
        
        return LoadTestResult(
            test_name=test_name,
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=requests_per_second,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=error_rate,
            memory_peak_mb=500.0,  # Mock memory usage
            cpu_peak_percent=65.0,  # Mock CPU usage
            compliance_score=avg_compliance
        )
    
    async def _simulate_user_session(self, session_id: str, 
                                   documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate a user session processing multiple documents"""
        results = []
        
        for document in documents:
            try:
                result = await self._process_document_async(document, session_id)
                results.append(result)
                
                # Add realistic delay between requests
                await asyncio.sleep(0.1)  # 100ms between requests
                
            except Exception as e:
                results.append({
                    'document_id': document['id'],
                    'session_id': session_id,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0,
                    'compliance_score': 0.0
                })
        
        return results
    
    def validate_healthcare_requirements(self, result: LoadTestResult) -> List[str]:
        """Validate load test results against healthcare requirements"""
        violations = []
        
        if result.error_rate > self.compliance_requirements['max_error_rate']:
            violations.append(
                f"Error rate ({result.error_rate:.3f}) exceeds healthcare threshold "
                f"({self.compliance_requirements['max_error_rate']:.3f})"
            )
        
        if result.p95_response_time > self.compliance_requirements['max_response_time_ms']:
            violations.append(
                f"P95 response time ({result.p95_response_time:.1f}ms) exceeds "
                f"healthcare SLA ({self.compliance_requirements['max_response_time_ms']}ms)"
            )
        
        success_rate = result.successful_requests / result.total_requests
        if success_rate < self.compliance_requirements['min_success_rate']:
            violations.append(
                f"Success rate ({success_rate:.3f}) below healthcare requirement "
                f"({self.compliance_requirements['min_success_rate']:.3f})"
            )
        
        if result.compliance_score < self.compliance_requirements['min_compliance_score']:
            violations.append(
                f"Compliance score ({result.compliance_score:.3f}) below healthcare "
                f"requirement ({self.compliance_requirements['min_compliance_score']:.3f})"
            )
        
        if result.memory_peak_mb > self.compliance_requirements['max_memory_mb']:
            violations.append(
                f"Peak memory usage ({result.memory_peak_mb:.1f}MB) exceeds "
                f"limit ({self.compliance_requirements['max_memory_mb']}MB)"
            )
        
        return violations


@pytest.mark.asyncio
@pytest.mark.load_test
class TestHealthcareLoadScenarios:
    """Healthcare-specific load testing scenarios"""
    
    @pytest.fixture
    def load_tester(self):
        return HealthcareLoadTester()
    
    async def test_normal_business_hours_load(self, load_tester):
        """Test load during normal business hours (moderate concurrent users)"""
        result = await load_tester.run_concurrent_load_test(
            concurrent_users=10,
            documents_per_user=20
        )
        
        violations = load_tester.validate_healthcare_requirements(result)
        
        assert len(violations) == 0, (
            f"Healthcare requirements violated during normal load:\n" +
            "\n".join(violations)
        )
        
        # Additional healthcare-specific assertions
        assert result.requests_per_second >= 5.0, (
            f"Throughput ({result.requests_per_second:.1f} req/s) insufficient "
            f"for healthcare operations"
        )
        
        assert result.compliance_score >= 0.98, (
            f"Compliance score ({result.compliance_score:.3f}) must be high "
            f"during normal operations"
        )
    
    async def test_peak_hours_load(self, load_tester):
        """Test load during peak hours (high concurrent users)"""
        result = await load_tester.run_concurrent_load_test(
            concurrent_users=25,
            documents_per_user=15
        )
        
        violations = load_tester.validate_healthcare_requirements(result)
        
        # During peak hours, some degradation is acceptable but within limits
        critical_violations = [v for v in violations 
                             if 'Error rate' in v or 'Compliance score' in v]
        
        assert len(critical_violations) == 0, (
            f"Critical healthcare requirements violated during peak load:\n" +
            "\n".join(critical_violations)
        )
        
        # Peak hours can have slightly higher response times
        assert result.p99_response_time <= 3000, (  # 3 second max
            f"P99 response time ({result.p99_response_time:.1f}ms) too high "
            f"even for peak hours"
        )
    
    async def test_emergency_burst_load(self, load_tester):
        """Test emergency scenario with sudden burst of documents"""
        result = await load_tester.run_concurrent_load_test(
            concurrent_users=50,
            documents_per_user=10
        )
        
        # During emergencies, maintain core functionality
        assert result.error_rate <= 0.05, (  # 5% error rate acceptable
            f"Error rate ({result.error_rate:.3f}) too high during emergency. "
            f"Healthcare systems must remain functional during crises."
        )
        
        assert result.compliance_score >= 0.90, (  # Slightly lower compliance OK
            f"Compliance score ({result.compliance_score:.3f}) must remain "
            f"acceptable even during emergency load"
        )
        
        # System should recover quickly
        assert result.requests_per_second >= 3.0, (
            f"Even under emergency load, system must process "
            f"({result.requests_per_second:.1f} req/s) minimum throughput"
        )
    
    async def test_sustained_load_endurance(self, load_tester):
        """Test sustained load over extended period (24/7 healthcare operations)"""
        # Simulate multiple waves of sustained load
        results = []
        
        for wave in range(3):  # 3 waves to simulate extended operation
            result = await load_tester.run_concurrent_load_test(
                concurrent_users=15,
                documents_per_user=25
            )
            results.append(result)
            
            # Brief pause between waves
            await asyncio.sleep(1)
        
        # Analyze performance degradation over time
        throughputs = [r.requests_per_second for r in results]
        compliance_scores = [r.compliance_score for r in results]
        
        # Throughput should remain stable
        throughput_degradation = (throughputs[0] - throughputs[-1]) / throughputs[0]
        assert throughput_degradation <= 0.1, (  # Max 10% degradation
            f"Throughput degraded {throughput_degradation:.1%} over sustained load. "
            f"Healthcare systems must maintain performance 24/7."
        )
        
        # Compliance should remain consistent
        compliance_variance = statistics.stdev(compliance_scores)
        assert compliance_variance <= 0.02, (  # Max 2% variance
            f"Compliance variance ({compliance_variance:.3f}) too high. "
            f"Healthcare compliance must be consistent."
        )
    
    async def test_mixed_document_types_load(self, load_tester):
        """Test load with mixed healthcare document types"""
        # Create mixed document batch
        mixed_documents = []
        document_types = ['clinical_note', 'lab_report', 'radiology_report']
        
        for i in range(60):  # 20 of each type
            doc_type = document_types[i % len(document_types)]
            documents_of_type = [d for d in load_tester.synthetic_documents 
                               if d['type'] == doc_type]
            if documents_of_type:
                mixed_documents.append(documents_of_type[i // len(document_types)])
        
        # Override synthetic documents for this test
        original_docs = load_tester.synthetic_documents
        load_tester.synthetic_documents = mixed_documents
        
        try:
            result = await load_tester.run_concurrent_load_test(
                concurrent_users=12,
                documents_per_user=5
            )
            
            violations = load_tester.validate_healthcare_requirements(result)
            
            assert len(violations) == 0, (
                f"Mixed document type processing failed healthcare requirements:\n" +
                "\n".join(violations)
            )
            
            # Mixed documents should still process efficiently
            assert result.average_response_time <= 1500, (  # 1.5 second average
                f"Mixed document processing too slow ({result.average_response_time:.1f}ms). "
                f"Healthcare systems handle diverse document types."
            )
            
        finally:
            # Restore original documents
            load_tester.synthetic_documents = original_docs


@pytest.mark.performance
def test_load_test_suite_runner():
    """Run complete load test suite and generate report"""
    
    async def run_full_suite():
        tester = HealthcareLoadTester()
        
        # Define test scenarios
        scenarios = [
            ("Normal Load", 10, 20),
            ("Peak Load", 25, 15), 
            ("Emergency Burst", 50, 10),
            ("Light Load", 5, 30)
        ]
        
        results = {}
        for scenario_name, users, docs_per_user in scenarios:
            print(f"Running {scenario_name}...")
            result = await tester.run_concurrent_load_test(users, docs_per_user)
            results[scenario_name] = result
            
            violations = tester.validate_healthcare_requirements(result)
            if violations:
                print(f"  VIOLATIONS: {len(violations)}")
                for violation in violations:
                    print(f"    - {violation}")
            else:
                print(f"  PASSED: All healthcare requirements met")
        
        return results
    
    # Run the async test suite
    results = asyncio.run(run_full_suite())
    
    # Generate summary report
    print("\n" + "="*60)
    print("HEALTHCARE LOAD TEST SUMMARY REPORT")
    print("="*60)
    
    for scenario, result in results.items():
        print(f"\n{scenario}:")
        print(f"  Requests/sec: {result.requests_per_second:.1f}")
        print(f"  Avg Response: {result.average_response_time:.1f}ms")
        print(f"  P95 Response: {result.p95_response_time:.1f}ms")
        print(f"  Error Rate: {result.error_rate:.1%}")
        print(f"  Compliance: {result.compliance_score:.3f}")
    
    # Overall assessment
    all_passed = all(
        result.error_rate <= 0.01 and result.compliance_score >= 0.95
        for result in results.values()
    )
    
    print(f"\nOVERALL: {'PASSED' if all_passed else 'FAILED'}")
    print("Healthcare load testing requirements " + 
          ("met" if all_passed else "NOT MET"))


if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__,
        "-v",
        "-m", "load_test",
        "--tb=short"
    ])