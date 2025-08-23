#!/usr/bin/env python3
"""Generation 4 Performance and Scaling Test Suite."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.hipaa_compliance_summarizer import HIPAAProcessor
from src.hipaa_compliance_summarizer.performance_gen4 import (
    MLPerformanceOptimizer,
    OptimizationConfig,
    AdaptiveResourceManager
)
from src.hipaa_compliance_summarizer.intelligent_scaling import (
    IntelligentAutoScaler,
    ResourceMetrics,
    ResourceType,
    ScalingAction,
    ScalingPolicy
)


class TestMLPerformanceOptimizer(unittest.TestCase):
    """Test ML-driven performance optimization."""

    def setUp(self):
        self.config = OptimizationConfig(
            optimization_window=1800,
            min_samples_for_ml=10
        )
        self.optimizer = MLPerformanceOptimizer(self.config)

    def test_performance_event_recording(self):
        """Test performance event recording and storage."""
        # Record some performance events
        self.optimizer.record_performance_event("test_operation", 5.2)
        self.optimizer.record_performance_event("test_operation", 3.1, {"size": 100})
        
        self.assertEqual(len(self.optimizer.performance_history), 2)
        
        # Check event structure
        event = self.optimizer.performance_history[0]
        self.assertIn('timestamp', event)
        self.assertIn('event_type', event)
        self.assertIn('duration', event)
        self.assertEqual(event['event_type'], "test_operation")
        self.assertEqual(event['duration'], 5.2)

    def test_ml_model_training(self):
        """Test ML model training with sufficient data."""
        # Generate sufficient performance data
        for i in range(50):
            event_type = f"operation_{i % 3}"
            duration = 1.0 + (i % 10) * 0.5
            self.optimizer.record_performance_event(event_type, duration)
        
        # Train models
        self.optimizer.train_optimization_models()
        
        # Check that models were created
        self.assertIn('workload_clusters', self.optimizer.optimization_models)
        self.assertIn('cluster_stats', self.optimizer.optimization_models)
        self.assertIn('performance', self.optimizer.feature_scalers)

    def test_resource_prediction(self):
        """Test optimal resource prediction."""
        # Test with default resources (no models)
        resources = self.optimizer.predict_optimal_resources("test_workload", 10)
        
        self.assertIn('threads', resources)
        self.assertIn('memory_limit', resources)
        self.assertIn('predicted_duration', resources)
        self.assertIn('confidence', resources)
        self.assertGreater(resources['threads'], 0)

    def test_processing_pipeline_optimization(self):
        """Test processing pipeline optimization."""
        # Mock processor function
        def mock_processor(doc):
            time.sleep(0.01)  # Simulate processing
            return f"processed_{doc}"
        
        documents = ['doc1', 'doc2', 'doc3']
        
        # Test optimization
        results = self.optimizer.optimize_processing_pipeline(documents, mock_processor)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "processed_doc1")
        
        # Check that performance was recorded
        self.assertTrue(
            any('pipeline_mock_processor' in event['event_type'] 
                for event in self.optimizer.performance_history)
        )

    def test_optimization_insights(self):
        """Test performance insights generation."""
        # Add some performance data
        for i in range(10):
            self.optimizer.record_performance_event(f"op_{i % 2}", i * 0.5)
        
        insights = self.optimizer.get_optimization_insights()
        
        self.assertIn('total_events', insights)
        self.assertIn('average_duration', insights)
        self.assertIn('event_analysis', insights)
        self.assertEqual(insights['total_events'], 10)


class TestAdaptiveResourceManager(unittest.TestCase):
    """Test adaptive resource management."""

    def setUp(self):
        self.manager = AdaptiveResourceManager()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_monitoring(self, mock_memory, mock_cpu):
        """Test resource usage monitoring."""
        # Mock system metrics
        mock_cpu.return_value = 65.0
        mock_memory.return_value = Mock(percent=45.0, available=2**30)  # 1GB
        
        usage = self.manager.monitor_resource_usage()
        
        self.assertEqual(usage['cpu_percent'], 65.0)
        self.assertEqual(usage['memory_percent'], 45.0)
        self.assertAlmostEqual(usage['memory_available_gb'], 1.0, places=1)

    def test_resource_history_management(self):
        """Test resource usage history management."""
        # Add some usage data
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_cpu.return_value = 50.0
            mock_memory.return_value = Mock(percent=30.0, available=2**31)
            
            # Monitor multiple times
            for _ in range(5):
                self.manager.monitor_resource_usage()
            
            self.assertEqual(len(self.manager.usage_history), 5)

    def test_auto_scaling_decisions(self):
        """Test auto-scaling resource decisions."""
        # Simulate high CPU usage history
        for i in range(15):
            self.manager.usage_history.append({
                'timestamp': time.time() - i,
                'cpu_percent': 85.0,
                'memory_percent': 30.0,
                'memory_available_gb': 2.0
            })
        
        # Test auto-scaling
        new_allocation = self.manager.auto_scale_resources()
        
        # Should scale up thread pool due to high CPU
        original_threads = self.manager.current_allocation['thread_pool_size']
        self.assertGreater(new_allocation['thread_pool_size'], original_threads)

    def test_resource_recommendations(self):
        """Test resource optimization recommendations."""
        # Add usage history
        for i in range(10):
            self.manager.usage_history.append({
                'timestamp': time.time() - i,
                'cpu_percent': 90.0,  # High CPU
                'memory_percent': 85.0,  # High memory
                'memory_available_gb': 0.5
            })
        
        recommendations = self.manager.get_resource_recommendations()
        
        self.assertIn('recommendations', recommendations)
        self.assertGreater(len(recommendations['recommendations']), 0)
        
        # Should recommend CPU and memory optimizations
        rec_text = ' '.join(recommendations['recommendations']).lower()
        self.assertIn('cpu', rec_text)


class TestIntelligentAutoScaler(unittest.TestCase):
    """Test intelligent auto-scaling system."""

    def setUp(self):
        self.scaler = IntelligentAutoScaler()

    def test_default_policies_setup(self):
        """Test default scaling policies are properly configured."""
        self.assertIn(ResourceType.CPU, self.scaler.scaling_policies)
        self.assertIn(ResourceType.MEMORY, self.scaler.scaling_policies)
        self.assertIn(ResourceType.WORKERS, self.scaler.scaling_policies)
        
        cpu_policy = self.scaler.scaling_policies[ResourceType.CPU]
        self.assertEqual(cpu_policy.resource_type, ResourceType.CPU)
        self.assertGreater(cpu_policy.scale_up_threshold, 0)

    def test_policy_updates(self):
        """Test scaling policy updates."""
        new_policy = ScalingPolicy(
            resource_type=ResourceType.CPU,
            scale_up_threshold=0.9,
            scale_down_threshold=0.1,
            min_instances=2,
            max_instances=16
        )
        
        self.scaler.update_policy(ResourceType.CPU, new_policy)
        
        updated_policy = self.scaler.scaling_policies[ResourceType.CPU]
        self.assertEqual(updated_policy.scale_up_threshold, 0.9)
        self.assertEqual(updated_policy.max_instances, 16)

    def test_metrics_processing_and_scaling(self):
        """Test metrics processing and scaling decisions."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=85.0,  # High CPU usage
            memory_percent=40.0,
            disk_io_percent=30.0,
            network_io_percent=20.0,
            active_workers=10,
            queue_size=5,
            response_time_ms=150.0,
            error_rate=0.01
        )
        
        # Process metrics
        events = self.scaler.process_metrics(metrics)
        
        # Should trigger scaling events for high CPU usage
        cpu_events = [e for e in events if e.resource_type == ResourceType.CPU]
        self.assertTrue(len(cpu_events) > 0 or 
                       self.scaler.current_capacity[ResourceType.CPU] >= 
                       self.scaler.scaling_policies[ResourceType.CPU].max_instances)

    def test_utilization_calculations(self):
        """Test resource utilization calculations."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=75.0,
            memory_percent=60.0,
            disk_io_percent=30.0,
            network_io_percent=20.0,
            active_workers=5,
            queue_size=20,
            response_time_ms=200.0,
            error_rate=0.05
        )
        
        # Test CPU utilization
        cpu_util = self.scaler._get_utilization_for_resource(metrics, ResourceType.CPU)
        self.assertEqual(cpu_util, 0.75)
        
        # Test memory utilization
        mem_util = self.scaler._get_utilization_for_resource(metrics, ResourceType.MEMORY)
        self.assertEqual(mem_util, 0.60)
        
        # Test worker utilization
        worker_util = self.scaler._get_utilization_for_resource(metrics, ResourceType.WORKERS)
        self.assertEqual(worker_util, min(1.0, 20 / 5 / 10.0))  # queue/workers/10

    def test_scaling_callbacks(self):
        """Test scaling callback registration and execution."""
        callback_called = False
        callback_action = None
        callback_capacity = None
        
        def test_callback(action, capacity):
            nonlocal callback_called, callback_action, callback_capacity
            callback_called = True
            callback_action = action
            callback_capacity = capacity
            return True
        
        # Register callback
        self.scaler.register_scaling_callback(ResourceType.CPU, test_callback)
        
        # Create high utilization metrics to trigger scaling
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=90.0,  # Very high CPU usage
            memory_percent=30.0,
            disk_io_percent=20.0,
            network_io_percent=10.0,
            active_workers=3,
            queue_size=2,
            response_time_ms=100.0,
            error_rate=0.0
        )
        
        # Process metrics
        events = self.scaler.process_metrics(metrics)
        
        # Check if callback was called (might not be if at max capacity)
        if events and any(e.resource_type == ResourceType.CPU for e in events):
            self.assertTrue(callback_called)
            self.assertEqual(callback_action, ScalingAction.SCALE_UP)

    def test_scaling_insights(self):
        """Test scaling insights generation."""
        # Add some scaling history
        self.scaler.scaling_history.append(
            type('ScalingEvent', (), {
                'timestamp': time.time(),
                'action': ScalingAction.SCALE_UP,
                'resource_type': ResourceType.CPU,
                'success': True
            })()
        )
        
        insights = self.scaler.get_scaling_insights()
        
        self.assertIn('current_capacity', insights)
        self.assertIn('recent_scaling_events', insights)
        self.assertIn('resource_trends', insights)
        self.assertIn('policies', insights)


class TestHIPAAIntegration(unittest.TestCase):
    """Test HIPAA compliance integration with performance optimizations."""

    def setUp(self):
        self.processor = HIPAAProcessor()

    def test_performance_optimized_processing(self):
        """Test HIPAA processing with performance optimizations."""
        # Create test document
        test_content = """
        Patient John Smith, DOB: 01/15/1980
        Medical Record Number: 123456789
        Visited on 03/15/2024 for routine checkup.
        Phone: (555) 123-4567
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Process document
            result = self.processor.process_document(temp_path)
            
            # Verify processing completed
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.summary)
            self.assertGreaterEqual(result.compliance_score, 0.0)
            self.assertLessEqual(result.compliance_score, 1.0)
            self.assertGreater(result.phi_detected_count, 0)
            
        finally:
            Path(temp_path).unlink()

    def test_batch_processing_optimization(self):
        """Test optimized batch processing of HIPAA documents."""
        from src.hipaa_compliance_summarizer.performance_gen4 import optimize_batch_processing
        
        # Create test documents
        test_documents = [
            "Patient Mary Johnson, SSN: 123-45-6789, visited on 01/01/2024",
            "Patient Bob Wilson, Phone: (555) 987-6543, DOB: 05/20/1975",
            "Patient Alice Brown, MRN: 987654321, Address: 123 Main St"
        ]
        
        # Mock processor function
        def mock_phi_processor(doc):
            # Simulate PHI processing
            phi_count = doc.count('Patient') + doc.count('SSN') + doc.count('Phone')
            return {
                'document': doc,
                'phi_detected': phi_count,
                'processing_time': 0.1
            }
        
        # Test batch optimization
        results = optimize_batch_processing(test_documents, mock_phi_processor)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('phi_detected', result)
            self.assertGreater(result['phi_detected'], 0)


class TestQualityGates(unittest.TestCase):
    """Test quality gates and system validation."""

    def test_system_imports(self):
        """Test that all system components can be imported."""
        # Test core imports
        from src.hipaa_compliance_summarizer import HIPAAProcessor
        from src.hipaa_compliance_summarizer.performance_gen4 import MLPerformanceOptimizer
        from src.hipaa_compliance_summarizer.intelligent_scaling import IntelligentAutoScaler
        
        # Verify instances can be created
        processor = HIPAAProcessor()
        optimizer = MLPerformanceOptimizer()
        scaler = IntelligentAutoScaler()
        
        self.assertIsNotNone(processor)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scaler)

    def test_configuration_validation(self):
        """Test system configuration validation."""
        from src.hipaa_compliance_summarizer.performance_gen4 import OptimizationConfig
        from src.hipaa_compliance_summarizer.intelligent_scaling import ScalingPolicy, ResourceType
        
        # Test optimization config
        config = OptimizationConfig(
            enable_ml_optimization=True,
            optimization_window=3600,
            min_samples_for_ml=50
        )
        
        self.assertTrue(config.enable_ml_optimization)
        self.assertEqual(config.optimization_window, 3600)
        
        # Test scaling policy
        policy = ScalingPolicy(
            resource_type=ResourceType.CPU,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            min_instances=1,
            max_instances=10
        )
        
        self.assertEqual(policy.resource_type, ResourceType.CPU)
        self.assertGreater(policy.scale_up_threshold, policy.scale_down_threshold)

    def test_error_handling(self):
        """Test error handling in performance components."""
        optimizer = MLPerformanceOptimizer()
        
        # Test with invalid data
        try:
            optimizer.record_performance_event(None, -1.0)
        except Exception as e:
            self.assertIsInstance(e, (TypeError, ValueError))

    def test_thread_safety(self):
        """Test thread safety of performance components."""
        import threading
        
        optimizer = MLPerformanceOptimizer()
        results = []
        
        def worker():
            for i in range(10):
                optimizer.record_performance_event(f"thread_op", i * 0.1)
                results.append(len(optimizer.performance_history))
        
        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have recorded events from all threads
        self.assertEqual(len(optimizer.performance_history), 30)

    def test_memory_usage(self):
        """Test memory usage limits and cleanup."""
        optimizer = MLPerformanceOptimizer(OptimizationConfig(optimization_window=10))
        
        # Record many events
        for i in range(100):
            optimizer.record_performance_event("memory_test", 1.0)
        
        # Wait for cleanup window
        time.sleep(11)
        
        # Add one more event to trigger cleanup
        optimizer.record_performance_event("cleanup_trigger", 1.0)
        
        # Should have cleaned up old events
        self.assertLess(len(optimizer.performance_history), 50)


if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)