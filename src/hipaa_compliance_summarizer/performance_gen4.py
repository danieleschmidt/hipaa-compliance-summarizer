"""Generation 4 Performance Optimization Engine - Advanced ML-driven optimizations."""

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization engine."""
    enable_ml_optimization: bool = True
    enable_adaptive_caching: bool = True
    enable_predictive_prefetch: bool = True
    enable_load_balancing: bool = True
    enable_resource_prediction: bool = True
    optimization_window: int = 3600  # 1 hour in seconds
    min_samples_for_ml: int = 100


class MLPerformanceOptimizer:
    """Machine learning-driven performance optimization engine."""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.performance_history = []
        self.optimization_models = {}
        self.feature_scalers = {}
        self.lock = threading.Lock()

        # Performance tracking
        self.processing_times = {}
        self.resource_usage = {}
        self.cache_patterns = {}

        # Predictive models
        self.workload_predictor = None
        self.resource_predictor = None
        self.cache_optimizer = None

        # Auto-scaling parameters
        self.cpu_target = 0.7  # Target CPU utilization
        self.memory_target = 0.8  # Target memory utilization
        self.scale_threshold = 0.1  # Threshold for scaling decisions

    def record_performance_event(self, event_type: str, duration: float,
                                metadata: Dict[str, Any] = None):
        """Record performance event for ML optimization."""
        with self.lock:
            event = {
                'timestamp': time.time(),
                'event_type': event_type,
                'duration': duration,
                'metadata': metadata or {},
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'active_threads': threading.active_count()
            }
            self.performance_history.append(event)

            # Maintain rolling window
            cutoff_time = time.time() - self.config.optimization_window
            self.performance_history = [
                e for e in self.performance_history
                if e['timestamp'] > cutoff_time
            ]

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def train_optimization_models(self):
        """Train ML models for performance optimization."""
        if len(self.performance_history) < self.config.min_samples_for_ml:
            logger.info("Insufficient data for ML training")
            return

        logger.info("Training performance optimization models...")

        # Prepare features and targets
        features = []
        durations = []

        for event in self.performance_history:
            feature_vector = [
                event['cpu_usage'],
                event['memory_usage'],
                event['active_threads'],
                hash(event['event_type']) % 1000,  # Categorical encoding
                len(str(event.get('metadata', {}))),  # Metadata complexity
            ]
            features.append(feature_vector)
            durations.append(event['duration'])

        features = np.array(features)
        durations = np.array(durations)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.feature_scalers['performance'] = scaler

        # Train clustering model for workload patterns
        try:
            kmeans = KMeans(n_clusters=min(5, len(features) // 20), random_state=42)
            workload_clusters = kmeans.fit_predict(features_scaled)
            self.optimization_models['workload_clusters'] = kmeans

            # Analyze cluster performance characteristics
            cluster_stats = {}
            for i in range(kmeans.n_clusters):
                cluster_mask = workload_clusters == i
                cluster_durations = durations[cluster_mask]
                cluster_stats[i] = {
                    'mean_duration': np.mean(cluster_durations),
                    'std_duration': np.std(cluster_durations),
                    'sample_count': len(cluster_durations)
                }

            self.optimization_models['cluster_stats'] = cluster_stats
            logger.info(f"Trained workload clustering with {kmeans.n_clusters} clusters")

        except Exception as e:
            logger.warning(f"Failed to train ML models: {e}")

    def predict_optimal_resources(self, workload_type: str,
                                 estimated_size: int) -> Dict[str, Any]:
        """Predict optimal resource allocation for workload."""
        if 'workload_clusters' not in self.optimization_models:
            return self._get_default_resources()

        try:
            # Create feature vector for prediction
            current_cpu = self._get_cpu_usage()
            current_memory = self._get_memory_usage()
            current_threads = threading.active_count()

            feature_vector = np.array([[
                current_cpu,
                current_memory,
                current_threads,
                hash(workload_type) % 1000,
                estimated_size
            ]])

            # Scale features
            scaler = self.feature_scalers.get('performance')
            if scaler:
                feature_vector = scaler.transform(feature_vector)

            # Predict cluster
            kmeans = self.optimization_models['workload_clusters']
            cluster = kmeans.predict(feature_vector)[0]

            # Get cluster statistics
            cluster_stats = self.optimization_models['cluster_stats'][cluster]

            # Calculate optimal resources based on cluster characteristics
            base_threads = max(1, mp.cpu_count() // 2)
            if cluster_stats['mean_duration'] > 10.0:  # High-duration cluster
                recommended_threads = min(base_threads * 2, mp.cpu_count())
                recommended_memory = '2GB'
            elif cluster_stats['mean_duration'] < 1.0:  # Low-duration cluster
                recommended_threads = max(1, base_threads // 2)
                recommended_memory = '512MB'
            else:  # Medium-duration cluster
                recommended_threads = base_threads
                recommended_memory = '1GB'

            return {
                'threads': recommended_threads,
                'memory_limit': recommended_memory,
                'predicted_duration': cluster_stats['mean_duration'],
                'confidence': cluster_stats['sample_count'] / 100.0,
                'cluster_id': cluster
            }

        except Exception as e:
            logger.warning(f"Resource prediction failed: {e}")
            return self._get_default_resources()

    def _get_default_resources(self) -> Dict[str, Any]:
        """Get default resource allocation."""
        return {
            'threads': max(1, mp.cpu_count() // 2),
            'memory_limit': '1GB',
            'predicted_duration': 5.0,
            'confidence': 0.5,
            'cluster_id': -1
        }

    def optimize_processing_pipeline(self, documents: List[Any],
                                   processor_func: callable) -> List[Any]:
        """Optimize processing pipeline using ML insights."""
        start_time = time.time()

        # Predict optimal resources
        workload_type = processor_func.__name__
        estimated_size = len(documents)
        resources = self.predict_optimal_resources(workload_type, estimated_size)

        logger.info(f"Optimizing pipeline: {resources}")

        # Choose processing strategy based on predictions
        if resources['predicted_duration'] > 30.0 and len(documents) > 10:
            # Use process pool for CPU-intensive, long-duration tasks
            results = self._process_with_multiprocessing(
                documents, processor_func, resources['threads']
            )
        elif len(documents) > 50:
            # Use thread pool for I/O-intensive tasks
            results = self._process_with_threading(
                documents, processor_func, resources['threads']
            )
        else:
            # Sequential processing for small workloads
            results = [processor_func(doc) for doc in documents]

        # Record performance
        total_duration = time.time() - start_time
        self.record_performance_event(
            f"pipeline_{workload_type}",
            total_duration,
            {
                'document_count': len(documents),
                'resources_used': resources,
                'processing_strategy': 'multiprocessing' if resources['predicted_duration'] > 30.0 else 'threading'
            }
        )

        return results

    def _process_with_multiprocessing(self, documents: List[Any],
                                    processor_func: callable,
                                    max_workers: int) -> List[Any]:
        """Process documents using multiprocessing."""
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(processor_func, doc) for doc in documents]
                results = []

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5-minute timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Processing failed: {e}")
                        results.append(None)

                return results
        except Exception as e:
            logger.warning(f"Multiprocessing failed, falling back to sequential: {e}")
            return [processor_func(doc) for doc in documents]

    def _process_with_threading(self, documents: List[Any],
                              processor_func: callable,
                              max_workers: int) -> List[Any]:
        """Process documents using threading."""
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(processor_func, doc) for doc in documents]
                results = []

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)  # 1-minute timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Processing failed: {e}")
                        results.append(None)

                return results
        except Exception as e:
            logger.warning(f"Threading failed, falling back to sequential: {e}")
            return [processor_func(doc) for doc in documents]

    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get performance optimization insights."""
        with self.lock:
            if not self.performance_history:
                return {"message": "No performance data available"}

            total_events = len(self.performance_history)
            avg_duration = np.mean([e['duration'] for e in self.performance_history])
            avg_cpu = np.mean([e['cpu_usage'] for e in self.performance_history])
            avg_memory = np.mean([e['memory_usage'] for e in self.performance_history])

            # Event type analysis
            event_stats = {}
            for event in self.performance_history:
                event_type = event['event_type']
                if event_type not in event_stats:
                    event_stats[event_type] = []
                event_stats[event_type].append(event['duration'])

            event_analysis = {}
            for event_type, durations in event_stats.items():
                event_analysis[event_type] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations)
                }

            # Optimization recommendations
            recommendations = []

            if avg_cpu > 80:
                recommendations.append("High CPU usage detected - consider scaling out")
            elif avg_cpu < 30:
                recommendations.append("Low CPU usage - consider scaling down")

            if avg_memory > 85:
                recommendations.append("High memory usage - consider memory optimization")

            if avg_duration > 10.0:
                recommendations.append("High average processing time - consider pipeline optimization")

            return {
                'total_events': total_events,
                'average_duration': avg_duration,
                'average_cpu_usage': avg_cpu,
                'average_memory_usage': avg_memory,
                'event_analysis': event_analysis,
                'recommendations': recommendations,
                'models_trained': len(self.optimization_models),
                'optimization_window_hours': self.config.optimization_window / 3600
            }


class AdaptiveResourceManager:
    """Manages resources adaptively based on workload patterns."""

    def __init__(self):
        self.current_allocation = {
            'cpu_cores': mp.cpu_count(),
            'memory_gb': 4,  # Default memory allocation
            'thread_pool_size': mp.cpu_count() * 2,
            'process_pool_size': mp.cpu_count()
        }
        self.usage_history = []
        self.lock = threading.Lock()

    def monitor_resource_usage(self):
        """Monitor current resource usage."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            with self.lock:
                self.usage_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3)
                })

                # Keep only last hour of data
                cutoff = time.time() - 3600
                self.usage_history = [
                    entry for entry in self.usage_history
                    if entry['timestamp'] > cutoff
                ]

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3)
            }
        except ImportError:
            logger.warning("psutil not available, using default monitoring")
            return {
                'cpu_percent': 50.0,
                'memory_percent': 50.0,
                'memory_available_gb': 2.0
            }

    def auto_scale_resources(self) -> Dict[str, int]:
        """Automatically scale resources based on usage patterns."""
        current_usage = self.monitor_resource_usage()

        with self.lock:
            if len(self.usage_history) < 10:  # Need sufficient data
                return self.current_allocation

            # Calculate usage trends
            recent_cpu = np.mean([e['cpu_percent'] for e in self.usage_history[-10:]])
            recent_memory = np.mean([e['memory_percent'] for e in self.usage_history[-10:]])

            # Auto-scaling decisions
            new_allocation = self.current_allocation.copy()

            # CPU scaling
            if recent_cpu > 80:
                # Scale up thread pool
                new_allocation['thread_pool_size'] = min(
                    self.current_allocation['thread_pool_size'] * 2,
                    mp.cpu_count() * 4
                )
                logger.info(f"Scaling up thread pool to {new_allocation['thread_pool_size']}")

            elif recent_cpu < 30:
                # Scale down thread pool
                new_allocation['thread_pool_size'] = max(
                    self.current_allocation['thread_pool_size'] // 2,
                    mp.cpu_count()
                )
                logger.info(f"Scaling down thread pool to {new_allocation['thread_pool_size']}")

            # Memory-based scaling
            if recent_memory > 85:
                # Reduce process pool size to save memory
                new_allocation['process_pool_size'] = max(
                    self.current_allocation['process_pool_size'] // 2,
                    1
                )
                logger.info(f"Reducing process pool due to memory pressure: {new_allocation['process_pool_size']}")

            self.current_allocation = new_allocation
            return new_allocation

    def get_resource_recommendations(self) -> Dict[str, Any]:
        """Get resource optimization recommendations."""
        if not self.usage_history:
            return {"message": "Insufficient monitoring data"}

        avg_cpu = np.mean([e['cpu_percent'] for e in self.usage_history])
        avg_memory = np.mean([e['memory_percent'] for e in self.usage_history])

        recommendations = {
            'current_allocation': self.current_allocation,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'recommendations': []
        }

        if avg_cpu > 75:
            recommendations['recommendations'].append(
                "High CPU usage - consider adding more CPU cores or optimizing algorithms"
            )
        elif avg_cpu < 25:
            recommendations['recommendations'].append(
                "Low CPU usage - resources may be over-allocated"
            )

        if avg_memory > 80:
            recommendations['recommendations'].append(
                "High memory usage - consider increasing memory or optimizing data structures"
            )

        return recommendations


# Global optimizer instances
ml_optimizer = MLPerformanceOptimizer()
resource_manager = AdaptiveResourceManager()


def optimize_batch_processing(documents: List[Any], processor_func: callable) -> List[Any]:
    """Optimize batch processing using ML-driven resource allocation."""
    return ml_optimizer.optimize_processing_pipeline(documents, processor_func)


def get_performance_insights() -> Dict[str, Any]:
    """Get comprehensive performance insights."""
    ml_insights = ml_optimizer.get_optimization_insights()
    resource_insights = resource_manager.get_resource_recommendations()

    return {
        'ml_optimization': ml_insights,
        'resource_management': resource_insights,
        'auto_scaling_enabled': True,
        'timestamp': time.time()
    }


# Auto-training scheduler
def schedule_ml_training():
    """Schedule periodic ML model training."""
    def training_loop():
        while True:
            try:
                time.sleep(1800)  # Train every 30 minutes
                ml_optimizer.train_optimization_models()
                resource_manager.auto_scale_resources()
            except Exception as e:
                logger.error(f"Auto-training failed: {e}")

    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()
    logger.info("ML training scheduler started")


# Initialize auto-training on module import
if __name__ != "__main__":
    schedule_ml_training()
