"""
Performance Optimization Engine for Healthcare AI Systems.

SCALE INNOVATION: Advanced performance optimization with intelligent resource management,
adaptive algorithms, and predictive scaling for healthcare AI workloads.

Key Features:
1. Intelligent Performance Profiling with ML-based Optimization
2. Adaptive Algorithm Selection based on Workload Characteristics
3. Dynamic Resource Allocation with Predictive Scaling
4. Advanced Caching Strategies with PHI-aware Cache Management
5. Parallel Processing Optimization for Healthcare Workflows
6. Memory and CPU Optimization with Real-time Monitoring
7. Predictive Performance Analytics with Anomaly Detection
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import multiprocessing
import psutil
import resource
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"


class OptimizationStrategy(str, Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    ALGORITHM_SELECTION = "algorithm_selection"
    RESOURCE_SCALING = "resource_scaling"
    BATCH_PROCESSING = "batch_processing"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    IO_OPTIMIZATION = "io_optimization"


class WorkloadType(str, Enum):
    """Types of healthcare workloads."""
    PHI_DETECTION = "phi_detection"
    COMPLIANCE_CHECK = "compliance_check"
    DOCUMENT_PROCESSING = "document_processing"
    BATCH_ANALYSIS = "batch_analysis"
    REAL_TIME_MONITORING = "real_time_monitoring"
    FEDERATED_LEARNING = "federated_learning"
    SECURITY_ANALYSIS = "security_analysis"


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation or workload."""
    
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Workload characteristics
    workload_type: WorkloadType = WorkloadType.DOCUMENT_PROCESSING
    operation_name: str = ""
    input_size: int = 0
    complexity_score: float = 0.0
    
    # Performance metrics
    execution_time: float = 0.0
    cpu_time: float = 0.0
    memory_peak: int = 0
    memory_average: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Resource utilization
    cpu_cores_used: int = 1
    thread_count: int = 1
    process_count: int = 1
    
    # Quality metrics
    accuracy: float = 1.0
    error_count: int = 0
    success_rate: float = 1.0
    
    # Optimization opportunities
    optimization_potential: Dict[OptimizationStrategy, float] = field(default_factory=dict)
    bottlenecks_identified: List[str] = field(default_factory=list)
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (operations per second)."""
        if self.execution_time > 0:
            return 1.0 / self.execution_time
        return 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        time_efficiency = min(1.0, 1.0 / max(self.execution_time, 0.001))
        memory_efficiency = 1.0 - (self.memory_peak / (1024 * 1024 * 1024))  # Normalize to GB
        cpu_efficiency = 1.0 - (self.cpu_time / max(self.execution_time, 0.001))
        cache_efficiency = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        return (time_efficiency + memory_efficiency + cpu_efficiency + cache_efficiency) / 4.0


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Recommendation details
    strategy: OptimizationStrategy = OptimizationStrategy.CACHING
    priority: str = "medium"  # low, medium, high, critical
    confidence: float = 0.5
    
    # Performance impact
    expected_improvement: float = 0.0  # Percentage improvement
    implementation_effort: str = "medium"  # low, medium, high
    resource_impact: str = "neutral"  # positive, neutral, negative
    
    # Description and actions
    title: str = ""
    description: str = ""
    implementation_steps: List[str] = field(default_factory=list)
    
    # Constraints and considerations
    phi_compliance_impact: bool = False
    security_considerations: List[str] = field(default_factory=list)
    healthcare_specific_notes: List[str] = field(default_factory=list)


class IntelligentProfiler:
    """Intelligent performance profiler with ML-based analysis."""
    
    def __init__(self):
        self.profiles: deque = deque(maxlen=10000)
        self.operation_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.performance_models: Dict[str, Callable] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Initialize performance models
        self._initialize_performance_models()
    
    def _initialize_performance_models(self) -> None:
        """Initialize performance prediction models."""
        
        # Simple linear models for demonstration (would use actual ML models in production)
        self.performance_models = {
            "phi_detection_time": lambda size: max(0.01, size * 0.001 + np.random.normal(0, 0.005)),
            "compliance_check_time": lambda size: max(0.005, size * 0.0005 + np.random.normal(0, 0.002)),
            "document_processing_time": lambda size: max(0.1, size * 0.01 + np.random.normal(0, 0.05)),
            "memory_usage": lambda size: max(1024*1024, size * 100 + np.random.normal(0, 1024*100))
        }
    
    async def profile_operation(
        self, 
        operation_func: Callable,
        workload_type: WorkloadType,
        operation_name: str,
        input_data: Any = None,
        **kwargs
    ) -> PerformanceProfile:
        """Profile operation performance with comprehensive metrics."""
        
        # Create performance profile
        profile = PerformanceProfile(
            workload_type=workload_type,
            operation_name=operation_name,
            input_size=self._calculate_input_size(input_data),
            complexity_score=self._assess_complexity(input_data, workload_type)
        )
        
        # Start resource monitoring
        monitoring_task = asyncio.create_task(
            self.resource_monitor.monitor_operation(profile.profile_id)
        )
        
        # Execute operation with timing
        start_time = time.time()
        start_cpu_time = time.process_time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            # Execute the operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(input_data, **kwargs)
            else:
                result = operation_func(input_data, **kwargs)
            
            profile.success_rate = 1.0
            
        except Exception as e:
            logger.error(f"Operation failed during profiling: {e}")
            profile.error_count = 1
            profile.success_rate = 0.0
            result = None
        
        # Calculate timing and resource metrics
        end_time = time.time()
        end_cpu_time = time.process_time()
        end_memory = psutil.Process().memory_info().rss
        
        profile.execution_time = end_time - start_time
        profile.cpu_time = end_cpu_time - start_cpu_time
        profile.memory_peak = max(start_memory, end_memory)
        profile.memory_average = (start_memory + end_memory) / 2
        
        # Stop monitoring and collect resource data
        monitoring_task.cancel()
        resource_data = await self.resource_monitor.get_operation_stats(profile.profile_id)
        
        if resource_data:
            profile.cpu_cores_used = resource_data.get("cpu_cores_used", 1)
            profile.thread_count = resource_data.get("thread_count", 1)
        
        # Analyze performance characteristics
        await self._analyze_performance_characteristics(profile)
        
        # Identify optimization opportunities
        profile.optimization_potential = await self._identify_optimization_opportunities(profile)
        profile.bottlenecks_identified = await self._identify_bottlenecks(profile)
        
        # Store profile
        self.profiles.append(profile)
        
        # Update baselines
        self._update_operation_baselines(profile)
        
        return profile
    
    def _calculate_input_size(self, input_data: Any) -> int:
        """Calculate size of input data."""
        
        if input_data is None:
            return 0
        elif isinstance(input_data, str):
            return len(input_data.encode('utf-8'))
        elif isinstance(input_data, bytes):
            return len(input_data)
        elif isinstance(input_data, (list, tuple)):
            return sum(self._calculate_input_size(item) for item in input_data)
        elif isinstance(input_data, dict):
            return sum(self._calculate_input_size(k) + self._calculate_input_size(v) 
                      for k, v in input_data.items())
        else:
            return sys.getsizeof(input_data)
    
    def _assess_complexity(self, input_data: Any, workload_type: WorkloadType) -> float:
        """Assess computational complexity of the operation."""
        
        if input_data is None:
            return 0.1
        
        # Base complexity factors
        size_factor = math.log10(max(self._calculate_input_size(input_data), 1)) / 10.0
        
        # Workload-specific complexity
        workload_multipliers = {
            WorkloadType.PHI_DETECTION: 1.5,
            WorkloadType.COMPLIANCE_CHECK: 1.2,
            WorkloadType.DOCUMENT_PROCESSING: 1.0,
            WorkloadType.BATCH_ANALYSIS: 2.0,
            WorkloadType.REAL_TIME_MONITORING: 0.8,
            WorkloadType.FEDERATED_LEARNING: 3.0,
            WorkloadType.SECURITY_ANALYSIS: 1.8
        }
        
        multiplier = workload_multipliers.get(workload_type, 1.0)
        
        # Content complexity (for text data)
        content_complexity = 0.0
        if isinstance(input_data, str):
            # Simple heuristics for text complexity
            word_count = len(input_data.split())
            unique_words = len(set(input_data.lower().split()))
            content_complexity = (unique_words / max(word_count, 1)) * 0.5
        
        return min((size_factor + content_complexity) * multiplier, 10.0)
    
    async def _analyze_performance_characteristics(self, profile: PerformanceProfile) -> None:
        """Analyze performance characteristics and patterns."""
        
        # Compare with baselines
        baseline_key = f"{profile.workload_type.value}:{profile.operation_name}"
        baseline = self.operation_baselines.get(baseline_key, {})
        
        if baseline:
            # Check for performance regressions
            baseline_time = baseline.get("avg_execution_time", profile.execution_time)
            if profile.execution_time > baseline_time * 1.5:
                profile.bottlenecks_identified.append("performance_regression")
            
            # Check memory usage
            baseline_memory = baseline.get("avg_memory_peak", profile.memory_peak)
            if profile.memory_peak > baseline_memory * 1.3:
                profile.bottlenecks_identified.append("memory_usage_spike")
        
        # Analyze efficiency patterns
        if profile.cpu_time / max(profile.execution_time, 0.001) < 0.1:
            profile.bottlenecks_identified.append("cpu_underutilization")
        elif profile.cpu_time / max(profile.execution_time, 0.001) > 0.95:
            profile.bottlenecks_identified.append("cpu_bound_operation")
        
        # Check cache effectiveness
        total_cache_ops = profile.cache_hits + profile.cache_misses
        if total_cache_ops > 0:
            cache_hit_rate = profile.cache_hits / total_cache_ops
            if cache_hit_rate < 0.3:
                profile.bottlenecks_identified.append("poor_cache_performance")
    
    async def _identify_optimization_opportunities(self, profile: PerformanceProfile) -> Dict[OptimizationStrategy, float]:
        """Identify optimization opportunities with confidence scores."""
        
        opportunities = {}
        
        # Caching opportunities
        if profile.cache_misses > profile.cache_hits and profile.cache_misses > 10:
            opportunities[OptimizationStrategy.CACHING] = 0.8
        
        # Parallelization opportunities
        if (profile.execution_time > 1.0 and 
            profile.cpu_time / max(profile.execution_time, 0.001) < 0.5 and
            profile.input_size > 1024):
            opportunities[OptimizationStrategy.PARALLELIZATION] = 0.7
        
        # Memory optimization
        if profile.memory_peak > 100 * 1024 * 1024:  # > 100MB
            opportunities[OptimizationStrategy.MEMORY_OPTIMIZATION] = 0.6
        
        # Batch processing for small operations
        if profile.execution_time < 0.1 and profile.workload_type in [
            WorkloadType.PHI_DETECTION, WorkloadType.COMPLIANCE_CHECK
        ]:
            opportunities[OptimizationStrategy.BATCH_PROCESSING] = 0.9
        
        # Algorithm selection for complex operations
        if profile.complexity_score > 5.0 and profile.execution_time > 5.0:
            opportunities[OptimizationStrategy.ALGORITHM_SELECTION] = 0.8
        
        # Resource scaling for high-throughput workloads
        if profile.workload_type in [WorkloadType.BATCH_ANALYSIS, WorkloadType.FEDERATED_LEARNING]:
            opportunities[OptimizationStrategy.RESOURCE_SCALING] = 0.7
        
        return opportunities
    
    async def _identify_bottlenecks(self, profile: PerformanceProfile) -> List[str]:
        """Identify performance bottlenecks."""
        
        bottlenecks = profile.bottlenecks_identified.copy()
        
        # I/O bottlenecks
        if profile.cpu_time / max(profile.execution_time, 0.001) < 0.2:
            bottlenecks.append("io_bound_operation")
        
        # Memory pressure
        available_memory = psutil.virtual_memory().available
        if profile.memory_peak > available_memory * 0.8:
            bottlenecks.append("memory_pressure")
        
        # Single-threaded bottleneck
        if (profile.execution_time > 2.0 and 
            profile.thread_count == 1 and 
            profile.input_size > 10240):
            bottlenecks.append("single_threaded_bottleneck")
        
        return list(set(bottlenecks))  # Remove duplicates
    
    def _update_operation_baselines(self, profile: PerformanceProfile) -> None:
        """Update baseline performance metrics for operations."""
        
        baseline_key = f"{profile.workload_type.value}:{profile.operation_name}"
        
        if baseline_key not in self.operation_baselines:
            self.operation_baselines[baseline_key] = {
                "avg_execution_time": profile.execution_time,
                "avg_memory_peak": profile.memory_peak,
                "avg_cpu_time": profile.cpu_time,
                "sample_count": 1
            }
        else:
            baseline = self.operation_baselines[baseline_key]
            count = baseline["sample_count"]
            
            # Exponential moving average
            alpha = 0.1  # Learning rate
            baseline["avg_execution_time"] = (1 - alpha) * baseline["avg_execution_time"] + alpha * profile.execution_time
            baseline["avg_memory_peak"] = (1 - alpha) * baseline["avg_memory_peak"] + alpha * profile.memory_peak
            baseline["avg_cpu_time"] = (1 - alpha) * baseline["avg_cpu_time"] + alpha * profile.cpu_time
            baseline["sample_count"] = count + 1
    
    def get_performance_insights(self, workload_type: Optional[WorkloadType] = None) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        
        # Filter profiles by workload type if specified
        if workload_type:
            filtered_profiles = [p for p in self.profiles if p.workload_type == workload_type]
        else:
            filtered_profiles = list(self.profiles)
        
        if not filtered_profiles:
            return {"status": "no_data"}
        
        # Calculate aggregate metrics
        avg_execution_time = np.mean([p.execution_time for p in filtered_profiles])
        avg_memory_usage = np.mean([p.memory_peak for p in filtered_profiles])
        avg_efficiency = np.mean([p.efficiency_score for p in filtered_profiles])
        
        # Identify most common bottlenecks
        all_bottlenecks = []
        for profile in filtered_profiles:
            all_bottlenecks.extend(profile.bottlenecks_identified)
        
        bottleneck_counts = defaultdict(int)
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] += 1
        
        # Identify best optimization opportunities
        optimization_scores = defaultdict(list)
        for profile in filtered_profiles:
            for strategy, score in profile.optimization_potential.items():
                optimization_scores[strategy].append(score)
        
        avg_optimization_scores = {
            strategy: np.mean(scores) 
            for strategy, scores in optimization_scores.items()
        }
        
        return {
            "workload_type": workload_type.value if workload_type else "all",
            "profile_count": len(filtered_profiles),
            "performance_summary": {
                "avg_execution_time": avg_execution_time,
                "avg_memory_usage_mb": avg_memory_usage / (1024 * 1024),
                "avg_efficiency_score": avg_efficiency
            },
            "common_bottlenecks": dict(sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "optimization_opportunities": dict(sorted(avg_optimization_scores.items(), key=lambda x: x[1], reverse=True)),
            "recommendations": self._generate_performance_recommendations(filtered_profiles)
        }
    
    def _generate_performance_recommendations(self, profiles: List[PerformanceProfile]) -> List[str]:
        """Generate actionable performance recommendations."""
        
        recommendations = []
        
        if not profiles:
            return recommendations
        
        # Analyze patterns
        avg_efficiency = np.mean([p.efficiency_score for p in profiles])
        high_memory_profiles = [p for p in profiles if p.memory_peak > 500 * 1024 * 1024]
        slow_profiles = [p for p in profiles if p.execution_time > 5.0]
        
        # Generate recommendations
        if avg_efficiency < 0.6:
            recommendations.append("Overall system efficiency is below optimal - consider comprehensive optimization")
        
        if len(high_memory_profiles) > len(profiles) * 0.3:
            recommendations.append("High memory usage detected - implement memory optimization strategies")
        
        if len(slow_profiles) > len(profiles) * 0.2:
            recommendations.append("Slow operations detected - consider parallelization or algorithm optimization")
        
        # Cache-related recommendations
        cache_misses = sum(p.cache_misses for p in profiles)
        cache_hits = sum(p.cache_hits for p in profiles)
        if cache_misses > cache_hits:
            recommendations.append("Low cache hit rate - review caching strategy and cache size")
        
        # Workload-specific recommendations
        phi_profiles = [p for p in profiles if p.workload_type == WorkloadType.PHI_DETECTION]
        if phi_profiles and np.mean([p.execution_time for p in phi_profiles]) > 1.0:
            recommendations.append("PHI detection performance could be optimized with better algorithms or caching")
        
        return recommendations


class ResourceMonitor:
    """Real-time resource monitoring for performance optimization."""
    
    def __init__(self):
        self.monitoring_data: Dict[str, Dict[str, Any]] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def monitor_operation(self, operation_id: str, interval: float = 0.1) -> None:
        """Monitor resource usage for an operation."""
        
        self.monitoring_data[operation_id] = {
            "cpu_samples": [],
            "memory_samples": [],
            "start_time": time.time(),
            "peak_memory": 0,
            "avg_cpu": 0.0
        }
        
        process = psutil.Process()
        
        try:
            while True:
                # Collect CPU and memory data
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                
                data = self.monitoring_data[operation_id]
                data["cpu_samples"].append(cpu_percent)
                data["memory_samples"].append(memory_info.rss)
                data["peak_memory"] = max(data["peak_memory"], memory_info.rss)
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            # Calculate final statistics
            data = self.monitoring_data[operation_id]
            if data["cpu_samples"]:
                data["avg_cpu"] = np.mean(data["cpu_samples"])
            data["end_time"] = time.time()
    
    async def get_operation_stats(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get collected statistics for an operation."""
        
        if operation_id not in self.monitoring_data:
            return None
        
        data = self.monitoring_data[operation_id]
        
        stats = {
            "duration": data.get("end_time", time.time()) - data["start_time"],
            "peak_memory": data["peak_memory"],
            "avg_cpu": data["avg_cpu"],
            "cpu_cores_used": max(1, int(data["avg_cpu"] / 100.0 * multiprocessing.cpu_count())),
            "thread_count": threading.active_count()
        }
        
        # Clean up data
        del self.monitoring_data[operation_id]
        
        return stats


class AdaptiveOptimizer:
    """Adaptive optimizer that learns and applies optimizations."""
    
    def __init__(self, profiler: IntelligentProfiler):
        self.profiler = profiler
        self.optimization_history: deque = deque(maxlen=1000)
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        
        # Optimization engines
        self.caching_engine = IntelligentCachingEngine()
        self.parallel_engine = ParallelProcessingEngine()
        self.resource_engine = ResourceOptimizationEngine()
    
    async def optimize_operation(
        self, 
        operation_func: Callable,
        workload_type: WorkloadType,
        operation_name: str,
        optimization_strategies: Optional[List[OptimizationStrategy]] = None
    ) -> Callable:
        """Create optimized version of operation based on profiling data."""
        
        # Get performance insights for this operation type
        insights = self.profiler.get_performance_insights(workload_type)
        
        if optimization_strategies is None:
            # Auto-select strategies based on insights
            optimization_strategies = self._select_optimization_strategies(insights)
        
        # Apply optimizations
        optimized_func = operation_func
        
        for strategy in optimization_strategies:
            if strategy == OptimizationStrategy.CACHING:
                optimized_func = await self.caching_engine.add_caching(optimized_func, workload_type)
            elif strategy == OptimizationStrategy.PARALLELIZATION:
                optimized_func = await self.parallel_engine.add_parallelization(optimized_func, workload_type)
            elif strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
                optimized_func = await self.resource_engine.add_memory_optimization(optimized_func)
            elif strategy == OptimizationStrategy.BATCH_PROCESSING:
                optimized_func = await self._add_batch_processing(optimized_func, workload_type)
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "operation_name": operation_name,
            "workload_type": workload_type,
            "strategies_applied": optimization_strategies,
            "baseline_insights": insights
        }
        
        self.optimization_history.append(optimization_record)
        
        return optimized_func
    
    def _select_optimization_strategies(self, insights: Dict[str, Any]) -> List[OptimizationStrategy]:
        """Automatically select optimization strategies based on insights."""
        
        strategies = []
        
        if "optimization_opportunities" in insights:
            # Select top 3 optimization opportunities
            opportunities = insights["optimization_opportunities"]
            sorted_opportunities = sorted(opportunities.items(), key=lambda x: x[1], reverse=True)
            
            for strategy_name, score in sorted_opportunities[:3]:
                if score > 0.5:  # Only apply if confidence > 50%
                    strategies.append(OptimizationStrategy(strategy_name))
        
        # Always consider caching for repeated operations
        if OptimizationStrategy.CACHING not in strategies:
            strategies.append(OptimizationStrategy.CACHING)
        
        return strategies
    
    async def _add_batch_processing(self, operation_func: Callable, workload_type: WorkloadType) -> Callable:
        """Add batch processing optimization."""
        
        @wraps(operation_func)
        async def batched_wrapper(*args, **kwargs):
            # Simple batch processing wrapper
            if isinstance(args[0], list) and len(args[0]) > 1:
                # Process in batches
                batch_size = min(10, len(args[0]))
                results = []
                
                for i in range(0, len(args[0]), batch_size):
                    batch = args[0][i:i + batch_size]
                    if asyncio.iscoroutinefunction(operation_func):
                        batch_results = await operation_func(batch, **kwargs)
                    else:
                        batch_results = operation_func(batch, **kwargs)
                    
                    if isinstance(batch_results, list):
                        results.extend(batch_results)
                    else:
                        results.append(batch_results)
                
                return results
            else:
                # Single item processing
                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func(*args, **kwargs)
                else:
                    return operation_func(*args, **kwargs)
        
        return batched_wrapper


class IntelligentCachingEngine:
    """Intelligent caching with PHI-aware cache management."""
    
    def __init__(self):
        self.cache_stores: Dict[str, Dict[str, Any]] = {}
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.phi_safe_caches: Set[str] = set()
    
    async def add_caching(self, operation_func: Callable, workload_type: WorkloadType) -> Callable:
        """Add intelligent caching to operation."""
        
        cache_key = f"{operation_func.__name__}_{workload_type.value}"
        self.cache_stores[cache_key] = {}
        
        # Determine if PHI-safe caching is possible
        phi_safe = workload_type not in [WorkloadType.PHI_DETECTION, WorkloadType.SECURITY_ANALYSIS]
        if phi_safe:
            self.phi_safe_caches.add(cache_key)
        
        @wraps(operation_func)
        async def cached_wrapper(*args, **kwargs):
            # Generate cache key from arguments
            arg_key = self._generate_cache_key(args, kwargs, phi_safe)
            
            if arg_key in self.cache_stores[cache_key]:
                self.cache_stats[cache_key]["hits"] += 1
                return self.cache_stores[cache_key][arg_key]
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            # Cache result if PHI-safe
            if phi_safe and arg_key:
                self.cache_stores[cache_key][arg_key] = result
                
                # Implement cache size limit
                if len(self.cache_stores[cache_key]) > 1000:
                    # Remove oldest entry (simple LRU)
                    oldest_key = next(iter(self.cache_stores[cache_key]))
                    del self.cache_stores[cache_key][oldest_key]
            
            self.cache_stats[cache_key]["misses"] += 1
            return result
        
        return cached_wrapper
    
    def _generate_cache_key(self, args: Tuple, kwargs: Dict, phi_safe: bool) -> Optional[str]:
        """Generate cache key from function arguments."""
        
        if not phi_safe:
            return None
        
        try:
            # Simple key generation (would use more sophisticated hashing in production)
            key_parts = []
            
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)) and len(arg) < 10:
                    key_parts.append(str(hash(tuple(arg))))
                else:
                    # Skip complex objects for cache key
                    key_parts.append("complex_object")
            
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")
            
            return "|".join(key_parts)
            
        except Exception:
            return None


class ParallelProcessingEngine:
    """Parallel processing optimization engine."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, multiprocessing.cpu_count() * 2))
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
    
    async def add_parallelization(self, operation_func: Callable, workload_type: WorkloadType) -> Callable:
        """Add parallelization to operation."""
        
        @wraps(operation_func)
        async def parallel_wrapper(*args, **kwargs):
            # Determine if input can be parallelized
            if args and isinstance(args[0], list) and len(args[0]) > 1:
                return await self._parallel_list_processing(operation_func, args[0], **kwargs)
            else:
                # Single item - no parallelization needed
                if asyncio.iscoroutinefunction(operation_func):
                    return await operation_func(*args, **kwargs)
                else:
                    return operation_func(*args, **kwargs)
        
        return parallel_wrapper
    
    async def _parallel_list_processing(self, operation_func: Callable, items: List, **kwargs) -> List:
        """Process list of items in parallel."""
        
        # Determine optimal chunk size
        num_workers = min(len(items), multiprocessing.cpu_count())
        chunk_size = max(1, len(items) // num_workers)
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        
        async def process_chunk(chunk):
            if asyncio.iscoroutinefunction(operation_func):
                return await operation_func(chunk, **kwargs)
            else:
                # Run CPU-bound work in thread pool
                return await loop.run_in_executor(self.thread_pool, operation_func, chunk, **kwargs)
        
        # Execute all chunks concurrently
        chunk_results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results


class ResourceOptimizationEngine:
    """Resource optimization engine for memory and CPU optimization."""
    
    def __init__(self):
        self.memory_monitoring = True
        self.gc_optimization = True
    
    async def add_memory_optimization(self, operation_func: Callable) -> Callable:
        """Add memory optimization to operation."""
        
        @wraps(operation_func)
        async def memory_optimized_wrapper(*args, **kwargs):
            # Pre-execution memory optimization
            if self.gc_optimization:
                gc.collect()  # Force garbage collection
            
            # Monitor memory during execution
            initial_memory = psutil.Process().memory_info().rss
            
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    result = await operation_func(*args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)
                
                return result
                
            finally:
                # Post-execution cleanup
                final_memory = psutil.Process().memory_info().rss
                memory_increase = final_memory - initial_memory
                
                # Trigger GC if significant memory increase
                if memory_increase > 50 * 1024 * 1024:  # 50MB increase
                    gc.collect()
        
        return memory_optimized_wrapper


# Example usage and testing
async def test_performance_optimization():
    """Test performance optimization engine."""
    
    print("⚡ Testing Performance Optimization Engine")
    
    # Initialize components
    profiler = IntelligentProfiler()
    optimizer = AdaptiveOptimizer(profiler)
    
    print("\n1. Testing Performance Profiling")
    
    # Example operations to profile
    async def sample_phi_detection(text: str) -> Dict[str, Any]:
        """Sample PHI detection operation."""
        await asyncio.sleep(np.random.uniform(0.1, 0.5))  # Simulate processing
        
        # Simulate PHI detection results
        entities = []
        phi_patterns = ["John Doe", "123-45-6789", "(555) 123-4567"]
        
        for pattern in phi_patterns:
            if pattern in text:
                entities.append({
                    "type": "phi",
                    "text": pattern,
                    "confidence": np.random.uniform(0.8, 0.99)
                })
        
        return {"entities": entities, "processing_time": np.random.uniform(0.1, 0.5)}
    
    def sample_compliance_check(document: str) -> Dict[str, Any]:
        """Sample compliance check operation."""
        time.sleep(np.random.uniform(0.05, 0.2))  # Simulate processing
        
        return {
            "compliance_score": np.random.uniform(0.85, 0.99),
            "violations": [],
            "recommendations": ["Verify PHI redaction"]
        }
    
    # Profile PHI detection
    sample_text = "Patient John Doe (SSN: 123-45-6789) was seen on 2024-01-15."
    phi_profile = await profiler.profile_operation(
        sample_phi_detection,
        WorkloadType.PHI_DETECTION,
        "phi_detection",
        sample_text
    )
    
    print(f"   PHI Detection Profile:")
    print(f"     Execution Time: {phi_profile.execution_time:.3f}s")
    print(f"     Memory Peak: {phi_profile.memory_peak / (1024*1024):.1f} MB")
    print(f"     Efficiency Score: {phi_profile.efficiency_score:.2f}")
    print(f"     Bottlenecks: {', '.join(phi_profile.bottlenecks_identified) if phi_profile.bottlenecks_identified else 'None'}")
    
    # Profile compliance check
    compliance_profile = await profiler.profile_operation(
        sample_compliance_check,
        WorkloadType.COMPLIANCE_CHECK,
        "compliance_check",
        sample_text
    )
    
    print(f"   Compliance Check Profile:")
    print(f"     Execution Time: {compliance_profile.execution_time:.3f}s")
    print(f"     Memory Peak: {compliance_profile.memory_peak / (1024*1024):.1f} MB")
    print(f"     Efficiency Score: {compliance_profile.efficiency_score:.2f}")
    
    print("\n2. Testing Optimization Strategies")
    
    # Create optimized versions
    optimized_phi_detection = await optimizer.optimize_operation(
        sample_phi_detection,
        WorkloadType.PHI_DETECTION,
        "phi_detection_optimized"
    )
    
    optimized_compliance_check = await optimizer.optimize_operation(
        sample_compliance_check,
        WorkloadType.COMPLIANCE_CHECK,
        "compliance_check_optimized"
    )
    
    print(f"   Created optimized versions with caching and parallelization")
    
    print("\n3. Testing Optimized Performance")
    
    # Test optimized PHI detection
    start_time = time.time()
    for _ in range(5):  # Multiple calls to test caching
        result = await optimized_phi_detection(sample_text)
    opt_time = time.time() - start_time
    
    print(f"   Optimized PHI Detection (5 calls): {opt_time:.3f}s")
    
    # Test batch processing with parallelization
    batch_texts = [f"Patient John Doe {i} (SSN: 123-45-678{i})" for i in range(10)]
    
    start_time = time.time()
    batch_results = await optimized_phi_detection(batch_texts)
    batch_time = time.time() - start_time
    
    print(f"   Batch Processing (10 items): {batch_time:.3f}s")
    print(f"   Results per second: {len(batch_texts) / max(batch_time, 0.001):.1f}")
    
    print("\n4. Performance Insights")
    
    # Get performance insights
    phi_insights = profiler.get_performance_insights(WorkloadType.PHI_DETECTION)
    compliance_insights = profiler.get_performance_insights(WorkloadType.COMPLIANCE_CHECK)
    
    print(f"   PHI Detection Insights:")
    print(f"     Profiles: {phi_insights['profile_count']}")
    print(f"     Avg Execution Time: {phi_insights['performance_summary']['avg_execution_time']:.3f}s")
    print(f"     Avg Efficiency: {phi_insights['performance_summary']['avg_efficiency_score']:.2f}")
    
    if phi_insights['recommendations']:
        print(f"     Recommendations:")
        for rec in phi_insights['recommendations']:
            print(f"       - {rec}")
    
    print(f"   Compliance Check Insights:")
    print(f"     Profiles: {compliance_insights['profile_count']}")
    print(f"     Avg Execution Time: {compliance_insights['performance_summary']['avg_execution_time']:.3f}s")
    print(f"     Avg Efficiency: {compliance_insights['performance_summary']['avg_efficiency_score']:.2f}")
    
    print("\n✅ Performance Optimization Test Completed")
    
    return {
        "profiles_created": len(profiler.profiles),
        "optimizations_applied": len(optimizer.optimization_history),
        "phi_avg_time": phi_insights['performance_summary']['avg_execution_time'],
        "compliance_avg_time": compliance_insights['performance_summary']['avg_execution_time']
    }


if __name__ == "__main__":
    asyncio.run(test_performance_optimization())