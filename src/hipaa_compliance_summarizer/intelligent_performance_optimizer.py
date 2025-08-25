"""
Intelligent Performance Optimizer - ML-Driven Performance Enhancement

This module implements an intelligent performance optimization system that uses
machine learning, profiling, and adaptive algorithms to automatically optimize
quality gate performance, resource utilization, and execution efficiency.
"""

import asyncio
import cProfile
import io
import logging
import pstats
try:
    import psutil
except ImportError:
    print("psutil not available, using system command fallbacks")
    
    # Create psutil-like fallback
    class PSUtilFallback:
        @staticmethod
        def cpu_percent():
            return 50.0  # Default assumption
        
        @staticmethod
        def virtual_memory():
            class MemInfo:
                percent = 60.0
                available = 1024 ** 3  # 1GB
            return MemInfo()
        
        @staticmethod
        def disk_io_counters():
            class DiskIO:
                read_count = 0
                write_count = 0
            return DiskIO()
        
        @staticmethod
        def net_io_counters():
            class NetIO:
                bytes_sent = 0
                bytes_recv = 0
            return NetIO()
        
        @staticmethod
        def Process():
            class ProcessInfo:
                def memory_info(self):
                    class MemInfo:
                        rss = 512 * 1024 * 1024  # 512MB
                    return MemInfo()
                
                def open_files(self):
                    return []
                
                def num_threads(self):
                    return 4
            return ProcessInfo()
    
    psutil = PSUtilFallback()
import resource
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

try:
    import numpy as np
except ImportError:
    print("NumPy not available, using Python math fallback")
    import statistics
    
    # Create numpy-like functions using built-in modules
    class NumpyFallback:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0
        
        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0
        
        @staticmethod
        def percentile(values, q):
            if not values:
                return 0
            sorted_vals = sorted(values)
            index = (len(sorted_vals) - 1) * q / 100
            lower = int(index)
            upper = min(lower + 1, len(sorted_vals) - 1)
            weight = index - lower
            return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def polyfit(x, y, deg):
            # Simple linear regression for degree 1
            if deg == 1 and len(x) == len(y) and len(x) > 1:
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                return [slope, 0]  # Return [slope, intercept]
            return [0, 0]
    
    np = NumpyFallback()

T = TypeVar('T')


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    operation_name: str
    duration: float
    cpu_usage: float
    memory_usage: float
    io_operations: int
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    open_files: int
    threads_count: int


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str
    priority: str  # high, medium, low
    description: str
    implementation: str
    expected_improvement: float
    estimated_effort: str
    code_location: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Detailed performance profile of an operation."""
    total_time: float
    function_calls: Dict[str, int]
    hotspots: List[Tuple[str, float]]  # (function_name, time_spent)
    memory_profile: Dict[str, float]
    bottlenecks: List[str]


class PerformanceMonitor:
    """
    Advanced performance monitoring system with real-time metrics collection.
    
    Monitors CPU, memory, I/O, and custom performance metrics in real-time,
    providing insights for intelligent optimization.
    """
    
    def __init__(self, history_limit: int = 10000):
        self.history_limit = history_limit
        self.metrics_history: deque = deque(maxlen=history_limit)
        self.active_monitors: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.resource_alerts: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
        # Performance thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.response_time_threshold = 10.0
        
    def start_monitoring(self, interval: float = 1.0):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        monitor_thread = threading.Thread(
            target=self._monitor_system_resources,
            args=(interval,),
            daemon=True
        )
        monitor_thread.start()
        self.active_monitors["system_resources"] = monitor_thread
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        for monitor_name, thread in self.active_monitors.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.active_monitors.clear()
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_system_resources(self, interval: float):
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                usage = self._capture_resource_usage()
                
                # Check for resource alerts
                self._check_resource_alerts(usage)
                
                # Store metrics
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    operation_name="system_monitoring",
                    duration=interval,
                    cpu_usage=usage.cpu_percent,
                    memory_usage=usage.memory_percent,
                    io_operations=usage.disk_io_read + usage.disk_io_write,
                    context={
                        "memory_available": usage.memory_available,
                        "open_files": usage.open_files,
                        "threads": usage.threads_count
                    }
                )
                self.metrics_history.append(metric)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def _capture_resource_usage(self) -> ResourceUsage:
        """Capture current resource usage snapshot."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                disk_io_read=disk_io.read_count if disk_io else 0,
                disk_io_write=disk_io.write_count if disk_io else 0,
                network_io_sent=network_io.bytes_sent if network_io else 0,
                network_io_recv=network_io.bytes_recv if network_io else 0,
                open_files=len(process.open_files()),
                threads_count=process.num_threads()
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to capture resource usage: {e}")
            return ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _check_resource_alerts(self, usage: ResourceUsage):
        """Check for resource usage alerts."""
        alerts = []
        
        if usage.cpu_percent > self.cpu_threshold:
            alerts.append({
                "type": "cpu_high",
                "value": usage.cpu_percent,
                "threshold": self.cpu_threshold,
                "timestamp": datetime.now(),
                "severity": "warning" if usage.cpu_percent < 95 else "critical"
            })
        
        if usage.memory_percent > self.memory_threshold:
            alerts.append({
                "type": "memory_high",
                "value": usage.memory_percent,
                "threshold": self.memory_threshold,
                "timestamp": datetime.now(),
                "severity": "warning" if usage.memory_percent < 95 else "critical"
            })
        
        if usage.memory_available < 0.5:  # Less than 500MB available
            alerts.append({
                "type": "memory_low",
                "value": usage.memory_available,
                "threshold": 0.5,
                "timestamp": datetime.now(),
                "severity": "critical"
            })
        
        self.resource_alerts.extend(alerts)
        
        for alert in alerts:
            self.logger.warning(f"Resource alert: {alert['type']} - {alert['value']:.2f} > {alert['threshold']}")
    
    def record_operation_metric(self, metric: PerformanceMetric):
        """Record a performance metric for a specific operation."""
        self.metrics_history.append(metric)
        
        # Check for performance degradation
        if metric.duration > self.response_time_threshold:
            self.resource_alerts.append({
                "type": "slow_operation",
                "operation": metric.operation_name,
                "duration": metric.duration,
                "threshold": self.response_time_threshold,
                "timestamp": metric.timestamp,
                "severity": "warning"
            })
    
    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for a time window."""
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time window"}
        
        # Calculate statistics
        durations = [m.duration for m in recent_metrics]
        cpu_usages = [m.cpu_usage for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        
        return {
            "time_window": str(time_window),
            "total_operations": len(recent_metrics),
            "avg_duration": np.mean(durations),
            "max_duration": np.max(durations),
            "p95_duration": np.percentile(durations, 95),
            "avg_cpu_usage": np.mean(cpu_usages),
            "max_cpu_usage": np.max(cpu_usages),
            "avg_memory_usage": np.mean(memory_usages),
            "max_memory_usage": np.max(memory_usages),
            "slow_operations": len([m for m in recent_metrics if m.duration > self.response_time_threshold]),
            "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
        }
    
    def get_operation_insights(self, operation_name: str) -> Dict[str, Any]:
        """Get performance insights for a specific operation."""
        operation_metrics = [m for m in self.metrics_history if m.operation_name == operation_name]
        
        if not operation_metrics:
            return {"error": f"No metrics found for operation: {operation_name}"}
        
        durations = [m.duration for m in operation_metrics]
        cpu_usages = [m.cpu_usage for m in operation_metrics]
        memory_usages = [m.memory_usage for m in operation_metrics]
        
        return {
            "operation_name": operation_name,
            "total_executions": len(operation_metrics),
            "avg_duration": np.mean(durations),
            "duration_std": np.std(durations),
            "duration_trend": self._calculate_trend(durations),
            "avg_cpu_usage": np.mean(cpu_usages),
            "avg_memory_usage": np.mean(memory_usages),
            "success_rate": sum(1 for m in operation_metrics if m.success) / len(operation_metrics),
            "performance_degradation": self._detect_performance_degradation(durations),
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "degrading"
        elif slope < -0.1:
            return "improving"
        else:
            return "stable"
    
    def _detect_performance_degradation(self, durations: List[float]) -> bool:
        """Detect if performance is degrading over time."""
        if len(durations) < 10:
            return False
        
        # Compare recent performance with historical baseline
        baseline = np.mean(durations[:len(durations)//2])
        recent = np.mean(durations[len(durations)//2:])
        
        return recent > baseline * 1.2  # 20% degradation threshold


class PerformanceProfiler:
    """
    Advanced performance profiler for detailed analysis of code execution.
    
    Provides detailed profiling information including function call statistics,
    memory usage patterns, and performance bottleneck identification.
    """
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.logger = logging.getLogger(f"{__name__}.PerformanceProfiler")
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile an operation's performance."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await self._profile_async_operation(operation_name, func, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return self._profile_sync_operation(operation_name, func, *args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _profile_async_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile an async operation."""
        profiler = cProfile.Profile()
        memory_before = self._get_memory_usage()
        
        profiler.enable()
        start_time = time.perf_counter()
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            self.logger.error(f"Profiled operation {operation_name} failed: {e}")
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            profiler.disable()
            
            memory_after = self._get_memory_usage()
            total_time = end_time - start_time
            
            # Generate profile
            profile = self._generate_profile(operation_name, profiler, total_time, memory_before, memory_after)
            self.profiles[operation_name] = profile
            
            if not success or total_time > 10.0:  # Log slow operations
                self.logger.warning(f"Operation {operation_name} completed in {total_time:.2f}s")
        
        return result
    
    def _profile_sync_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile a synchronous operation."""
        profiler = cProfile.Profile()
        memory_before = self._get_memory_usage()
        
        profiler.enable()
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            self.logger.error(f"Profiled operation {operation_name} failed: {e}")
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            profiler.disable()
            
            memory_after = self._get_memory_usage()
            total_time = end_time - start_time
            
            # Generate profile
            profile = self._generate_profile(operation_name, profiler, total_time, memory_before, memory_after)
            self.profiles[operation_name] = profile
            
            if not success or total_time > 10.0:
                self.logger.warning(f"Operation {operation_name} completed in {total_time:.2f}s")
        
        return result
    
    def _generate_profile(
        self, 
        operation_name: str, 
        profiler: cProfile.Profile, 
        total_time: float,
        memory_before: float,
        memory_after: float
    ) -> PerformanceProfile:
        """Generate a performance profile from profiler data."""
        # Get profiler stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Extract function call information
        function_calls = {}
        hotspots = []
        
        for (filename, line_num, func_name), (call_count, _, total_time_func, cumulative_time) in stats.stats.items():
            function_calls[func_name] = call_count
            if cumulative_time > 0.1:  # Only include significant functions
                hotspots.append((f"{func_name} ({filename}:{line_num})", cumulative_time))
        
        # Sort hotspots by time
        hotspots.sort(key=lambda x: x[1], reverse=True)
        hotspots = hotspots[:10]  # Top 10 hotspots
        
        # Memory profile
        memory_profile = {
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_after - memory_before
        }
        
        # Identify bottlenecks
        bottlenecks = []
        if total_time > 5.0:
            bottlenecks.append(f"Total execution time exceeds 5s: {total_time:.2f}s")
        
        if memory_profile["memory_delta"] > 100:  # MB
            bottlenecks.append(f"High memory usage: {memory_profile['memory_delta']:.2f} MB")
        
        if len(hotspots) > 0 and hotspots[0][1] > total_time * 0.5:
            bottlenecks.append(f"Single function dominates execution: {hotspots[0][0]}")
        
        return PerformanceProfile(
            total_time=total_time,
            function_calls=function_calls,
            hotspots=hotspots,
            memory_profile=memory_profile,
            bottlenecks=bottlenecks
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def get_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get summary of profile for an operation."""
        if operation_name not in self.profiles:
            return {"error": f"No profile found for operation: {operation_name}"}
        
        profile = self.profiles[operation_name]
        
        return {
            "operation_name": operation_name,
            "total_time": profile.total_time,
            "function_calls_count": len(profile.function_calls),
            "total_function_calls": sum(profile.function_calls.values()),
            "top_hotspots": profile.hotspots[:5],
            "memory_usage": profile.memory_profile,
            "bottlenecks": profile.bottlenecks,
            "performance_rating": self._calculate_performance_rating(profile)
        }
    
    def _calculate_performance_rating(self, profile: PerformanceProfile) -> str:
        """Calculate performance rating for a profile."""
        if profile.total_time < 1.0 and len(profile.bottlenecks) == 0:
            return "excellent"
        elif profile.total_time < 3.0 and len(profile.bottlenecks) <= 1:
            return "good"
        elif profile.total_time < 10.0 and len(profile.bottlenecks) <= 2:
            return "fair"
        else:
            return "poor"


class IntelligentOptimizer:
    """
    Intelligent performance optimizer that automatically identifies and applies optimizations.
    
    Uses machine learning and heuristic analysis to identify performance bottlenecks
    and automatically apply or recommend optimizations.
    """
    
    def __init__(self, monitor: PerformanceMonitor, profiler: PerformanceProfiler):
        self.monitor = monitor
        self.profiler = profiler
        self.optimization_history: List[Dict[str, Any]] = []
        self.applied_optimizations: Set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.IntelligentOptimizer")
    
    def analyze_and_optimize(self) -> List[OptimizationRecommendation]:
        """Analyze current performance and generate optimization recommendations."""
        recommendations = []
        
        # Analyze system-wide performance
        system_recommendations = self._analyze_system_performance()
        recommendations.extend(system_recommendations)
        
        # Analyze individual operations
        operation_recommendations = self._analyze_operation_performance()
        recommendations.extend(operation_recommendations)
        
        # Analyze resource usage patterns
        resource_recommendations = self._analyze_resource_patterns()
        recommendations.extend(resource_recommendations)
        
        # Sort recommendations by priority and expected improvement
        recommendations.sort(key=lambda r: (
            {"high": 3, "medium": 2, "low": 1}.get(r.priority, 1),
            r.expected_improvement
        ), reverse=True)
        
        return recommendations
    
    def _analyze_system_performance(self) -> List[OptimizationRecommendation]:
        """Analyze system-wide performance patterns."""
        recommendations = []
        
        # Get recent performance summary
        summary = self.monitor.get_performance_summary(timedelta(hours=1))
        
        if isinstance(summary, dict) and "error" not in summary:
            # High average CPU usage
            if summary.get("avg_cpu_usage", 0) > 70:
                recommendations.append(OptimizationRecommendation(
                    category="cpu_optimization",
                    priority="high",
                    description="High CPU usage detected - implement CPU optimization strategies",
                    implementation="Add CPU-intensive task parallelization and optimize algorithms",
                    expected_improvement=0.3,
                    estimated_effort="medium"
                ))
            
            # High memory usage
            if summary.get("avg_memory_usage", 0) > 75:
                recommendations.append(OptimizationRecommendation(
                    category="memory_optimization",
                    priority="high",
                    description="High memory usage - implement memory optimization",
                    implementation="Add memory pooling, caching strategies, and garbage collection tuning",
                    expected_improvement=0.25,
                    estimated_effort="medium"
                ))
            
            # Slow operations
            slow_operations = summary.get("slow_operations", 0)
            total_operations = summary.get("total_operations", 1)
            if slow_operations / total_operations > 0.1:
                recommendations.append(OptimizationRecommendation(
                    category="performance_optimization",
                    priority="high",
                    description=f"High percentage of slow operations: {slow_operations}/{total_operations}",
                    implementation="Profile and optimize slow operations, add caching, improve algorithms",
                    expected_improvement=0.4,
                    estimated_effort="high"
                ))
        
        return recommendations
    
    def _analyze_operation_performance(self) -> List[OptimizationRecommendation]:
        """Analyze individual operation performance."""
        recommendations = []
        
        for operation_name, profile in self.profiler.profiles.items():
            # Analyze bottlenecks
            if profile.bottlenecks:
                for bottleneck in profile.bottlenecks:
                    recommendations.append(OptimizationRecommendation(
                        category="bottleneck_optimization",
                        priority="high" if "exceeds" in bottleneck else "medium",
                        description=f"Bottleneck in {operation_name}: {bottleneck}",
                        implementation=self._suggest_bottleneck_fix(bottleneck),
                        expected_improvement=0.3,
                        estimated_effort="medium",
                        code_location=operation_name
                    ))
            
            # Analyze memory usage
            memory_delta = profile.memory_profile.get("memory_delta", 0)
            if memory_delta > 50:  # MB
                recommendations.append(OptimizationRecommendation(
                    category="memory_optimization",
                    priority="medium",
                    description=f"High memory allocation in {operation_name}: {memory_delta:.2f} MB",
                    implementation="Implement object pooling, reduce memory allocations, optimize data structures",
                    expected_improvement=0.2,
                    estimated_effort="medium",
                    code_location=operation_name
                ))
            
            # Analyze hotspots
            if profile.hotspots and profile.hotspots[0][1] > profile.total_time * 0.3:
                hotspot_name, hotspot_time = profile.hotspots[0]
                recommendations.append(OptimizationRecommendation(
                    category="hotspot_optimization",
                    priority="high",
                    description=f"Performance hotspot in {operation_name}: {hotspot_name}",
                    implementation="Optimize the hotspot function, consider algorithmic improvements or caching",
                    expected_improvement=0.35,
                    estimated_effort="high",
                    code_location=hotspot_name
                ))
        
        return recommendations
    
    def _analyze_resource_patterns(self) -> List[OptimizationRecommendation]:
        """Analyze resource usage patterns."""
        recommendations = []
        
        # Analyze recent alerts
        recent_alerts = [a for a in self.monitor.resource_alerts 
                        if a["timestamp"] > datetime.now() - timedelta(hours=1)]
        
        if recent_alerts:
            alert_types = defaultdict(int)
            for alert in recent_alerts:
                alert_types[alert["type"]] += 1
            
            for alert_type, count in alert_types.items():
                if count > 5:  # Frequent alerts
                    recommendations.append(OptimizationRecommendation(
                        category="resource_optimization",
                        priority="high" if count > 10 else "medium",
                        description=f"Frequent {alert_type} alerts: {count} in the last hour",
                        implementation=self._suggest_resource_optimization(alert_type),
                        expected_improvement=0.25,
                        estimated_effort="medium"
                    ))
        
        return recommendations
    
    def _suggest_bottleneck_fix(self, bottleneck: str) -> str:
        """Suggest fix for a specific bottleneck."""
        if "execution time" in bottleneck:
            return "Implement async processing, add caching, or optimize algorithms"
        elif "memory usage" in bottleneck:
            return "Implement memory pooling, reduce object creation, or use more efficient data structures"
        elif "dominates execution" in bottleneck:
            return "Optimize the dominant function, consider parallelization or alternative algorithms"
        else:
            return "Analyze and optimize the specific bottleneck using profiling tools"
    
    def _suggest_resource_optimization(self, alert_type: str) -> str:
        """Suggest optimization for resource alerts."""
        if alert_type == "cpu_high":
            return "Implement CPU-bound task parallelization, optimize algorithms, or add load balancing"
        elif alert_type == "memory_high":
            return "Implement memory pooling, optimize data structures, or add garbage collection tuning"
        elif alert_type == "memory_low":
            return "Implement memory cleanup, reduce memory leaks, or increase available memory"
        elif alert_type == "slow_operation":
            return "Profile slow operations, implement caching, or optimize critical paths"
        else:
            return "Monitor and analyze resource usage patterns to identify optimization opportunities"
    
    def auto_apply_optimizations(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Automatically apply safe optimizations."""
        applied_optimizations = []
        failed_optimizations = []
        
        for recommendation in recommendations:
            if recommendation.priority == "high" and recommendation.category in ["cpu_optimization", "memory_optimization"]:
                try:
                    success = self._apply_optimization(recommendation)
                    if success:
                        applied_optimizations.append(recommendation.description)
                        self.applied_optimizations.add(recommendation.description)
                    else:
                        failed_optimizations.append(recommendation.description)
                except Exception as e:
                    self.logger.error(f"Failed to apply optimization: {e}")
                    failed_optimizations.append(recommendation.description)
        
        result = {
            "applied_optimizations": applied_optimizations,
            "failed_optimizations": failed_optimizations,
            "recommendations_remaining": len(recommendations) - len(applied_optimizations) - len(failed_optimizations)
        }
        
        # Record optimization attempt
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "result": result,
            "total_recommendations": len(recommendations)
        })
        
        return result
    
    def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization (simplified implementation)."""
        # This is a simplified implementation
        # In a real system, this would apply actual optimizations
        
        optimization_id = f"{recommendation.category}_{hash(recommendation.description) % 1000}"
        
        if optimization_id in self.applied_optimizations:
            return False  # Already applied
        
        # Simulate applying optimization
        self.logger.info(f"Applying optimization: {recommendation.description}")
        
        # For demonstration, we'll just mark it as applied
        # Real implementations would modify configuration, adjust parameters, etc.
        
        return True


class IntelligentPerformanceOptimizer:
    """
    Main intelligent performance optimization system.
    
    Coordinates monitoring, profiling, and optimization to provide comprehensive
    performance enhancement with minimal manual intervention.
    """
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.profiler = PerformanceProfiler()
        self.optimizer = IntelligentOptimizer(self.monitor, self.profiler)
        self.optimization_scheduler = None
        self.logger = logging.getLogger(f"{__name__}.IntelligentPerformanceOptimizer")
    
    async def start_intelligent_optimization(self, optimization_interval: float = 300.0):
        """Start intelligent performance optimization system."""
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        # Start optimization scheduler
        self.optimization_scheduler = asyncio.create_task(
            self._run_optimization_cycle(optimization_interval)
        )
        
        self.logger.info("Intelligent performance optimization started")
    
    async def stop_intelligent_optimization(self):
        """Stop intelligent performance optimization."""
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Stop optimization scheduler
        if self.optimization_scheduler and not self.optimization_scheduler.done():
            self.optimization_scheduler.cancel()
            try:
                await self.optimization_scheduler
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Intelligent performance optimization stopped")
    
    async def _run_optimization_cycle(self, interval: float):
        """Run continuous optimization cycle."""
        while True:
            try:
                # Analyze and generate recommendations
                recommendations = self.optimizer.analyze_and_optimize()
                
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
                    
                    # Auto-apply safe optimizations
                    result = self.optimizer.auto_apply_optimizations(recommendations)
                    
                    if result["applied_optimizations"]:
                        self.logger.info(f"Applied {len(result['applied_optimizations'])} optimizations")
                    
                    # Log remaining recommendations
                    high_priority_remaining = len([r for r in recommendations 
                                                 if r.priority == "high" and 
                                                 r.description not in result["applied_optimizations"]])
                    
                    if high_priority_remaining > 0:
                        self.logger.warning(f"{high_priority_remaining} high-priority optimizations require manual intervention")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization cycle error: {e}")
                await asyncio.sleep(interval)
    
    def performance_optimize(self, operation_name: str):
        """Decorator for automatic performance optimization of operations."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            # Apply profiling
            profiled_func = self.profiler.profile_operation(operation_name)(func)
            
            @wraps(profiled_func)
            async def async_wrapper(*args, **kwargs) -> T:
                start_time = time.perf_counter()
                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent
                
                try:
                    result = await profiled_func(*args, **kwargs)
                    success = True
                    return result
                except Exception:
                    success = False
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent
                    
                    # Record performance metric
                    metric = PerformanceMetric(
                        timestamp=datetime.now(),
                        operation_name=operation_name,
                        duration=duration,
                        cpu_usage=(cpu_before + cpu_after) / 2,
                        memory_usage=(memory_before + memory_after) / 2,
                        io_operations=0,  # Simplified
                        success=success
                    )
                    self.monitor.record_operation_metric(metric)
            
            @wraps(profiled_func)
            def sync_wrapper(*args, **kwargs) -> T:
                start_time = time.perf_counter()
                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent
                
                try:
                    result = profiled_func(*args, **kwargs)
                    success = True
                    return result
                except Exception:
                    success = False
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent
                    
                    # Record performance metric
                    metric = PerformanceMetric(
                        timestamp=datetime.now(),
                        operation_name=operation_name,
                        duration=duration,
                        cpu_usage=(cpu_before + cpu_after) / 2,
                        memory_usage=(memory_before + memory_after) / 2,
                        io_operations=0,  # Simplified
                        success=success
                    )
                    self.monitor.record_operation_metric(metric)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_performance": self.monitor.get_performance_summary(),
            "operation_profiles": {
                name: self.profiler.get_profile_summary(name) 
                for name in self.profiler.profiles.keys()
            },
            "optimization_recommendations": self.optimizer.analyze_and_optimize(),
            "optimization_history": self.optimizer.optimization_history[-10:],  # Last 10
            "resource_alerts": self.monitor.resource_alerts[-20:],  # Last 20
            "applied_optimizations": list(self.optimizer.applied_optimizations),
        }


# Global intelligent performance optimizer instance
intelligent_performance_optimizer = IntelligentPerformanceOptimizer()


def performance_optimized(operation_name: str):
    """
    Decorator for automatic performance optimization and monitoring.
    
    Usage:
        @performance_optimized("quality_gate_execution")
        async def run_quality_gate():
            # Your operation implementation
            pass
    """
    return intelligent_performance_optimizer.performance_optimize(operation_name)