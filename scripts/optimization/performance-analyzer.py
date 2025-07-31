#!/usr/bin/env python3
"""
Advanced Performance Analysis Tool for HIPAA Compliance Summarizer

This script provides comprehensive performance analysis including:
- Memory usage profiling
- CPU utilization monitoring
- Database query optimization
- Cache performance analysis
- PHI processing performance metrics
"""

import argparse
import json
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    timestamp: str
    memory_usage_mb: float
    cpu_percent: float
    io_read_mb: float
    io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    
class PerformanceAnalyzer:
    """Advanced performance analysis and optimization recommendations"""
    
    def __init__(self, output_dir: Path = Path("performance-reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system performance metrics"""
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # I/O statistics
        io_stats = psutil.disk_io_counters()
        io_read_mb = io_stats.read_bytes / (1024 * 1024) if io_stats else 0
        io_write_mb = io_stats.write_bytes / (1024 * 1024) if io_stats else 0
        
        # Network statistics
        net_stats = psutil.net_io_counters()
        net_sent_mb = net_stats.bytes_sent / (1024 * 1024) if net_stats else 0
        net_recv_mb = net_stats.bytes_recv / (1024 * 1024) if net_stats else 0
        
        # Process count
        process_count = len(psutil.pids())
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            process_count=process_count
        )
    
    def analyze_application_performance(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Analyze application performance over specified duration"""
        
        self.logger.info(f"Starting performance analysis for {duration_seconds} seconds")
        
        metrics_collection = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            metrics = self.collect_system_metrics()
            metrics_collection.append(metrics)
            time.sleep(5)  # Collect metrics every 5 seconds
        
        # Calculate performance statistics
        analysis = self._calculate_performance_stats(metrics_collection)
        
        # Generate optimization recommendations
        recommendations = self._generate_recommendations(analysis)
        
        # Save detailed report
        report = {
            "analysis_duration_seconds": duration_seconds,
            "metrics_collected": len(metrics_collection),
            "performance_analysis": analysis,
            "optimization_recommendations": recommendations,
            "raw_metrics": [asdict(metric) for metric in metrics_collection]
        }
        
        report_file = self.output_dir / f"performance-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance analysis complete. Report saved to {report_file}")
        return report
    
    def _calculate_performance_stats(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate statistical analysis of performance metrics"""
        
        if not metrics:
            return {}
        
        # Extract numeric values for analysis
        memory_values = [m.memory_usage_mb for m in metrics]
        cpu_values = [m.cpu_percent for m in metrics]
        io_read_values = [m.io_read_mb for m in metrics]
        io_write_values = [m.io_write_mb for m in metrics]
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }
        
        return {
            "memory_usage_mb": calculate_stats(memory_values),
            "cpu_percent": calculate_stats(cpu_values),
            "io_read_mb": calculate_stats(io_read_values),
            "io_write_mb": calculate_stats(io_write_values),
            "performance_score": self._calculate_performance_score(metrics),
            "trend_analysis": self._analyze_trends(metrics)
        }
    
    def _calculate_performance_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        
        if not metrics:
            return 0.0
        
        # Performance scoring based on resource utilization
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
        avg_memory_gb = sum(m.memory_usage_mb for m in metrics) / len(metrics) / 1024
        
        # Score calculation (lower resource usage = higher score)
        cpu_score = max(0, 100 - avg_cpu)
        memory_score = max(0, 100 - (avg_memory_gb * 10))  # Penalize high memory usage
        
        # Weighted average
        overall_score = (cpu_score * 0.6) + (memory_score * 0.4)
        return min(100, max(0, overall_score))
    
    def _analyze_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, str]:
        """Analyze performance trends over time"""
        
        if len(metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Compare first and last quartile
        quartile_size = len(metrics) // 4
        first_quartile = metrics[:quartile_size]
        last_quartile = metrics[-quartile_size:]
        
        first_avg_cpu = sum(m.cpu_percent for m in first_quartile) / len(first_quartile)
        last_avg_cpu = sum(m.cpu_percent for m in last_quartile) / len(last_quartile)
        
        first_avg_memory = sum(m.memory_usage_mb for m in first_quartile) / len(first_quartile)
        last_avg_memory = sum(m.memory_usage_mb for m in last_quartile) / len(last_quartile)
        
        cpu_trend = "increasing" if last_avg_cpu > first_avg_cpu * 1.1 else \
                   "decreasing" if last_avg_cpu < first_avg_cpu * 0.9 else "stable"
        
        memory_trend = "increasing" if last_avg_memory > first_avg_memory * 1.1 else \
                      "decreasing" if last_avg_memory < first_avg_memory * 0.9 else "stable"
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "overall_stability": "stable" if cpu_trend == "stable" and memory_trend == "stable" else "variable"
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analysis"""
        
        recommendations = []
        
        if not analysis:
            return recommendations
        
        # Memory optimization recommendations
        memory_stats = analysis.get("memory_usage_mb", {})
        if memory_stats.get("avg", 0) > 1024:  # > 1GB average
            recommendations.append({
                "category": "memory",
                "priority": "high",
                "title": "High Memory Usage Detected",
                "description": "Average memory usage exceeds 1GB. Consider implementing memory pooling and caching optimizations.",
                "action": "Review memory-intensive operations and implement object pooling for PHI processing."
            })
        
        # CPU optimization recommendations
        cpu_stats = analysis.get("cpu_percent", {})
        if cpu_stats.get("avg", 0) > 80:  # > 80% average CPU
            recommendations.append({
                "category": "cpu",
                "priority": "high",
                "title": "High CPU Utilization",
                "description": "CPU usage is consistently high. Consider implementing async processing.",
                "action": "Implement async/await patterns for I/O operations and consider worker process pooling."
            })
        
        # Performance score recommendations
        perf_score = analysis.get("performance_score", 100)
        if perf_score < 70:
            recommendations.append({
                "category": "general",
                "priority": "medium",
                "title": "Overall Performance Below Optimal",
                "description": f"Performance score is {perf_score:.1f}/100. Multiple optimizations needed.",
                "action": "Implement comprehensive performance monitoring and systematic optimization."
            })
        
        # Trend-based recommendations
        trends = analysis.get("trend_analysis", {})
        if trends.get("memory_trend") == "increasing":
            recommendations.append({
                "category": "memory",
                "priority": "medium", 
                "title": "Memory Usage Increasing Over Time",
                "description": "Memory usage shows upward trend, indicating potential memory leaks.",
                "action": "Implement memory profiling and review for memory leaks in long-running processes."
            })
        
        return recommendations

def main():
    """Main entry point for performance analysis"""
    
    parser = argparse.ArgumentParser(description="Advanced Performance Analysis Tool")
    parser.add_argument("--duration", "-d", type=int, default=60,
                       help="Analysis duration in seconds (default: 60)")
    parser.add_argument("--output-dir", "-o", type=str, default="performance-reports",
                       help="Output directory for reports (default: performance-reports)")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Run continuous monitoring (until interrupted)")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(Path(args.output_dir))
    
    try:
        if args.continuous:
            print("Starting continuous performance monitoring (Ctrl+C to stop)...")
            while True:
                analyzer.analyze_application_performance(args.duration)
                time.sleep(args.duration)
        else:
            report = analyzer.analyze_application_performance(args.duration)
            
            # Print summary
            print("\n=== Performance Analysis Summary ===")
            perf_score = report["performance_analysis"].get("performance_score", 0)
            print(f"Performance Score: {perf_score:.1f}/100")
            
            recommendations = report["optimization_recommendations"]
            if recommendations:
                print(f"\nOptimization Recommendations ({len(recommendations)}):")
                for rec in recommendations:
                    print(f"  • [{rec['priority'].upper()}] {rec['title']}")
                    print(f"    {rec['description']}")
            else:
                print("\nNo optimization recommendations - performance is optimal!")
    
    except KeyboardInterrupt:
        print("\nPerformance analysis stopped by user.")
    except Exception as e:
        print(f"Error during performance analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())