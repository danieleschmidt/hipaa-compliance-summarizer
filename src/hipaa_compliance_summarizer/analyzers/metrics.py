"""Analytics metrics for comprehensive reporting and insights."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsMetrics:
    """Comprehensive analytics metrics container."""
    
    document_metrics: Dict[str, Any] = field(default_factory=dict)
    phi_metrics: Dict[str, Any] = field(default_factory=dict)
    compliance_metrics: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_metrics": self.document_metrics,
            "phi_metrics": self.phi_metrics,
            "compliance_metrics": self.compliance_metrics,
            "risk_metrics": self.risk_metrics,
            "performance_metrics": self.performance_metrics,
            "trend_analysis": self.trend_analysis,
            "generated_at": self.generated_at.isoformat()
        }


class MetricsAggregator:
    """Aggregates metrics from various analysis components."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics_history: List[AnalyticsMetrics] = []
        self.max_history_size = 1000
    
    def aggregate_metrics(self, document_analyses: List[Any] = None,
                         phi_analyses: List[Any] = None,
                         compliance_analyses: List[Any] = None,
                         risk_analyses: List[Any] = None,
                         time_window_hours: int = 24) -> AnalyticsMetrics:
        """Aggregate metrics from various analysis results.
        
        Args:
            document_analyses: List of document analysis results
            phi_analyses: List of PHI analysis results
            compliance_analyses: List of compliance analysis results
            risk_analyses: List of risk analysis results
            time_window_hours: Time window for analysis
            
        Returns:
            Aggregated analytics metrics
        """
        logger.info(f"Aggregating metrics for {time_window_hours} hour window")
        
        # Filter results by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Document metrics
        document_metrics = self._aggregate_document_metrics(
            document_analyses or [], cutoff_time
        )
        
        # PHI metrics
        phi_metrics = self._aggregate_phi_metrics(
            phi_analyses or [], cutoff_time
        )
        
        # Compliance metrics
        compliance_metrics = self._aggregate_compliance_metrics(
            compliance_analyses or [], cutoff_time
        )
        
        # Risk metrics
        risk_metrics = self._aggregate_risk_metrics(
            risk_analyses or [], cutoff_time
        )
        
        # Performance metrics (would integrate with monitoring system)
        performance_metrics = self._aggregate_performance_metrics(time_window_hours)
        
        # Trend analysis
        trend_analysis = self._perform_trend_analysis(
            document_analyses, phi_analyses, compliance_analyses, risk_analyses, cutoff_time
        )
        
        metrics = AnalyticsMetrics(
            document_metrics=document_metrics,
            phi_metrics=phi_metrics,
            compliance_metrics=compliance_metrics,
            risk_metrics=risk_metrics,
            performance_metrics=performance_metrics,
            trend_analysis=trend_analysis
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        logger.info("Metrics aggregation completed")
        return metrics
    
    def _aggregate_document_metrics(self, analyses: List[Any], cutoff_time: datetime) -> Dict[str, Any]:
        """Aggregate document analysis metrics."""
        if not analyses:
            return {}
        
        # Filter by time
        recent_analyses = [a for a in analyses if a.analysis_timestamp >= cutoff_time]
        
        if not recent_analyses:
            return {}
        
        # Calculate metrics
        word_counts = [a.word_count for a in recent_analyses]
        readability_scores = [a.readability_score for a in recent_analyses]
        complexity_scores = [a.complexity_score for a in recent_analyses]
        medical_densities = [a.medical_terminology_density for a in recent_analyses]
        
        # Document type distribution
        doc_types = Counter(a.document_type for a in recent_analyses)
        
        # Language distribution
        languages = Counter(a.language for a in recent_analyses)
        
        return {
            "total_documents": len(recent_analyses),
            "average_word_count": statistics.mean(word_counts),
            "median_word_count": statistics.median(word_counts),
            "average_readability": statistics.mean(readability_scores),
            "average_complexity": statistics.mean(complexity_scores),
            "average_medical_density": statistics.mean(medical_densities),
            "document_types": dict(doc_types),
            "languages": dict(languages),
            "total_phi_sections": sum(len(a.potential_phi_sections) for a in recent_analyses),
            "structured_data_summary": self._summarize_structured_data(recent_analyses)
        }
    
    def _summarize_structured_data(self, analyses: List[Any]) -> Dict[str, int]:
        """Summarize structured data elements across documents."""
        element_totals = defaultdict(int)
        
        for analysis in analyses:
            for element_type, count in analysis.structured_data_elements.items():
                element_totals[element_type] += count
        
        return dict(element_totals)
    
    def _aggregate_phi_metrics(self, analyses: List[Any], cutoff_time: datetime) -> Dict[str, Any]:
        """Aggregate PHI analysis metrics."""
        if not analyses:
            return {}
        
        # Filter by time
        recent_analyses = [a for a in analyses if a.analysis_timestamp >= cutoff_time]
        
        if not recent_analyses:
            return {}
        
        # Calculate metrics
        total_entities = sum(a.total_phi_entities for a in recent_analyses)
        phi_densities = [a.phi_density for a in recent_analyses]
        risk_scores = [a.privacy_risk_score for a in recent_analyses]
        
        # Category distribution
        category_totals = defaultdict(int)
        for analysis in recent_analyses:
            for category, count in analysis.phi_by_category.items():
                category_totals[category] += count
        
        # High-risk categories
        high_risk_docs = sum(1 for a in recent_analyses if a.high_risk_categories)
        
        # Redaction complexity distribution
        complexity_dist = Counter(a.redaction_complexity for a in recent_analyses)
        
        # Sensitive patterns
        all_patterns = []
        for analysis in recent_analyses:
            all_patterns.extend([p["pattern_name"] for p in analysis.sensitive_patterns])
        
        pattern_frequency = Counter(all_patterns)
        
        return {
            "total_phi_entities": total_entities,
            "average_entities_per_document": total_entities / len(recent_analyses),
            "average_phi_density": statistics.mean(phi_densities),
            "average_privacy_risk": statistics.mean(risk_scores),
            "category_distribution": dict(category_totals),
            "high_risk_documents": high_risk_docs,
            "redaction_complexity_distribution": dict(complexity_dist),
            "common_sensitive_patterns": dict(pattern_frequency.most_common(10)),
            "co_occurrence_summary": self._summarize_co_occurrences(recent_analyses)
        }
    
    def _summarize_co_occurrences(self, analyses: List[Any]) -> Dict[str, int]:
        """Summarize PHI co-occurrence patterns."""
        co_occurrence_pairs = defaultdict(int)
        
        for analysis in analyses:
            for category, co_categories in analysis.co_occurrence_analysis.items():
                for co_category in co_categories:
                    pair = tuple(sorted([category, co_category]))
                    co_occurrence_pairs[f"{pair[0]} + {pair[1]}"] += 1
        
        return dict(Counter(co_occurrence_pairs).most_common(10))
    
    def _aggregate_compliance_metrics(self, analyses: List[Any], cutoff_time: datetime) -> Dict[str, Any]:
        """Aggregate compliance analysis metrics."""
        if not analyses:
            return {}
        
        # Filter by time
        recent_analyses = [a for a in analyses if a.analysis_timestamp >= cutoff_time]
        
        if not recent_analyses:
            return {}
        
        # Calculate metrics
        compliance_scores = [a.overall_compliance_score for a in recent_analyses]
        compliance_statuses = [a.compliance_status for a in recent_analyses]
        
        # Safe Harbor compliance
        safe_harbor_compliant = sum(1 for a in recent_analyses if a.safe_harbor_compliance)
        
        # Violations analysis
        all_violations = []
        for analysis in recent_analyses:
            all_violations.extend(analysis.violations)
        
        violation_types = Counter(v["type"] for v in all_violations)
        violation_severities = Counter(v["severity"] for v in all_violations)
        
        # Risk assessments
        risk_levels = [a.risk_assessment["risk_level"] for a in recent_analyses]
        immediate_attention = sum(1 for a in recent_analyses 
                                if a.risk_assessment.get("requires_immediate_attention", False))
        
        return {
            "average_compliance_score": statistics.mean(compliance_scores),
            "median_compliance_score": statistics.median(compliance_scores),
            "compliance_distribution": dict(Counter(compliance_statuses)),
            "safe_harbor_compliance_rate": safe_harbor_compliant / len(recent_analyses) * 100,
            "total_violations": len(all_violations),
            "violation_types": dict(violation_types),
            "violation_severities": dict(violation_severities),
            "risk_level_distribution": dict(Counter(risk_levels)),
            "documents_needing_attention": immediate_attention,
            "compliance_trends": self._calculate_compliance_trends(recent_analyses)
        }
    
    def _calculate_compliance_trends(self, analyses: List[Any]) -> Dict[str, Any]:
        """Calculate compliance trend indicators."""
        if len(analyses) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_analyses = sorted(analyses, key=lambda a: a.analysis_timestamp)
        
        # Split into two halves for comparison
        mid_point = len(sorted_analyses) // 2
        first_half = sorted_analyses[:mid_point]
        second_half = sorted_analyses[mid_point:]
        
        first_avg = statistics.mean(a.overall_compliance_score for a in first_half)
        second_avg = statistics.mean(a.overall_compliance_score for a in second_half)
        
        trend_direction = "improving" if second_avg > first_avg else "declining" if second_avg < first_avg else "stable"
        trend_magnitude = abs(second_avg - first_avg)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "first_period_avg": first_avg,
            "second_period_avg": second_avg
        }
    
    def _aggregate_risk_metrics(self, analyses: List[Any], cutoff_time: datetime) -> Dict[str, Any]:
        """Aggregate risk analysis metrics."""
        if not analyses:
            return {}
        
        # Filter by time
        recent_analyses = [a for a in analyses if a.analysis_timestamp >= cutoff_time]
        
        if not recent_analyses:
            return {}
        
        # Calculate metrics
        risk_scores = [a.overall_risk_score for a in recent_analyses]
        risk_levels = [a.risk_level for a in recent_analyses]
        breach_probabilities = [a.breach_probability for a in recent_analyses]
        
        # Risk factors analysis
        all_risk_factors = []
        for analysis in recent_analyses:
            all_risk_factors.extend([f["type"] for f in analysis.risk_factors])
        
        risk_factor_frequency = Counter(all_risk_factors)
        
        # High-risk documents
        high_risk_count = sum(1 for a in recent_analyses if a.risk_level in ["high", "critical"])
        
        return {
            "average_risk_score": statistics.mean(risk_scores),
            "median_risk_score": statistics.median(risk_scores),
            "max_risk_score": max(risk_scores),
            "risk_level_distribution": dict(Counter(risk_levels)),
            "average_breach_probability": statistics.mean(breach_probabilities),
            "high_risk_documents": high_risk_count,
            "high_risk_percentage": high_risk_count / len(recent_analyses) * 100,
            "common_risk_factors": dict(risk_factor_frequency.most_common(10)),
            "risk_trends": self._calculate_risk_trends(recent_analyses)
        }
    
    def _calculate_risk_trends(self, analyses: List[Any]) -> Dict[str, Any]:
        """Calculate risk trend indicators."""
        if len(analyses) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_analyses = sorted(analyses, key=lambda a: a.analysis_timestamp)
        
        # Calculate trend over time
        risk_scores = [a.overall_risk_score for a in sorted_analyses]
        
        # Simple linear trend
        if len(risk_scores) >= 3:
            first_third = statistics.mean(risk_scores[:len(risk_scores)//3])
            last_third = statistics.mean(risk_scores[-len(risk_scores)//3:])
            
            if last_third > first_third + 0.1:
                trend = "increasing"
            elif last_third < first_third - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "volatility": statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0
        }
    
    def _aggregate_performance_metrics(self, time_window_hours: int) -> Dict[str, Any]:
        """Aggregate performance metrics (placeholder for integration with monitoring)."""
        # This would integrate with the monitoring system's metrics
        return {
            "processing_throughput": {
                "documents_per_hour": 10,  # Placeholder
                "average_processing_time_ms": 5000  # Placeholder
            },
            "system_performance": {
                "average_cpu_usage": 45.0,  # Placeholder
                "average_memory_usage": 60.0  # Placeholder
            },
            "api_performance": {
                "average_response_time_ms": 1200,  # Placeholder
                "error_rate": 0.02  # Placeholder
            }
        }
    
    def _perform_trend_analysis(self, document_analyses: List[Any], phi_analyses: List[Any],
                              compliance_analyses: List[Any], risk_analyses: List[Any],
                              cutoff_time: datetime) -> Dict[str, Any]:
        """Perform comprehensive trend analysis across all metrics."""
        trends = {}
        
        # Document processing trends
        if document_analyses:
            recent_docs = [a for a in document_analyses if a.analysis_timestamp >= cutoff_time]
            if recent_docs:
                trends["document_processing"] = {
                    "volume_trend": self._calculate_volume_trend(recent_docs),
                    "complexity_trend": self._calculate_complexity_trend(recent_docs)
                }
        
        # PHI detection trends
        if phi_analyses:
            recent_phi = [a for a in phi_analyses if a.analysis_timestamp >= cutoff_time]
            if recent_phi:
                trends["phi_detection"] = {
                    "density_trend": self._calculate_phi_density_trend(recent_phi),
                    "category_trends": self._calculate_category_trends(recent_phi)
                }
        
        # Compliance trends
        if compliance_analyses:
            recent_compliance = [a for a in compliance_analyses if a.analysis_timestamp >= cutoff_time]
            if recent_compliance:
                trends["compliance"] = self._calculate_compliance_trends(recent_compliance)
        
        # Risk trends
        if risk_analyses:
            recent_risk = [a for a in risk_analyses if a.analysis_timestamp >= cutoff_time]
            if recent_risk:
                trends["risk"] = self._calculate_risk_trends(recent_risk)
        
        return trends
    
    def _calculate_volume_trend(self, analyses: List[Any]) -> str:
        """Calculate document volume trend."""
        if len(analyses) < 2:
            return "stable"
        
        # Group by day and count
        daily_counts = defaultdict(int)
        for analysis in analyses:
            day = analysis.analysis_timestamp.date()
            daily_counts[day] += 1
        
        if len(daily_counts) < 2:
            return "stable"
        
        sorted_days = sorted(daily_counts.keys())
        first_half_avg = statistics.mean(daily_counts[day] for day in sorted_days[:len(sorted_days)//2])
        second_half_avg = statistics.mean(daily_counts[day] for day in sorted_days[len(sorted_days)//2:])
        
        if second_half_avg > first_half_avg * 1.2:
            return "increasing"
        elif second_half_avg < first_half_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_complexity_trend(self, analyses: List[Any]) -> str:
        """Calculate document complexity trend."""
        if len(analyses) < 2:
            return "stable"
        
        sorted_analyses = sorted(analyses, key=lambda a: a.analysis_timestamp)
        complexity_scores = [a.complexity_score for a in sorted_analyses]
        
        first_half = complexity_scores[:len(complexity_scores)//2]
        second_half = complexity_scores[len(complexity_scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.1:
            return "increasing"
        elif second_avg < first_avg - 0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_phi_density_trend(self, analyses: List[Any]) -> str:
        """Calculate PHI density trend."""
        if len(analyses) < 2:
            return "stable"
        
        sorted_analyses = sorted(analyses, key=lambda a: a.analysis_timestamp)
        density_scores = [a.phi_density for a in sorted_analyses]
        
        first_half = density_scores[:len(density_scores)//2]
        second_half = density_scores[len(density_scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.2:
            return "increasing"
        elif second_avg < first_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_category_trends(self, analyses: List[Any]) -> Dict[str, str]:
        """Calculate trends for specific PHI categories."""
        category_trends = {}
        
        # Collect category counts over time
        category_time_series = defaultdict(list)
        
        for analysis in sorted(analyses, key=lambda a: a.analysis_timestamp):
            for category, count in analysis.phi_by_category.items():
                category_time_series[category].append(count)
        
        # Calculate trend for each category
        for category, counts in category_time_series.items():
            if len(counts) >= 2:
                first_half = counts[:len(counts)//2]
                second_half = counts[len(counts)//2:]
                
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                if second_avg > first_avg * 1.3:
                    category_trends[category] = "increasing"
                elif second_avg < first_avg * 0.7:
                    category_trends[category] = "decreasing"
                else:
                    category_trends[category] = "stable"
        
        return category_trends
    
    def get_historical_metrics(self, hours_back: int = 168) -> List[AnalyticsMetrics]:
        """Get historical metrics within specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        return [m for m in self.metrics_history if m.generated_at >= cutoff_time]
    
    def generate_executive_summary(self, metrics: AnalyticsMetrics) -> Dict[str, Any]:
        """Generate executive summary of key metrics."""
        summary = {
            "period": "last_24_hours",
            "generated_at": datetime.utcnow().isoformat(),
            "key_metrics": {
                "documents_processed": metrics.document_metrics.get("total_documents", 0),
                "average_compliance_score": metrics.compliance_metrics.get("average_compliance_score", 0),
                "high_risk_documents": metrics.risk_metrics.get("high_risk_documents", 0),
                "total_phi_entities": metrics.phi_metrics.get("total_phi_entities", 0)
            },
            "alerts": [],
            "recommendations": []
        }
        
        # Generate alerts based on metrics
        compliance_score = metrics.compliance_metrics.get("average_compliance_score", 100)
        if compliance_score < 85:
            summary["alerts"].append({
                "type": "compliance_warning",
                "message": f"Average compliance score is {compliance_score:.1f}% - below target"
            })
        
        high_risk_pct = metrics.risk_metrics.get("high_risk_percentage", 0)
        if high_risk_pct > 20:
            summary["alerts"].append({
                "type": "risk_warning",
                "message": f"{high_risk_pct:.1f}% of documents are high risk"
            })
        
        # Generate recommendations
        if compliance_score < 90:
            summary["recommendations"].append("Focus on improving redaction processes")
        
        if high_risk_pct > 15:
            summary["recommendations"].append("Implement enhanced risk mitigation strategies")
        
        return summary