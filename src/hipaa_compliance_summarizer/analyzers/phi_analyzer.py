"""PHI-specific analysis for HIPAA compliance."""

import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List

from ..models.phi_entity import PHICategory
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


@dataclass
class PHIAnalysisResult:
    """Result of PHI-specific analysis."""

    document_id: str
    total_phi_entities: int
    phi_by_category: Dict[str, int]
    phi_density: float
    high_risk_categories: List[str]
    privacy_risk_score: float
    redaction_complexity: str
    sensitive_patterns: List[Dict[str, Any]]
    co_occurrence_analysis: Dict[str, List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "total_phi_entities": self.total_phi_entities,
            "phi_by_category": self.phi_by_category,
            "phi_density": self.phi_density,
            "high_risk_categories": self.high_risk_categories,
            "privacy_risk_score": self.privacy_risk_score,
            "redaction_complexity": self.redaction_complexity,
            "sensitive_patterns": self.sensitive_patterns,
            "co_occurrence_analysis": self.co_occurrence_analysis,
            "metadata": self.metadata,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


class PHIAnalyzer:
    """Specialized analyzer for PHI detection and risk assessment."""

    def __init__(self):
        """Initialize PHI analyzer."""
        # Risk levels for different PHI categories
        self.category_risk_levels = {
            PHICategory.NAMES.value: 0.8,
            PHICategory.SOCIAL_SECURITY_NUMBERS.value: 1.0,
            PHICategory.ACCOUNT_NUMBERS.value: 0.9,
            PHICategory.CERTIFICATE_NUMBERS.value: 0.7,
            PHICategory.VEHICLE_IDENTIFIERS.value: 0.6,
            PHICategory.DEVICE_IDENTIFIERS.value: 0.7,
            PHICategory.WEB_URLS.value: 0.5,
            PHICategory.IP_ADDRESSES.value: 0.6,
            PHICategory.BIOMETRIC_IDENTIFIERS.value: 1.0,
            PHICategory.FULL_FACE_PHOTOS.value: 0.9,
            PHICategory.OTHER_IDENTIFYING_NUMBERS.value: 0.7,
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value: 0.6,
            PHICategory.DATES.value: 0.4,
            PHICategory.TELEPHONE_NUMBERS.value: 0.7,
            PHICategory.FAX_NUMBERS.value: 0.6,
            PHICategory.EMAIL_ADDRESSES.value: 0.8,
            PHICategory.HEALTH_PLAN_NUMBERS.value: 0.9,
            PHICategory.MEDICAL_RECORD_NUMBERS.value: 0.9
        }

        # High-risk combinations (PHI categories that are dangerous together)
        self.high_risk_combinations = [
            [PHICategory.NAMES.value, PHICategory.DATES.value, PHICategory.GEOGRAPHIC_SUBDIVISIONS.value],
            [PHICategory.SOCIAL_SECURITY_NUMBERS.value, PHICategory.NAMES.value],
            [PHICategory.MEDICAL_RECORD_NUMBERS.value, PHICategory.NAMES.value],
            [PHICategory.NAMES.value, PHICategory.TELEPHONE_NUMBERS.value, PHICategory.EMAIL_ADDRESSES.value]
        ]

        # Sensitive pattern templates
        self.sensitive_patterns = {
            "identity_cluster": {
                "description": "Cluster of identifying information",
                "risk_level": "high",
                "categories": [PHICategory.NAMES.value, PHICategory.DATES.value, PHICategory.GEOGRAPHIC_SUBDIVISIONS.value]
            },
            "financial_cluster": {
                "description": "Financial/account information cluster",
                "risk_level": "critical",
                "categories": [PHICategory.ACCOUNT_NUMBERS.value, PHICategory.SOCIAL_SECURITY_NUMBERS.value]
            },
            "contact_cluster": {
                "description": "Complete contact information",
                "risk_level": "high",
                "categories": [PHICategory.TELEPHONE_NUMBERS.value, PHICategory.EMAIL_ADDRESSES.value, PHICategory.GEOGRAPHIC_SUBDIVISIONS.value]
            },
            "medical_cluster": {
                "description": "Medical identification cluster",
                "risk_level": "high",
                "categories": [PHICategory.MEDICAL_RECORD_NUMBERS.value, PHICategory.HEALTH_PLAN_NUMBERS.value]
            }
        }

        # Redaction complexity factors
        self.redaction_complexity_factors = {
            "simple": {"max_entities": 5, "max_categories": 2, "max_risk_score": 0.3},
            "moderate": {"max_entities": 15, "max_categories": 4, "max_risk_score": 0.6},
            "complex": {"max_entities": 30, "max_categories": 6, "max_risk_score": 0.8},
            "very_complex": {"max_entities": float('inf'), "max_categories": float('inf'), "max_risk_score": 1.0}
        }

    @trace_operation("phi_analysis")
    def analyze_phi_distribution(self, phi_entities: List[Any], content: str,
                                document_id: str = None) -> PHIAnalysisResult:
        """Analyze PHI distribution and risk patterns in a document.
        
        Args:
            phi_entities: List of detected PHI entities
            content: Original document content
            document_id: Optional document identifier
            
        Returns:
            PHI analysis result
        """
        logger.info(f"Starting PHI analysis for document: {document_id}")

        # Basic PHI statistics
        total_entities = len(phi_entities)
        phi_by_category = self._count_phi_by_category(phi_entities)

        # Calculate PHI density
        word_count = len(re.findall(r'\b\w+\b', content))
        phi_density = total_entities / word_count if word_count > 0 else 0.0

        # Identify high-risk categories
        high_risk_categories = self._identify_high_risk_categories(phi_by_category)

        # Calculate privacy risk score
        privacy_risk_score = self._calculate_privacy_risk(phi_entities, phi_by_category)

        # Determine redaction complexity
        redaction_complexity = self._assess_redaction_complexity(
            total_entities, len(phi_by_category), privacy_risk_score
        )

        # Identify sensitive patterns
        sensitive_patterns = self._identify_sensitive_patterns(phi_by_category, phi_entities)

        # Analyze co-occurrences
        co_occurrence_analysis = self._analyze_co_occurrences(phi_entities, content)

        result = PHIAnalysisResult(
            document_id=document_id or "unknown",
            total_phi_entities=total_entities,
            phi_by_category=phi_by_category,
            phi_density=phi_density,
            high_risk_categories=high_risk_categories,
            privacy_risk_score=privacy_risk_score,
            redaction_complexity=redaction_complexity,
            sensitive_patterns=sensitive_patterns,
            co_occurrence_analysis=co_occurrence_analysis
        )

        logger.info(f"PHI analysis completed: {total_entities} entities, risk score: {privacy_risk_score:.2f}")
        return result

    def _count_phi_by_category(self, phi_entities: List[Any]) -> Dict[str, int]:
        """Count PHI entities by category."""
        category_counts = {}

        for entity in phi_entities:
            category = getattr(entity, 'category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        return category_counts

    def _identify_high_risk_categories(self, phi_by_category: Dict[str, int]) -> List[str]:
        """Identify high-risk PHI categories present in the document."""
        high_risk_categories = []

        for category, count in phi_by_category.items():
            if count > 0:
                risk_level = self.category_risk_levels.get(category, 0.5)
                if risk_level >= 0.8:  # High risk threshold
                    high_risk_categories.append(category)

        return high_risk_categories

    def _calculate_privacy_risk(self, phi_entities: List[Any],
                               phi_by_category: Dict[str, int]) -> float:
        """Calculate overall privacy risk score."""
        if not phi_entities:
            return 0.0

        risk_factors = []

        # Base risk from individual categories
        for category, count in phi_by_category.items():
            category_risk = self.category_risk_levels.get(category, 0.5)
            # Risk increases with count but with diminishing returns
            normalized_count = min(count / 5.0, 1.0)  # Cap at 5 entities
            risk_factors.append(category_risk * normalized_count)

        # Risk from dangerous combinations
        combination_risk = self._assess_combination_risk(phi_by_category)
        if combination_risk > 0:
            risk_factors.append(combination_risk)

        # Risk from PHI density
        total_entities = sum(phi_by_category.values())
        density_risk = min(total_entities / 20.0, 1.0)  # Cap at 20 entities
        risk_factors.append(density_risk)

        # Calculate weighted average
        if risk_factors:
            base_risk = statistics.mean(risk_factors)
            # Apply combination multiplier if dangerous combinations exist
            if combination_risk > 0:
                base_risk *= 1.2  # 20% increase for dangerous combinations

            return min(base_risk, 1.0)  # Cap at 1.0

        return 0.0

    def _assess_combination_risk(self, phi_by_category: Dict[str, int]) -> float:
        """Assess risk from dangerous PHI category combinations."""
        present_categories = set(cat for cat, count in phi_by_category.items() if count > 0)

        max_combination_risk = 0.0

        for risk_combination in self.high_risk_combinations:
            # Check how many categories from this risky combination are present
            overlap = len(set(risk_combination) & present_categories)
            overlap_ratio = overlap / len(risk_combination)

            if overlap_ratio >= 0.67:  # At least 2/3 of the risky combination is present
                combination_risk = overlap_ratio * 0.8  # Max 0.8 risk from combinations
                max_combination_risk = max(max_combination_risk, combination_risk)

        return max_combination_risk

    def _assess_redaction_complexity(self, total_entities: int,
                                   category_count: int, risk_score: float) -> str:
        """Assess the complexity of redacting PHI from this document."""
        for complexity, thresholds in self.redaction_complexity_factors.items():
            if (total_entities <= thresholds["max_entities"] and
                category_count <= thresholds["max_categories"] and
                risk_score <= thresholds["max_risk_score"]):
                return complexity

        return "very_complex"

    def _identify_sensitive_patterns(self, phi_by_category: Dict[str, int],
                                   phi_entities: List[Any]) -> List[Dict[str, Any]]:
        """Identify sensitive PHI patterns in the document."""
        sensitive_patterns = []
        present_categories = set(cat for cat, count in phi_by_category.items() if count > 0)

        for pattern_name, pattern_info in self.sensitive_patterns.items():
            required_categories = set(pattern_info["categories"])
            overlap = required_categories & present_categories

            if len(overlap) >= len(required_categories) * 0.67:  # At least 67% match
                pattern_result = {
                    "pattern_name": pattern_name,
                    "description": pattern_info["description"],
                    "risk_level": pattern_info["risk_level"],
                    "matching_categories": list(overlap),
                    "completeness": len(overlap) / len(required_categories)
                }
                sensitive_patterns.append(pattern_result)

        return sensitive_patterns

    def _analyze_co_occurrences(self, phi_entities: List[Any], content: str) -> Dict[str, List[str]]:
        """Analyze co-occurrence patterns of PHI categories."""
        co_occurrences = defaultdict(list)

        if not phi_entities:
            return dict(co_occurrences)

        # Group entities by their position in the document
        entity_positions = []
        for entity in phi_entities:
            position = getattr(entity, 'start_position', 0)
            category = getattr(entity, 'category', 'unknown')
            entity_positions.append((position, category))

        # Sort by position
        entity_positions.sort()

        # Look for entities within proximity (500 characters)
        proximity_threshold = 500

        for i, (pos1, cat1) in enumerate(entity_positions):
            nearby_categories = []

            for j, (pos2, cat2) in enumerate(entity_positions):
                if i != j and abs(pos1 - pos2) <= proximity_threshold:
                    if cat2 not in nearby_categories and cat2 != cat1:
                        nearby_categories.append(cat2)

            if nearby_categories:
                co_occurrences[cat1].extend(nearby_categories)

        # Remove duplicates and sort
        for category in co_occurrences:
            co_occurrences[category] = sorted(list(set(co_occurrences[category])))

        return dict(co_occurrences)

    def analyze_phi_trends(self, analysis_results: List[PHIAnalysisResult],
                          time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze PHI trends across multiple documents."""
        if not analysis_results:
            return {}

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_results = [r for r in analysis_results if r.analysis_timestamp >= cutoff_time]

        if not recent_results:
            return {}

        # Calculate trend metrics
        phi_densities = [r.phi_density for r in recent_results]
        risk_scores = [r.privacy_risk_score for r in recent_results]
        entity_counts = [r.total_phi_entities for r in recent_results]

        # Category frequency analysis
        category_frequencies = defaultdict(int)
        for result in recent_results:
            for category, count in result.phi_by_category.items():
                category_frequencies[category] += count

        # Complexity distribution
        complexity_distribution = Counter(r.redaction_complexity for r in recent_results)

        # High-risk document identification
        high_risk_threshold = 0.7
        high_risk_docs = [r for r in recent_results if r.privacy_risk_score >= high_risk_threshold]

        return {
            "analysis_period_hours": time_window_hours,
            "total_documents": len(recent_results),
            "average_phi_density": statistics.mean(phi_densities),
            "average_risk_score": statistics.mean(risk_scores),
            "average_entity_count": statistics.mean(entity_counts),
            "category_frequencies": dict(category_frequencies),
            "complexity_distribution": dict(complexity_distribution),
            "high_risk_documents": len(high_risk_docs),
            "high_risk_percentage": len(high_risk_docs) / len(recent_results) * 100,
            "most_common_categories": sorted(category_frequencies.items(),
                                           key=lambda x: x[1], reverse=True)[:5],
            "trend_summary": {
                "max_risk_score": max(risk_scores),
                "min_risk_score": min(risk_scores),
                "max_phi_density": max(phi_densities),
                "documents_with_sensitive_patterns": sum(1 for r in recent_results if r.sensitive_patterns)
            }
        }

    def generate_redaction_recommendations(self, analysis_result: PHIAnalysisResult) -> Dict[str, Any]:
        """Generate specific redaction recommendations based on PHI analysis."""
        recommendations = {
            "priority_categories": [],
            "redaction_strategy": "",
            "estimated_effort": "",
            "risk_mitigation_steps": [],
            "automation_feasibility": ""
        }

        # Priority categories (highest risk first)
        category_priorities = []
        for category, count in analysis_result.phi_by_category.items():
            risk_level = self.category_risk_levels.get(category, 0.5)
            category_priorities.append((category, risk_level, count))

        category_priorities.sort(key=lambda x: x[1], reverse=True)
        recommendations["priority_categories"] = [
            {"category": cat, "risk_level": risk, "count": count}
            for cat, risk, count in category_priorities[:5]
        ]

        # Redaction strategy based on complexity
        complexity = analysis_result.redaction_complexity
        if complexity == "simple":
            recommendations["redaction_strategy"] = "Automated redaction with minimal manual review"
        elif complexity == "moderate":
            recommendations["redaction_strategy"] = "Automated redaction with targeted manual review"
        elif complexity == "complex":
            recommendations["redaction_strategy"] = "Manual redaction with automated assistance"
        else:
            recommendations["redaction_strategy"] = "Full manual redaction with expert review"

        # Effort estimation
        effort_map = {
            "simple": "Low (< 1 hour)",
            "moderate": "Medium (1-3 hours)",
            "complex": "High (3-8 hours)",
            "very_complex": "Very High (> 8 hours)"
        }
        recommendations["estimated_effort"] = effort_map[complexity]

        # Risk mitigation steps
        if analysis_result.privacy_risk_score >= 0.8:
            recommendations["risk_mitigation_steps"].append("Immediate redaction required")
            recommendations["risk_mitigation_steps"].append("Restrict access during redaction")

        if analysis_result.sensitive_patterns:
            recommendations["risk_mitigation_steps"].append("Review sensitive pattern clusters carefully")

        if analysis_result.co_occurrence_analysis:
            recommendations["risk_mitigation_steps"].append("Pay special attention to co-located PHI")

        # Automation feasibility
        if analysis_result.privacy_risk_score < 0.5 and complexity in ["simple", "moderate"]:
            recommendations["automation_feasibility"] = "High - suitable for automated redaction"
        elif complexity == "complex":
            recommendations["automation_feasibility"] = "Medium - partial automation possible"
        else:
            recommendations["automation_feasibility"] = "Low - requires manual intervention"

        return recommendations
