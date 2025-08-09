"""Risk analysis for HIPAA compliance and data protection."""

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List

from ..models.phi_entity import PHICategory
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


@dataclass
class RiskAnalysisResult:
    """Result of risk analysis."""

    document_id: str
    overall_risk_score: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    breach_probability: float
    impact_assessment: Dict[str, Any]
    mitigation_strategies: List[str]
    monitoring_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "breach_probability": self.breach_probability,
            "impact_assessment": self.impact_assessment,
            "mitigation_strategies": self.mitigation_strategies,
            "monitoring_recommendations": self.monitoring_recommendations,
            "metadata": self.metadata,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


class RiskAnalyzer:
    """Analyzes privacy and security risks for HIPAA compliance."""

    def __init__(self):
        """Initialize risk analyzer."""
        # Risk weights for different PHI categories
        self.category_risk_weights = {
            PHICategory.SOCIAL_SECURITY_NUMBERS.value: 1.0,
            PHICategory.BIOMETRIC_IDENTIFIERS.value: 1.0,
            PHICategory.FULL_FACE_PHOTOS.value: 0.95,
            PHICategory.MEDICAL_RECORD_NUMBERS.value: 0.9,
            PHICategory.HEALTH_PLAN_NUMBERS.value: 0.85,
            PHICategory.ACCOUNT_NUMBERS.value: 0.8,
            PHICategory.NAMES.value: 0.75,
            PHICategory.EMAIL_ADDRESSES.value: 0.7,
            PHICategory.TELEPHONE_NUMBERS.value: 0.65,
            PHICategory.CERTIFICATE_NUMBERS.value: 0.6,
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value: 0.5,
            PHICategory.DATES.value: 0.4,
            PHICategory.VEHICLE_IDENTIFIERS.value: 0.35,
            PHICategory.DEVICE_IDENTIFIERS.value: 0.3,
            PHICategory.WEB_URLS.value: 0.25,
            PHICategory.IP_ADDRESSES.value: 0.2,
            PHICategory.FAX_NUMBERS.value: 0.15,
            PHICategory.OTHER_IDENTIFYING_NUMBERS.value: 0.5
        }

        # Risk level thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9
        }

        # Breach impact factors
        self.impact_factors = {
            "regulatory_penalties": {
                "low": {"min": 1000, "max": 10000, "description": "Minor regulatory findings"},
                "medium": {"min": 10000, "max": 50000, "description": "Moderate compliance violations"},
                "high": {"min": 50000, "max": 250000, "description": "Significant HIPAA violations"},
                "critical": {"min": 250000, "max": 1000000, "description": "Major data breach with willful negligence"}
            },
            "reputational_damage": {
                "low": "Minor impact on organization reputation",
                "medium": "Moderate negative publicity and patient trust issues",
                "high": "Significant reputational damage affecting patient acquisition",
                "critical": "Severe long-term reputational harm and potential business closure"
            },
            "operational_disruption": {
                "low": "Minimal operational impact",
                "medium": "Temporary workflow disruptions",
                "high": "Significant operational delays and resource allocation",
                "critical": "Major operational shutdown and extensive remediation efforts"
            }
        }

        # High-risk combination patterns
        self.high_risk_patterns = [
            {
                "name": "identity_theft_enabler",
                "categories": [PHICategory.NAMES.value, PHICategory.SOCIAL_SECURITY_NUMBERS.value, PHICategory.DATES.value],
                "risk_multiplier": 1.5,
                "description": "Combination enables identity theft"
            },
            {
                "name": "financial_fraud_risk",
                "categories": [PHICategory.NAMES.value, PHICategory.ACCOUNT_NUMBERS.value, PHICategory.DATES.value],
                "risk_multiplier": 1.4,
                "description": "Combination enables financial fraud"
            },
            {
                "name": "medical_identity_theft",
                "categories": [PHICategory.NAMES.value, PHICategory.MEDICAL_RECORD_NUMBERS.value, PHICategory.HEALTH_PLAN_NUMBERS.value],
                "risk_multiplier": 1.3,
                "description": "Combination enables medical identity theft"
            },
            {
                "name": "comprehensive_profile",
                "categories": [PHICategory.NAMES.value, PHICategory.GEOGRAPHIC_SUBDIVISIONS.value,
                             PHICategory.TELEPHONE_NUMBERS.value, PHICategory.EMAIL_ADDRESSES.value],
                "risk_multiplier": 1.2,
                "description": "Complete personal profile exposure"
            }
        ]

    @trace_operation("risk_analysis")
    def analyze_risk(self, phi_entities: List[Any], document_content: str,
                    document_id: str = None, context: Dict[str, Any] = None) -> RiskAnalysisResult:
        """Perform comprehensive risk analysis.
        
        Args:
            phi_entities: List of detected PHI entities
            document_content: Original document content
            document_id: Optional document identifier
            context: Optional context information (storage location, access controls, etc.)
            
        Returns:
            Risk analysis result
        """
        logger.info(f"Starting risk analysis for document: {document_id}")

        # Categorize PHI entities
        phi_by_category = self._categorize_phi_entities(phi_entities)

        # Calculate base risk score
        base_risk_score = self._calculate_base_risk_score(phi_by_category)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(phi_entities, phi_by_category, document_content)

        # Apply risk multipliers
        adjusted_risk_score = self._apply_risk_multipliers(base_risk_score, risk_factors)

        # Determine risk level
        risk_level = self._determine_risk_level(adjusted_risk_score)

        # Calculate breach probability
        breach_probability = self._calculate_breach_probability(adjusted_risk_score, context)

        # Assess potential impact
        impact_assessment = self._assess_impact(risk_level, phi_by_category)

        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors, risk_level)

        # Generate monitoring recommendations
        monitoring_recommendations = self._generate_monitoring_recommendations(risk_level, risk_factors)

        result = RiskAnalysisResult(
            document_id=document_id or "unknown",
            overall_risk_score=adjusted_risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            breach_probability=breach_probability,
            impact_assessment=impact_assessment,
            mitigation_strategies=mitigation_strategies,
            monitoring_recommendations=monitoring_recommendations,
            metadata=context or {}
        )

        logger.info(f"Risk analysis completed: {risk_level} risk, score: {adjusted_risk_score:.2f}")
        return result

    def _categorize_phi_entities(self, phi_entities: List[Any]) -> Dict[str, List[Any]]:
        """Categorize PHI entities by type."""
        categorized = defaultdict(list)

        for entity in phi_entities:
            category = getattr(entity, 'category', 'unknown')
            categorized[category].append(entity)

        return dict(categorized)

    def _calculate_base_risk_score(self, phi_by_category: Dict[str, List[Any]]) -> float:
        """Calculate base risk score from PHI categories and counts."""
        if not phi_by_category:
            return 0.0

        total_risk = 0.0
        total_weight = 0.0

        for category, entities in phi_by_category.items():
            category_weight = self.category_risk_weights.get(category, 0.3)
            entity_count = len(entities)

            # Risk increases with count but with diminishing returns
            count_factor = min(1.0, 0.3 + 0.7 * (1 - 1 / (1 + entity_count * 0.2)))

            category_risk = category_weight * count_factor
            total_risk += category_risk
            total_weight += category_weight

        # Normalize by maximum possible risk
        max_possible_risk = sum(self.category_risk_weights.values())
        normalized_risk = min(1.0, total_risk / max_possible_risk * 2)  # Scale up for sensitivity

        return normalized_risk

    def _identify_risk_factors(self, phi_entities: List[Any],
                             phi_by_category: Dict[str, List[Any]],
                             document_content: str) -> List[Dict[str, Any]]:
        """Identify specific risk factors."""
        risk_factors = []

        # High PHI density risk
        total_entities = len(phi_entities)
        word_count = len(document_content.split())
        phi_density = total_entities / word_count if word_count > 0 else 0

        if phi_density > 0.05:  # More than 5% of words are PHI
            risk_factors.append({
                "type": "high_phi_density",
                "severity": "high" if phi_density > 0.1 else "medium",
                "description": f"High PHI density: {phi_density:.1%} of content",
                "risk_multiplier": 1.2 if phi_density > 0.1 else 1.1,
                "metadata": {"phi_density": phi_density, "total_entities": total_entities}
            })

        # High-risk category presence
        critical_categories = [cat for cat, entities in phi_by_category.items()
                             if entities and self.category_risk_weights.get(cat, 0) >= 0.9]

        if critical_categories:
            risk_factors.append({
                "type": "critical_phi_categories",
                "severity": "critical",
                "description": f"Critical PHI categories present: {', '.join(critical_categories)}",
                "risk_multiplier": 1.3,
                "metadata": {"categories": critical_categories}
            })

        # High-risk pattern detection
        present_categories = set(phi_by_category.keys())
        for pattern in self.high_risk_patterns:
            pattern_categories = set(pattern["categories"])
            overlap = pattern_categories & present_categories

            if len(overlap) >= len(pattern_categories) * 0.8:  # At least 80% of pattern present
                risk_factors.append({
                    "type": "high_risk_pattern",
                    "severity": "high",
                    "description": pattern["description"],
                    "risk_multiplier": pattern["risk_multiplier"],
                    "metadata": {
                        "pattern_name": pattern["name"],
                        "matching_categories": list(overlap),
                        "completeness": len(overlap) / len(pattern_categories)
                    }
                })

        # Document structure risks
        if self._has_structured_format(document_content):
            risk_factors.append({
                "type": "structured_data_format",
                "severity": "medium",
                "description": "Document contains structured data that may be harder to redact",
                "risk_multiplier": 1.1,
                "metadata": {"structured_format": True}
            })

        # Multiple identifier types risk
        unique_categories = len(phi_by_category)
        if unique_categories >= 5:
            risk_factors.append({
                "type": "multiple_identifier_types",
                "severity": "medium",
                "description": f"Multiple types of identifiers present ({unique_categories} categories)",
                "risk_multiplier": 1.1,
                "metadata": {"category_count": unique_categories}
            })

        return risk_factors

    def _has_structured_format(self, content: str) -> bool:
        """Check if document has structured data format."""
        structured_indicators = [
            r'\|.*\|.*\|',  # Table format
            r'^\s*\w+:\s*\w+',  # Key-value pairs
            r'<[^>]+>',  # XML/HTML tags
            r'\{[^}]*\}',  # JSON-like structures
            r'^\s*\d+\.\s+',  # Numbered lists
        ]

        import re
        for pattern in structured_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return True

        return False

    def _apply_risk_multipliers(self, base_risk: float, risk_factors: List[Dict[str, Any]]) -> float:
        """Apply risk multipliers from identified risk factors."""
        adjusted_risk = base_risk

        for factor in risk_factors:
            multiplier = factor.get("risk_multiplier", 1.0)
            adjusted_risk *= multiplier

        return min(1.0, adjusted_risk)  # Cap at 1.0

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score."""
        if risk_score >= self.risk_thresholds["critical"]:
            return "critical"
        elif risk_score >= self.risk_thresholds["high"]:
            return "high"
        elif risk_score >= self.risk_thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def _calculate_breach_probability(self, risk_score: float, context: Dict[str, Any] = None) -> float:
        """Calculate probability of data breach."""
        base_probability = risk_score * 0.3  # Base 30% max probability from content alone

        if context:
            # Adjust based on security controls
            security_score = context.get("security_score", 0.5)
            access_controls = context.get("access_controls", 0.5)
            encryption_level = context.get("encryption_level", 0.5)

            # Security controls reduce breach probability
            security_factor = (security_score + access_controls + encryption_level) / 3
            adjusted_probability = base_probability * (1 - security_factor * 0.7)
        else:
            # No context - assume moderate security
            adjusted_probability = base_probability * 0.7

        return min(1.0, max(0.0, adjusted_probability))

    def _assess_impact(self, risk_level: str, phi_by_category: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Assess potential impact of a breach."""
        impact = {}

        # Regulatory penalties
        penalty_info = self.impact_factors["regulatory_penalties"][risk_level]
        impact["regulatory_penalties"] = {
            "estimated_range": f"${penalty_info['min']:,} - ${penalty_info['max']:,}",
            "description": penalty_info["description"]
        }

        # Reputational damage
        impact["reputational_damage"] = self.impact_factors["reputational_damage"][risk_level]

        # Operational disruption
        impact["operational_disruption"] = self.impact_factors["operational_disruption"][risk_level]

        # Affected individuals estimate
        total_entities = sum(len(entities) for entities in phi_by_category.values())

        # Rough estimate based on PHI density and types
        if PHICategory.NAMES.value in phi_by_category:
            # Assume each name represents one individual
            estimated_individuals = len(phi_by_category[PHICategory.NAMES.value])
        else:
            # Estimate based on total entities
            estimated_individuals = max(1, total_entities // 3)

        impact["affected_individuals"] = {
            "estimated_count": estimated_individuals,
            "notification_requirements": estimated_individuals >= 500  # HIPAA breach notification threshold
        }

        return impact

    def _generate_mitigation_strategies(self, risk_factors: List[Dict[str, Any]],
                                      risk_level: str) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []

        # Risk level based strategies
        if risk_level == "critical":
            strategies.extend([
                "Implement immediate access restrictions",
                "Conduct emergency security review",
                "Consider temporary system isolation",
                "Engage legal and compliance teams immediately"
            ])
        elif risk_level == "high":
            strategies.extend([
                "Expedite redaction process",
                "Implement enhanced monitoring",
                "Review access controls and permissions",
                "Prepare incident response procedures"
            ])
        elif risk_level == "medium":
            strategies.extend([
                "Schedule comprehensive redaction review",
                "Implement additional access logging",
                "Conduct staff training on PHI handling"
            ])
        else:  # low risk
            strategies.extend([
                "Maintain standard redaction practices",
                "Continue regular compliance monitoring"
            ])

        # Risk factor specific strategies
        factor_types = [f["type"] for f in risk_factors]

        if "high_phi_density" in factor_types:
            strategies.append("Prioritize automated redaction tools for high-density PHI")

        if "critical_phi_categories" in factor_types:
            strategies.append("Implement specialized handling for critical PHI categories")

        if "high_risk_pattern" in factor_types:
            strategies.append("Focus on pattern-based redaction to address identifier clusters")

        if "structured_data_format" in factor_types:
            strategies.append("Use structured data redaction tools and techniques")

        # General strategies
        strategies.extend([
            "Maintain detailed audit logs of all access and modifications",
            "Implement regular risk assessment reviews",
            "Ensure staff training on identified risk patterns"
        ])

        return list(set(strategies))  # Remove duplicates

    def _generate_monitoring_recommendations(self, risk_level: str,
                                           risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []

        # Risk level based monitoring
        if risk_level in ["critical", "high"]:
            recommendations.extend([
                "Implement real-time access monitoring",
                "Set up automated alerts for unusual access patterns",
                "Conduct daily security reviews",
                "Monitor for unauthorized data access attempts"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Implement weekly access reviews",
                "Monitor for bulk data access",
                "Set up monthly security audits"
            ])
        else:  # low risk
            recommendations.extend([
                "Maintain standard access logging",
                "Conduct quarterly reviews"
            ])

        # Specific monitoring based on risk factors
        factor_types = [f["type"] for f in risk_factors]

        if "critical_phi_categories" in factor_types:
            recommendations.append("Implement specialized monitoring for critical PHI access")

        if "high_phi_density" in factor_types:
            recommendations.append("Monitor for bulk data extraction attempts")

        # General recommendations
        recommendations.extend([
            "Track redaction completion status",
            "Monitor compliance score trends",
            "Alert on new PHI detection in processed documents"
        ])

        return list(set(recommendations))  # Remove duplicates

    def generate_risk_trends_analysis(self, risk_results: List[RiskAnalysisResult],
                                    time_window_hours: int = 168) -> Dict[str, Any]:  # Default 1 week
        """Analyze risk trends across multiple documents."""
        if not risk_results:
            return {}

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_results = [r for r in risk_results if r.analysis_timestamp >= cutoff_time]

        if not recent_results:
            return {}

        # Risk score trends
        risk_scores = [r.overall_risk_score for r in recent_results]
        risk_levels = [r.risk_level for r in recent_results]

        # Risk factor analysis
        all_risk_factors = []
        for result in recent_results:
            all_risk_factors.extend([f["type"] for f in result.risk_factors])

        factor_frequency = Counter(all_risk_factors)

        # Breach probability trends
        breach_probabilities = [r.breach_probability for r in recent_results]

        return {
            "analysis_period_hours": time_window_hours,
            "documents_analyzed": len(recent_results),
            "risk_score_trends": {
                "average": statistics.mean(risk_scores),
                "median": statistics.median(risk_scores),
                "max": max(risk_scores),
                "min": min(risk_scores),
                "trend": "increasing" if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else "stable"
            },
            "risk_level_distribution": dict(Counter(risk_levels)),
            "common_risk_factors": dict(factor_frequency.most_common(10)),
            "breach_probability": {
                "average": statistics.mean(breach_probabilities),
                "high_risk_documents": sum(1 for p in breach_probabilities if p > 0.3)
            },
            "immediate_attention_required": sum(1 for r in recent_results if r.risk_level in ["critical", "high"]),
            "trends_summary": {
                "deteriorating_risk": sum(1 for r in recent_results if r.overall_risk_score > 0.7),
                "improving_controls": sum(1 for r in recent_results if r.breach_probability < 0.2)
            }
        }
