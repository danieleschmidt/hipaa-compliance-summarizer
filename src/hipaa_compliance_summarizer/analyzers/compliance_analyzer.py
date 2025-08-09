"""HIPAA compliance analysis and scoring."""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..models.phi_entity import PHICategory
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


@dataclass
class ComplianceAnalysisResult:
    """Result of HIPAA compliance analysis."""

    document_id: str
    overall_compliance_score: float
    category_scores: Dict[str, float]
    violations: List[Dict[str, Any]]
    compliance_status: str
    safe_harbor_compliance: bool
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "overall_compliance_score": self.overall_compliance_score,
            "category_scores": self.category_scores,
            "violations": self.violations,
            "compliance_status": self.compliance_status,
            "safe_harbor_compliance": self.safe_harbor_compliance,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


class ComplianceAnalyzer:
    """Analyzes HIPAA compliance and generates compliance scores."""

    def __init__(self):
        """Initialize compliance analyzer."""
        # HIPAA Safe Harbor compliance requirements
        self.safe_harbor_categories = {
            PHICategory.NAMES.value: {"required_removal": True, "weight": 0.15},
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value: {"required_removal": True, "weight": 0.10},
            PHICategory.DATES.value: {"required_removal": True, "weight": 0.08},
            PHICategory.TELEPHONE_NUMBERS.value: {"required_removal": True, "weight": 0.08},
            PHICategory.FAX_NUMBERS.value: {"required_removal": True, "weight": 0.05},
            PHICategory.EMAIL_ADDRESSES.value: {"required_removal": True, "weight": 0.08},
            PHICategory.SOCIAL_SECURITY_NUMBERS.value: {"required_removal": True, "weight": 0.15},
            PHICategory.MEDICAL_RECORD_NUMBERS.value: {"required_removal": True, "weight": 0.12},
            PHICategory.HEALTH_PLAN_NUMBERS.value: {"required_removal": True, "weight": 0.10},
            PHICategory.ACCOUNT_NUMBERS.value: {"required_removal": True, "weight": 0.09}
        }

        # Violation severity levels
        self.violation_severity = {
            "critical": {"score_impact": -50, "immediate_action": True},
            "high": {"score_impact": -30, "immediate_action": False},
            "medium": {"score_impact": -15, "immediate_action": False},
            "low": {"score_impact": -5, "immediate_action": False}
        }

        # Compliance thresholds
        self.compliance_thresholds = {
            "compliant": 95.0,
            "marginally_compliant": 85.0,
            "non_compliant": 0.0
        }

        # Risk factors for different PHI categories
        self.category_risk_factors = {
            PHICategory.SOCIAL_SECURITY_NUMBERS.value: 1.0,
            PHICategory.BIOMETRIC_IDENTIFIERS.value: 1.0,
            PHICategory.FULL_FACE_PHOTOS.value: 0.9,
            PHICategory.NAMES.value: 0.8,
            PHICategory.MEDICAL_RECORD_NUMBERS.value: 0.8,
            PHICategory.ACCOUNT_NUMBERS.value: 0.7,
            PHICategory.EMAIL_ADDRESSES.value: 0.6,
            PHICategory.TELEPHONE_NUMBERS.value: 0.6,
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value: 0.5,
            PHICategory.DATES.value: 0.4
        }

    @trace_operation("compliance_analysis")
    def analyze_compliance(self, phi_entities: List[Any], document_content: str,
                          document_id: str = None, redacted_content: str = None) -> ComplianceAnalysisResult:
        """Perform comprehensive HIPAA compliance analysis.
        
        Args:
            phi_entities: List of detected PHI entities
            document_content: Original document content
            document_id: Optional document identifier
            redacted_content: Optional redacted version of the document
            
        Returns:
            Compliance analysis result
        """
        logger.info(f"Starting compliance analysis for document: {document_id}")

        # Analyze PHI by category
        phi_by_category = self._categorize_phi_entities(phi_entities)

        # Calculate category-specific compliance scores
        category_scores = self._calculate_category_scores(phi_by_category, redacted_content)

        # Identify violations
        violations = self._identify_violations(phi_by_category, phi_entities)

        # Calculate overall compliance score
        overall_score = self._calculate_overall_score(category_scores, violations)

        # Determine compliance status
        compliance_status = self._determine_compliance_status(overall_score)

        # Check Safe Harbor compliance
        safe_harbor_compliant = self._check_safe_harbor_compliance(phi_by_category, redacted_content)

        # Perform risk assessment
        risk_assessment = self._perform_risk_assessment(phi_entities, phi_by_category, overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(violations, category_scores, overall_score)

        result = ComplianceAnalysisResult(
            document_id=document_id or "unknown",
            overall_compliance_score=overall_score,
            category_scores=category_scores,
            violations=violations,
            compliance_status=compliance_status,
            safe_harbor_compliance=safe_harbor_compliant,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )

        logger.info(f"Compliance analysis completed: {compliance_status}, score: {overall_score:.1f}")
        return result

    def _categorize_phi_entities(self, phi_entities: List[Any]) -> Dict[str, List[Any]]:
        """Categorize PHI entities by type."""
        categorized = defaultdict(list)

        for entity in phi_entities:
            category = getattr(entity, 'category', 'unknown')
            categorized[category].append(entity)

        return dict(categorized)

    def _calculate_category_scores(self, phi_by_category: Dict[str, List[Any]],
                                 redacted_content: str = None) -> Dict[str, float]:
        """Calculate compliance scores for each PHI category."""
        category_scores = {}

        for category, entities in phi_by_category.items():
            if category in self.safe_harbor_categories:
                # Check if entities in this category are properly handled
                if redacted_content is not None:
                    # Verify entities are redacted in the redacted version
                    redaction_score = self._verify_category_redaction(entities, redacted_content)
                else:
                    # No redacted content provided - assume not redacted
                    redaction_score = 0.0 if entities else 100.0

                category_scores[category] = redaction_score
            else:
                # For categories not in Safe Harbor, assess based on presence
                category_scores[category] = 90.0 if len(entities) <= 2 else max(50.0, 90.0 - len(entities) * 5)

        return category_scores

    def _verify_category_redaction(self, entities: List[Any], redacted_content: str) -> float:
        """Verify that entities are properly redacted."""
        if not entities:
            return 100.0

        redacted_count = 0

        for entity in entities:
            entity_text = getattr(entity, 'text', '')
            # Check if entity text appears in redacted content
            if entity_text not in redacted_content:
                redacted_count += 1

        redaction_rate = redacted_count / len(entities)
        return redaction_rate * 100.0

    def _identify_violations(self, phi_by_category: Dict[str, List[Any]],
                           phi_entities: List[Any]) -> List[Dict[str, Any]]:
        """Identify HIPAA compliance violations."""
        violations = []

        # Check for Safe Harbor violations
        for category, requirements in self.safe_harbor_categories.items():
            if requirements["required_removal"] and category in phi_by_category:
                entities = phi_by_category[category]
                if entities:
                    violations.append({
                        "type": "safe_harbor_violation",
                        "category": category,
                        "severity": self._determine_violation_severity(category),
                        "description": f"Safe Harbor violation: {category} entities present and not redacted",
                        "entity_count": len(entities),
                        "examples": [getattr(e, 'text', '') for e in entities[:3]]  # First 3 examples
                    })

        # Check for high-density PHI violations
        total_phi_count = len(phi_entities)
        if total_phi_count > 50:  # Threshold for high PHI density
            violations.append({
                "type": "high_phi_density",
                "category": "general",
                "severity": "medium",
                "description": f"High PHI density detected: {total_phi_count} entities",
                "entity_count": total_phi_count,
                "examples": []
            })

        # Check for sensitive category combinations
        sensitive_combinations = self._check_sensitive_combinations(phi_by_category)
        violations.extend(sensitive_combinations)

        return violations

    def _determine_violation_severity(self, category: str) -> str:
        """Determine violation severity based on PHI category."""
        risk_factor = self.category_risk_factors.get(category, 0.5)

        if risk_factor >= 0.9:
            return "critical"
        elif risk_factor >= 0.7:
            return "high"
        elif risk_factor >= 0.5:
            return "medium"
        else:
            return "low"

    def _check_sensitive_combinations(self, phi_by_category: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Check for sensitive PHI category combinations."""
        violations = []
        present_categories = set(phi_by_category.keys())

        # Identity reconstruction risk
        identity_categories = {
            PHICategory.NAMES.value,
            PHICategory.DATES.value,
            PHICategory.GEOGRAPHIC_SUBDIVISIONS.value
        }

        if len(identity_categories & present_categories) >= 2:
            violations.append({
                "type": "identity_reconstruction_risk",
                "category": "combination",
                "severity": "high",
                "description": "Combination of identifying information enables identity reconstruction",
                "entity_count": sum(len(phi_by_category[cat]) for cat in identity_categories & present_categories),
                "examples": []
            })

        # Financial risk combination
        financial_categories = {
            PHICategory.SOCIAL_SECURITY_NUMBERS.value,
            PHICategory.ACCOUNT_NUMBERS.value,
            PHICategory.NAMES.value
        }

        if len(financial_categories & present_categories) >= 2:
            violations.append({
                "type": "financial_risk_combination",
                "category": "combination",
                "severity": "critical",
                "description": "Combination of financial and identifying information",
                "entity_count": sum(len(phi_by_category[cat]) for cat in financial_categories & present_categories),
                "examples": []
            })

        return violations

    def _calculate_overall_score(self, category_scores: Dict[str, float],
                               violations: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score."""
        if not category_scores:
            return 0.0

        # Start with weighted average of category scores
        total_weight = 0
        weighted_score = 0

        for category, score in category_scores.items():
            weight = self.safe_harbor_categories.get(category, {}).get("weight", 0.05)
            weighted_score += score * weight
            total_weight += weight

        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = statistics.mean(category_scores.values())

        # Apply violation penalties
        for violation in violations:
            severity = violation["severity"]
            penalty = self.violation_severity.get(severity, {}).get("score_impact", -5)
            base_score += penalty

        return max(0.0, min(100.0, base_score))

    def _determine_compliance_status(self, overall_score: float) -> str:
        """Determine compliance status based on score."""
        if overall_score >= self.compliance_thresholds["compliant"]:
            return "compliant"
        elif overall_score >= self.compliance_thresholds["marginally_compliant"]:
            return "marginally_compliant"
        else:
            return "non_compliant"

    def _check_safe_harbor_compliance(self, phi_by_category: Dict[str, List[Any]],
                                    redacted_content: str = None) -> bool:
        """Check if document meets HIPAA Safe Harbor requirements."""
        if redacted_content is None:
            # Cannot verify redaction without redacted content
            return False

        for category in self.safe_harbor_categories:
            if category in phi_by_category and phi_by_category[category]:
                # Check if all entities in this category are redacted
                redaction_score = self._verify_category_redaction(
                    phi_by_category[category], redacted_content
                )
                if redaction_score < 100.0:  # Not fully redacted
                    return False

        return True

    def _perform_risk_assessment(self, phi_entities: List[Any],
                               phi_by_category: Dict[str, List[Any]],
                               compliance_score: float) -> Dict[str, Any]:
        """Perform comprehensive risk assessment."""
        risk_factors = []

        # Compliance score risk
        if compliance_score < 70:
            risk_factors.append("Low compliance score")

        # High-risk category presence
        high_risk_categories = []
        for category, entities in phi_by_category.items():
            if entities and self.category_risk_factors.get(category, 0) >= 0.8:
                high_risk_categories.append(category)

        if high_risk_categories:
            risk_factors.append(f"High-risk PHI categories present: {', '.join(high_risk_categories)}")

        # PHI density risk
        total_entities = len(phi_entities)
        if total_entities > 30:
            risk_factors.append(f"High PHI density: {total_entities} entities")

        # Calculate overall risk level
        risk_score = 0.0

        # Compliance score component (inverted)
        risk_score += (100 - compliance_score) / 100 * 0.4

        # High-risk category component
        high_risk_weight = sum(self.category_risk_factors.get(cat, 0) for cat in high_risk_categories)
        risk_score += min(high_risk_weight / 3.0, 1.0) * 0.3

        # PHI density component
        density_factor = min(total_entities / 50.0, 1.0)
        risk_score += density_factor * 0.3

        risk_level = "low"
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"

        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "high_risk_categories": high_risk_categories,
            "total_phi_entities": total_entities,
            "requires_immediate_attention": risk_level in ["critical", "high"]
        }

    def _generate_recommendations(self, violations: List[Dict[str, Any]],
                                category_scores: Dict[str, float],
                                overall_score: float) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []

        # Score-based recommendations
        if overall_score < 70:
            recommendations.append("Immediate compliance review required - score below acceptable threshold")

        if overall_score < 95:
            recommendations.append("Implement comprehensive PHI redaction process")

        # Violation-based recommendations
        critical_violations = [v for v in violations if v["severity"] == "critical"]
        if critical_violations:
            recommendations.append("Address critical violations immediately - potential HIPAA breach risk")

        high_violations = [v for v in violations if v["severity"] == "high"]
        if high_violations:
            recommendations.append("Review and remediate high-severity violations within 24 hours")

        # Category-specific recommendations
        low_scoring_categories = [cat for cat, score in category_scores.items() if score < 80]
        if low_scoring_categories:
            recommendations.append(f"Focus redaction efforts on categories: {', '.join(low_scoring_categories)}")

        # Safe Harbor recommendations
        safe_harbor_categories_present = [cat for cat in self.safe_harbor_categories
                                        if cat in category_scores and category_scores[cat] < 100]
        if safe_harbor_categories_present:
            recommendations.append("Ensure complete redaction of Safe Harbor categories for de-identification compliance")

        # General recommendations
        if not recommendations:
            recommendations.append("Maintain current compliance practices and conduct regular reviews")

        recommendations.append("Document all redaction decisions and maintain audit trail")

        return recommendations

    def generate_compliance_report(self, analysis_results: List[ComplianceAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive compliance report from multiple analyses."""
        if not analysis_results:
            return {}

        # Overall statistics
        scores = [r.overall_compliance_score for r in analysis_results]
        compliance_statuses = [r.compliance_status for r in analysis_results]

        # Violation analysis
        all_violations = []
        for result in analysis_results:
            all_violations.extend(result.violations)

        violation_by_type = defaultdict(int)
        violation_by_severity = defaultdict(int)

        for violation in all_violations:
            violation_by_type[violation["type"]] += 1
            violation_by_severity[violation["severity"]] += 1

        # Risk assessment summary
        risk_levels = [r.risk_assessment["risk_level"] for r in analysis_results]
        high_risk_docs = sum(1 for r in analysis_results
                           if r.risk_assessment["requires_immediate_attention"])

        return {
            "report_generated": datetime.utcnow().isoformat(),
            "documents_analyzed": len(analysis_results),
            "overall_statistics": {
                "average_compliance_score": statistics.mean(scores),
                "median_compliance_score": statistics.median(scores),
                "min_compliance_score": min(scores),
                "max_compliance_score": max(scores),
                "compliant_documents": compliance_statuses.count("compliant"),
                "marginally_compliant_documents": compliance_statuses.count("marginally_compliant"),
                "non_compliant_documents": compliance_statuses.count("non_compliant")
            },
            "violation_analysis": {
                "total_violations": len(all_violations),
                "violations_by_type": dict(violation_by_type),
                "violations_by_severity": dict(violation_by_severity),
                "documents_with_violations": sum(1 for r in analysis_results if r.violations)
            },
            "risk_assessment": {
                "high_risk_documents": high_risk_docs,
                "risk_distribution": dict(Counter(risk_levels)),
                "immediate_attention_required": high_risk_docs
            },
            "safe_harbor_compliance": {
                "compliant_documents": sum(1 for r in analysis_results if r.safe_harbor_compliance),
                "compliance_rate": sum(1 for r in analysis_results if r.safe_harbor_compliance) / len(analysis_results) * 100
            }
        }
