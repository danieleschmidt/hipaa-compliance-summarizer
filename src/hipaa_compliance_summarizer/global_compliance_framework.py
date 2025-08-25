"""
Global Compliance Framework - Multi-Jurisdiction Healthcare Compliance

This module implements a comprehensive global compliance framework that handles
multiple international healthcare privacy and security regulations including
HIPAA (US), GDPR (EU), PDPA (Singapore), PIPEDA (Canada), and other regional
standards with intelligent compliance mapping and automated validation.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class ComplianceJurisdiction(Enum):
    """Global compliance jurisdictions."""
    UNITED_STATES = "US"
    EUROPEAN_UNION = "EU" 
    UNITED_KINGDOM = "UK"
    CANADA = "CA"
    SINGAPORE = "SG"
    AUSTRALIA = "AU"
    JAPAN = "JP"
    SOUTH_KOREA = "KR"
    BRAZIL = "BR"
    INDIA = "IN"


class ComplianceStandard(Enum):
    """International compliance standards."""
    HIPAA = "hipaa"                    # US - Health Insurance Portability and Accountability Act
    GDPR = "gdpr"                      # EU - General Data Protection Regulation
    PDPA_SG = "pdpa_sg"                # Singapore - Personal Data Protection Act
    PIPEDA = "pipeda"                  # Canada - Personal Information Protection and Electronic Documents Act
    DPA_UK = "dpa_uk"                  # UK - Data Protection Act 2018
    PRIVACY_ACT = "privacy_act_au"      # Australia - Privacy Act 1988
    APPI = "appi"                      # Japan - Act on Protection of Personal Information
    PIPA_KR = "pipa_kr"                # South Korea - Personal Information Protection Act
    LGPD = "lgpd"                      # Brazil - Lei Geral de Proteção de Dados
    DPDPA = "dpdpa"                    # India - Digital Personal Data Protection Act 2023


class DataSensitivityLevel(Enum):
    """Data sensitivity classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class PHICategory:
    """Protected Health Information category definition."""
    name: str
    description: str
    sensitivity_level: DataSensitivityLevel
    applicable_standards: List[ComplianceStandard]
    retention_period_days: int
    encryption_required: bool = True
    audit_required: bool = True
    cross_border_restrictions: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    standard: ComplianceStandard
    jurisdiction: ComplianceJurisdiction
    category: str
    description: str
    requirements: List[str]
    penalties: Dict[str, Any]
    implementation_guidance: List[str]
    validation_criteria: List[str]
    automated_check: bool = True


@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    jurisdiction: ComplianceJurisdiction
    standards_evaluated: List[ComplianceStandard]
    compliance_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str
    assessment_date: datetime
    next_review_date: datetime


class GlobalComplianceFramework:
    """
    Comprehensive global compliance framework for healthcare data processing.
    
    Provides intelligent compliance mapping, automated validation, and 
    multi-jurisdiction compliance orchestration for healthcare systems.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GlobalComplianceFramework")
        
        # Initialize compliance mappings
        self.phi_categories = self._initialize_phi_categories()
        self.compliance_rules = self._initialize_compliance_rules()
        self.jurisdiction_mappings = self._initialize_jurisdiction_mappings()
        
        # Compliance assessment history
        self.assessment_history: List[ComplianceAssessment] = []
    
    def _initialize_phi_categories(self) -> Dict[str, PHICategory]:
        """Initialize PHI category definitions with global compliance mapping."""
        categories = {
            "direct_identifiers": PHICategory(
                name="Direct Identifiers",
                description="Names, addresses, phone numbers, email addresses",
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                applicable_standards=[
                    ComplianceStandard.HIPAA,
                    ComplianceStandard.GDPR,
                    ComplianceStandard.PDPA_SG,
                    ComplianceStandard.PIPEDA
                ],
                retention_period_days=2555,  # 7 years
                cross_border_restrictions={
                    "EU": False,  # Restricted under GDPR
                    "SG": True,   # Allowed with proper safeguards
                    "US": True,
                    "CA": True
                }
            ),
            "medical_identifiers": PHICategory(
                name="Medical Record Numbers",
                description="MRNs, account numbers, certificate/license numbers",
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                applicable_standards=[
                    ComplianceStandard.HIPAA,
                    ComplianceStandard.GDPR,
                    ComplianceStandard.PRIVACY_ACT,
                    ComplianceStandard.APPI
                ],
                retention_period_days=3650,  # 10 years for medical records
                cross_border_restrictions={
                    "EU": False,
                    "JP": False,  # Strict under APPI
                    "AU": True,
                    "US": True
                }
            ),
            "biometric_data": PHICategory(
                name="Biometric Identifiers",
                description="Fingerprints, voiceprints, retinal scans, facial recognition",
                sensitivity_level=DataSensitivityLevel.TOP_SECRET,
                applicable_standards=[
                    ComplianceStandard.GDPR,
                    ComplianceStandard.HIPAA,
                    ComplianceStandard.PIPA_KR,
                    ComplianceStandard.LGPD
                ],
                retention_period_days=1825,  # 5 years
                cross_border_restrictions={
                    "EU": False,   # Special category data under GDPR
                    "KR": False,   # Highly restricted
                    "BR": False,   # Sensitive data under LGPD
                    "US": True
                }
            ),
            "health_data": PHICategory(
                name="Health Information",
                description="Medical diagnoses, treatment plans, lab results",
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                applicable_standards=[
                    ComplianceStandard.HIPAA,
                    ComplianceStandard.GDPR,
                    ComplianceStandard.DPA_UK,
                    ComplianceStandard.DPDPA
                ],
                retention_period_days=2555,  # 7 years
                cross_border_restrictions={
                    "EU": False,   # Special category under GDPR
                    "UK": False,   # Special category under DPA 2018
                    "IN": False,   # Sensitive under DPDPA
                    "US": True,
                    "CA": True
                }
            ),
            "genetic_data": PHICategory(
                name="Genetic Information",
                description="DNA sequences, genetic test results, family medical history",
                sensitivity_level=DataSensitivityLevel.TOP_SECRET,
                applicable_standards=[
                    ComplianceStandard.GDPR,
                    ComplianceStandard.HIPAA,
                    ComplianceStandard.PRIVACY_ACT,
                    ComplianceStandard.LGPD
                ],
                retention_period_days=1825,  # 5 years
                cross_border_restrictions={
                    "EU": False,   # Special category data
                    "AU": False,   # Highly sensitive under Privacy Act
                    "BR": False,   # Sensitive data under LGPD
                    "US": True,
                    "CA": True
                }
            )
        }
        
        return categories
    
    def _initialize_compliance_rules(self) -> Dict[str, List[ComplianceRule]]:
        """Initialize comprehensive compliance rules by jurisdiction."""
        rules = {
            ComplianceJurisdiction.UNITED_STATES.value: [
                ComplianceRule(
                    rule_id="HIPAA-164.502",
                    standard=ComplianceStandard.HIPAA,
                    jurisdiction=ComplianceJurisdiction.UNITED_STATES,
                    category="data_use_disclosure",
                    description="Uses and disclosures of protected health information",
                    requirements=[
                        "Obtain patient authorization for disclosure",
                        "Implement minimum necessary standard",
                        "Maintain audit logs of all disclosures",
                        "Provide accounting of disclosures upon request"
                    ],
                    penalties={
                        "tier_1": {"min": 100, "max": 50000, "description": "Unknowing violation"},
                        "tier_2": {"min": 1000, "max": 50000, "description": "Reasonable cause"},
                        "tier_3": {"min": 10000, "max": 50000, "description": "Willful neglect - corrected"},
                        "tier_4": {"min": 50000, "max": 1500000, "description": "Willful neglect - uncorrected"}
                    },
                    implementation_guidance=[
                        "Implement role-based access controls",
                        "Create disclosure authorization workflows",
                        "Establish audit logging mechanisms",
                        "Train staff on minimum necessary principles"
                    ],
                    validation_criteria=[
                        "Authorization forms are properly executed",
                        "Access logs show minimum necessary access",
                        "Disclosure tracking is complete and accurate",
                        "Staff training records are current"
                    ]
                ),
                ComplianceRule(
                    rule_id="HIPAA-164.312",
                    standard=ComplianceStandard.HIPAA,
                    jurisdiction=ComplianceJurisdiction.UNITED_STATES,
                    category="technical_safeguards",
                    description="Technical safeguards for electronic PHI",
                    requirements=[
                        "Access control with unique user identification",
                        "Automatic logoff and encryption",
                        "Audit controls and integrity protections",
                        "Person or entity authentication"
                    ],
                    penalties={
                        "civil": {"max": 1900000, "description": "Civil penalties per violation category"},
                        "criminal": {"max": 250000, "prison": "10 years", "description": "Criminal penalties"}
                    },
                    implementation_guidance=[
                        "Deploy multi-factor authentication",
                        "Implement AES-256 encryption",
                        "Configure comprehensive audit logging",
                        "Establish regular access reviews"
                    ],
                    validation_criteria=[
                        "MFA is enabled for all users",
                        "Data at rest and in transit is encrypted",
                        "Audit logs capture all PHI access",
                        "Access reviews are conducted quarterly"
                    ]
                )
            ],
            ComplianceJurisdiction.EUROPEAN_UNION.value: [
                ComplianceRule(
                    rule_id="GDPR-Article-6",
                    standard=ComplianceStandard.GDPR,
                    jurisdiction=ComplianceJurisdiction.EUROPEAN_UNION,
                    category="lawfulness_processing",
                    description="Lawfulness of processing personal data",
                    requirements=[
                        "Establish lawful basis for processing",
                        "Obtain explicit consent when required",
                        "Document lawful basis decisions",
                        "Respect data subject rights"
                    ],
                    penalties={
                        "administrative_fine": {
                            "max_percentage": 4,
                            "max_amount": 20000000,
                            "description": "4% of annual global turnover or €20M, whichever is higher"
                        }
                    },
                    implementation_guidance=[
                        "Conduct lawful basis assessment",
                        "Implement consent management system",
                        "Create data processing records",
                        "Establish data subject request handling"
                    ],
                    validation_criteria=[
                        "Lawful basis is documented for all processing",
                        "Consent is freely given, specific, informed, and unambiguous",
                        "Data subject rights are fully implemented",
                        "Processing records are maintained and current"
                    ]
                ),
                ComplianceRule(
                    rule_id="GDPR-Article-32",
                    standard=ComplianceStandard.GDPR,
                    jurisdiction=ComplianceJurisdiction.EUROPEAN_UNION,
                    category="security_processing",
                    description="Security of processing",
                    requirements=[
                        "Implement appropriate technical measures",
                        "Ensure confidentiality, integrity, availability",
                        "Test and evaluate security effectiveness",
                        "Notify data breaches within 72 hours"
                    ],
                    penalties={
                        "administrative_fine": {
                            "max_percentage": 4,
                            "max_amount": 20000000,
                            "description": "4% of annual global turnover or €20M, whichever is higher"
                        }
                    },
                    implementation_guidance=[
                        "Deploy encryption and pseudonymization",
                        "Implement backup and recovery procedures",
                        "Conduct regular security assessments",
                        "Establish breach notification procedures"
                    ],
                    validation_criteria=[
                        "Encryption is implemented for personal data",
                        "Security measures are regularly tested",
                        "Breach detection and response procedures exist",
                        "Data protection impact assessments completed"
                    ]
                )
            ],
            ComplianceJurisdiction.SINGAPORE.value: [
                ComplianceRule(
                    rule_id="PDPA-Section-13",
                    standard=ComplianceStandard.PDPA_SG,
                    jurisdiction=ComplianceJurisdiction.SINGAPORE,
                    category="consent_obligation",
                    description="Consent obligation for personal data collection",
                    requirements=[
                        "Obtain consent before collecting personal data",
                        "Clearly state purpose of collection",
                        "Allow withdrawal of consent",
                        "Cease collection upon consent withdrawal"
                    ],
                    penalties={
                        "financial": {"max": 1000000, "description": "Up to S$1,000,000 fine"}
                    },
                    implementation_guidance=[
                        "Implement clear consent mechanisms",
                        "Provide purpose statements",
                        "Enable easy consent withdrawal",
                        "Monitor consent status continuously"
                    ],
                    validation_criteria=[
                        "Consent is obtained before data collection",
                        "Purpose statements are clear and specific",
                        "Withdrawal mechanisms are easily accessible",
                        "Collection ceases upon withdrawal"
                    ]
                )
            ]
        }
        
        return rules
    
    def _initialize_jurisdiction_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize jurisdiction-specific compliance mappings."""
        return {
            ComplianceJurisdiction.UNITED_STATES.value: {
                "primary_standards": [ComplianceStandard.HIPAA],
                "data_localization_required": False,
                "cross_border_restrictions": ["biometric_data"],
                "breach_notification_hours": 60,
                "encryption_standards": ["AES-256", "RSA-2048"],
                "audit_retention_years": 7,
                "privacy_rights": ["access", "accounting_disclosures", "amendment"],
                "regulatory_body": "Department of Health and Human Services (HHS)"
            },
            ComplianceJurisdiction.EUROPEAN_UNION.value: {
                "primary_standards": [ComplianceStandard.GDPR],
                "data_localization_required": True,
                "cross_border_restrictions": ["all_personal_data"],
                "breach_notification_hours": 72,
                "encryption_standards": ["AES-256", "RSA-4096"],
                "audit_retention_years": 3,
                "privacy_rights": [
                    "access", "rectification", "erasure", "restrict_processing",
                    "data_portability", "object", "automated_decision_making"
                ],
                "regulatory_body": "Data Protection Authorities (DPAs)"
            },
            ComplianceJurisdiction.SINGAPORE.value: {
                "primary_standards": [ComplianceStandard.PDPA_SG],
                "data_localization_required": False,
                "cross_border_restrictions": ["sensitive_personal_data"],
                "breach_notification_hours": 72,
                "encryption_standards": ["AES-256"],
                "audit_retention_years": 3,
                "privacy_rights": ["access", "correction", "withdraw_consent"],
                "regulatory_body": "Personal Data Protection Commission (PDPC)"
            }
        }
    
    def assess_compliance(
        self, 
        data_types: List[str], 
        processing_activities: List[str],
        jurisdictions: List[ComplianceJurisdiction],
        current_safeguards: Dict[str, bool]
    ) -> List[ComplianceAssessment]:
        """
        Conduct comprehensive compliance assessment across multiple jurisdictions.
        
        Args:
            data_types: Types of data being processed
            processing_activities: List of processing activities
            jurisdictions: Jurisdictions to assess compliance for
            current_safeguards: Current security and privacy safeguards implemented
            
        Returns:
            List of compliance assessments for each jurisdiction
        """
        assessments = []
        
        for jurisdiction in jurisdictions:
            self.logger.info(f"Assessing compliance for {jurisdiction.value}")
            
            # Get applicable rules for jurisdiction
            rules = self.compliance_rules.get(jurisdiction.value, [])
            
            # Assess each rule
            violations = []
            recommendations = []
            total_score = 0.0
            evaluated_rules = 0
            
            for rule in rules:
                rule_assessment = self._assess_rule_compliance(
                    rule, data_types, processing_activities, current_safeguards
                )
                
                total_score += rule_assessment["score"]
                evaluated_rules += 1
                
                if rule_assessment["violations"]:
                    violations.extend(rule_assessment["violations"])
                
                if rule_assessment["recommendations"]:
                    recommendations.extend(rule_assessment["recommendations"])
            
            # Calculate overall compliance score
            compliance_score = total_score / max(evaluated_rules, 1)
            
            # Determine risk level
            risk_level = self._calculate_risk_level(compliance_score, violations)
            
            # Create assessment
            assessment = ComplianceAssessment(
                jurisdiction=jurisdiction,
                standards_evaluated=[rule.standard for rule in rules],
                compliance_score=compliance_score,
                violations=violations,
                recommendations=recommendations,
                risk_level=risk_level,
                assessment_date=datetime.now(),
                next_review_date=datetime.now() + timedelta(days=90)
            )
            
            assessments.append(assessment)
            self.assessment_history.append(assessment)
        
        return assessments
    
    def _assess_rule_compliance(
        self,
        rule: ComplianceRule,
        data_types: List[str],
        processing_activities: List[str],
        current_safeguards: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Assess compliance with a specific rule."""
        violations = []
        recommendations = []
        score = 1.0  # Start with perfect score
        
        # Check if rule applies to current data types and activities
        if not self._rule_applies(rule, data_types, processing_activities):
            return {"score": 1.0, "violations": [], "recommendations": []}
        
        # Validate each requirement
        for requirement in rule.requirements:
            requirement_met = self._validate_requirement(requirement, current_safeguards)
            
            if not requirement_met:
                score -= 0.25  # Penalty for each unmet requirement
                
                violation = {
                    "rule_id": rule.rule_id,
                    "requirement": requirement,
                    "severity": "high" if rule.penalties else "medium",
                    "description": f"Requirement not met: {requirement}",
                    "potential_penalty": rule.penalties
                }
                violations.append(violation)
                
                # Generate recommendation
                guidance = self._get_implementation_guidance(rule, requirement)
                if guidance:
                    recommendations.append(guidance)
        
        score = max(0.0, score)  # Ensure score doesn't go below 0
        
        return {
            "score": score,
            "violations": violations,
            "recommendations": recommendations
        }
    
    def _rule_applies(
        self, 
        rule: ComplianceRule, 
        data_types: List[str], 
        processing_activities: List[str]
    ) -> bool:
        """Determine if a compliance rule applies to current context."""
        # This is a simplified implementation
        # In practice, this would involve complex mapping logic
        
        if rule.category == "data_use_disclosure":
            return any("health" in dt or "medical" in dt for dt in data_types)
        elif rule.category == "technical_safeguards":
            return any("electronic" in pa or "digital" in pa for pa in processing_activities)
        elif rule.category == "consent_obligation":
            return "collection" in processing_activities or "consent" in processing_activities
        
        return True  # Default to applicable
    
    def _validate_requirement(self, requirement: str, safeguards: Dict[str, bool]) -> bool:
        """Validate if a requirement is met by current safeguards."""
        requirement_lower = requirement.lower()
        
        # Map requirements to safeguards
        requirement_mappings = {
            "authorization": "user_authorization_system",
            "audit": "audit_logging",
            "encryption": "data_encryption",
            "access control": "access_controls",
            "authentication": "multi_factor_authentication",
            "consent": "consent_management",
            "breach notification": "breach_notification_system",
            "backup": "data_backup_recovery"
        }
        
        for keyword, safeguard_key in requirement_mappings.items():
            if keyword in requirement_lower:
                return safeguards.get(safeguard_key, False)
        
        # Default to not met if no mapping found
        return False
    
    def _get_implementation_guidance(self, rule: ComplianceRule, requirement: str) -> str:
        """Get implementation guidance for a specific requirement."""
        requirement_lower = requirement.lower()
        
        for guidance in rule.implementation_guidance:
            guidance_lower = guidance.lower()
            
            # Match guidance to requirement based on keywords
            if any(keyword in requirement_lower and keyword in guidance_lower 
                   for keyword in ["audit", "encryption", "access", "consent", "backup"]):
                return f"Implement {guidance} to satisfy: {requirement}"
        
        return f"Review implementation guidance for rule {rule.rule_id}: {requirement}"
    
    def _calculate_risk_level(self, compliance_score: float, violations: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on compliance score and violations."""
        high_severity_violations = sum(1 for v in violations if v.get("severity") == "high")
        
        if compliance_score >= 0.9 and high_severity_violations == 0:
            return "low"
        elif compliance_score >= 0.7 and high_severity_violations <= 2:
            return "medium"
        elif compliance_score >= 0.5:
            return "high"
        else:
            return "critical"
    
    def generate_cross_border_transfer_assessment(
        self, 
        source_jurisdiction: ComplianceJurisdiction,
        target_jurisdiction: ComplianceJurisdiction,
        data_categories: List[str]
    ) -> Dict[str, Any]:
        """Generate assessment for cross-border data transfers."""
        
        assessment = {
            "source_jurisdiction": source_jurisdiction.value,
            "target_jurisdiction": target_jurisdiction.value,
            "transfer_allowed": True,
            "restrictions": [],
            "required_safeguards": [],
            "legal_mechanisms": [],
            "risk_assessment": "low"
        }
        
        # Check for data localization requirements
        source_mapping = self.jurisdiction_mappings.get(source_jurisdiction.value, {})
        target_mapping = self.jurisdiction_mappings.get(target_jurisdiction.value, {})
        
        if source_mapping.get("data_localization_required", False):
            assessment["transfer_allowed"] = False
            assessment["restrictions"].append("Data localization required in source jurisdiction")
            assessment["risk_assessment"] = "high"
        
        # Check category-specific restrictions
        for category in data_categories:
            phi_category = self.phi_categories.get(category)
            if phi_category:
                restrictions = phi_category.cross_border_restrictions
                if not restrictions.get(target_jurisdiction.value.split("_")[0], True):
                    assessment["transfer_allowed"] = False
                    assessment["restrictions"].append(
                        f"Cross-border transfer of {category} restricted to {target_jurisdiction.value}"
                    )
                    assessment["risk_assessment"] = "high"
        
        # Determine required safeguards
        if source_jurisdiction == ComplianceJurisdiction.EUROPEAN_UNION:
            assessment["legal_mechanisms"].extend([
                "Adequacy Decision",
                "Standard Contractual Clauses (SCCs)",
                "Binding Corporate Rules (BCRs)",
                "Certification Mechanisms"
            ])
            assessment["required_safeguards"].extend([
                "Technical and organizational measures",
                "Data subject rights preservation",
                "Supervisory authority cooperation"
            ])
        
        if target_jurisdiction == ComplianceJurisdiction.UNITED_STATES:
            if source_jurisdiction == ComplianceJurisdiction.EUROPEAN_UNION:
                assessment["legal_mechanisms"].append("Privacy Shield successor framework")
                assessment["required_safeguards"].append("US government surveillance limitations")
        
        return assessment
    
    def generate_compliance_report(self, assessments: List[ComplianceAssessment]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "report_date": datetime.now().isoformat(),
            "jurisdictions_assessed": len(assessments),
            "overall_compliance_score": sum(a.compliance_score for a in assessments) / len(assessments),
            "total_violations": sum(len(a.violations) for a in assessments),
            "risk_distribution": {},
            "jurisdiction_summary": {},
            "priority_recommendations": [],
            "compliance_trends": {},
            "next_review_dates": {}
        }
        
        # Risk distribution
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for assessment in assessments:
            risk_counts[assessment.risk_level] += 1
        report["risk_distribution"] = risk_counts
        
        # Jurisdiction summaries
        for assessment in assessments:
            jurisdiction_key = assessment.jurisdiction.value
            report["jurisdiction_summary"][jurisdiction_key] = {
                "compliance_score": assessment.compliance_score,
                "risk_level": assessment.risk_level,
                "violations_count": len(assessment.violations),
                "standards_evaluated": [s.value for s in assessment.standards_evaluated],
                "next_review": assessment.next_review_date.isoformat()
            }
            
            report["next_review_dates"][jurisdiction_key] = assessment.next_review_date.isoformat()
        
        # Priority recommendations (top 10)
        all_recommendations = []
        for assessment in assessments:
            for rec in assessment.recommendations:
                all_recommendations.append({
                    "recommendation": rec,
                    "jurisdiction": assessment.jurisdiction.value,
                    "priority": self._calculate_recommendation_priority(rec, assessment)
                })
        
        # Sort by priority and take top 10
        sorted_recommendations = sorted(all_recommendations, key=lambda x: x["priority"], reverse=True)
        report["priority_recommendations"] = sorted_recommendations[:10]
        
        # Compliance trends (simplified)
        if len(self.assessment_history) > 1:
            report["compliance_trends"] = self._calculate_compliance_trends()
        
        return report
    
    def _calculate_recommendation_priority(self, recommendation: str, assessment: ComplianceAssessment) -> int:
        """Calculate priority score for a recommendation."""
        priority = 1
        
        # Higher priority for high-risk jurisdictions
        if assessment.risk_level == "critical":
            priority += 10
        elif assessment.risk_level == "high":
            priority += 5
        elif assessment.risk_level == "medium":
            priority += 2
        
        # Higher priority for security-related recommendations
        security_keywords = ["encryption", "authentication", "audit", "breach", "security"]
        if any(keyword in recommendation.lower() for keyword in security_keywords):
            priority += 3
        
        # Higher priority for mandatory compliance items
        mandatory_keywords = ["required", "mandatory", "must", "shall"]
        if any(keyword in recommendation.lower() for keyword in mandatory_keywords):
            priority += 5
        
        return priority
    
    def _calculate_compliance_trends(self) -> Dict[str, Any]:
        """Calculate compliance trends from assessment history."""
        if len(self.assessment_history) < 2:
            return {}
        
        trends = {}
        
        # Group assessments by jurisdiction
        by_jurisdiction = {}
        for assessment in self.assessment_history:
            jurisdiction = assessment.jurisdiction.value
            if jurisdiction not in by_jurisdiction:
                by_jurisdiction[jurisdiction] = []
            by_jurisdiction[jurisdiction].append(assessment)
        
        # Calculate trends for each jurisdiction
        for jurisdiction, assessments in by_jurisdiction.items():
            if len(assessments) >= 2:
                # Sort by date
                sorted_assessments = sorted(assessments, key=lambda x: x.assessment_date)
                
                # Calculate score trend
                recent_scores = [a.compliance_score for a in sorted_assessments[-3:]]  # Last 3 assessments
                if len(recent_scores) >= 2:
                    score_trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                    if abs(recent_scores[-1] - recent_scores[0]) < 0.05:
                        score_trend = "stable"
                    
                    trends[jurisdiction] = {
                        "score_trend": score_trend,
                        "current_score": recent_scores[-1],
                        "previous_score": recent_scores[-2] if len(recent_scores) >= 2 else recent_scores[0],
                        "score_change": recent_scores[-1] - recent_scores[-2] if len(recent_scores) >= 2 else 0
                    }
        
        return trends
    
    def get_jurisdiction_requirements(self, jurisdiction: ComplianceJurisdiction) -> Dict[str, Any]:
        """Get specific requirements for a jurisdiction."""
        return {
            "jurisdiction": jurisdiction.value,
            "mapping": self.jurisdiction_mappings.get(jurisdiction.value, {}),
            "compliance_rules": self.compliance_rules.get(jurisdiction.value, []),
            "phi_categories": {
                name: {
                    "applicable": any(std in category.applicable_standards 
                                   for std in self.jurisdiction_mappings.get(jurisdiction.value, {}).get("primary_standards", [])),
                    "retention_days": category.retention_period_days,
                    "encryption_required": category.encryption_required,
                    "cross_border_allowed": category.cross_border_restrictions.get(jurisdiction.value.split("_")[0], True)
                }
                for name, category in self.phi_categories.items()
            }
        }


# Global compliance framework instance
global_compliance_framework = GlobalComplianceFramework()


def assess_multi_jurisdiction_compliance(
    data_types: List[str],
    processing_activities: List[str],
    target_jurisdictions: List[str],
    current_safeguards: Dict[str, bool]
) -> List[ComplianceAssessment]:
    """
    Convenience function for multi-jurisdiction compliance assessment.
    
    Usage:
        assessments = assess_multi_jurisdiction_compliance(
            data_types=["health_data", "direct_identifiers"],
            processing_activities=["collection", "storage", "analysis"],
            target_jurisdictions=["US", "EU", "SG"],
            current_safeguards={
                "data_encryption": True,
                "audit_logging": True,
                "access_controls": True,
                "multi_factor_authentication": False
            }
        )
    """
    jurisdictions = []
    for jurisdiction_code in target_jurisdictions:
        for jurisdiction in ComplianceJurisdiction:
            if jurisdiction.value == jurisdiction_code:
                jurisdictions.append(jurisdiction)
                break
    
    return global_compliance_framework.assess_compliance(
        data_types=data_types,
        processing_activities=processing_activities,
        jurisdictions=jurisdictions,
        current_safeguards=current_safeguards
    )