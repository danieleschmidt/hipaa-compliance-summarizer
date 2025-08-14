"""Global compliance standards implementation for healthcare data processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
from datetime import datetime


class ComplianceStandard(str, Enum):
    """Supported global compliance standards."""
    HIPAA = "HIPAA"  # USA - Health Insurance Portability and Accountability Act
    GDPR = "GDPR"    # EU - General Data Protection Regulation  
    PDPA = "PDPA"    # Singapore - Personal Data Protection Act
    CCPA = "CCPA"    # California - California Consumer Privacy Act
    PIPEDA = "PIPEDA" # Canada - Personal Information Protection and Electronic Documents Act
    SOX = "SOX"      # USA - Sarbanes-Oxley Act
    HITRUST = "HITRUST" # Healthcare Information Trust Alliance


class Region(str, Enum):
    """Global regions with different compliance requirements."""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"  
    ASIA_PACIFIC = "asia_pacific"
    SOUTH_AMERICA = "south_america"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"


@dataclass
class ComplianceRule:
    """Individual compliance rule definition."""
    
    rule_id: str
    standard: ComplianceStandard
    title: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    regions: List[Region]
    data_categories: List[str]  # Types of data this rule applies to
    
    def applies_to_data(self, data_type: str) -> bool:
        """Check if this rule applies to given data type."""
        return data_type.lower() in [cat.lower() for cat in self.data_categories]


class GlobalComplianceManager:
    """Manager for multi-region compliance standards."""
    
    def __init__(self):
        """Initialize compliance manager with global standards."""
        self.compliance_rules = self._initialize_compliance_rules()
        self.regional_requirements = self._initialize_regional_requirements()
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceStandard, List[ComplianceRule]]:
        """Initialize compliance rules for different standards."""
        rules = {
            ComplianceStandard.HIPAA: [
                ComplianceRule(
                    rule_id="HIPAA-001",
                    standard=ComplianceStandard.HIPAA,
                    title="PHI Encryption at Rest",
                    description="All PHI must be encrypted when stored",
                    severity="critical",
                    regions=[Region.NORTH_AMERICA],
                    data_categories=["phi", "medical_records", "patient_data"]
                ),
                ComplianceRule(
                    rule_id="HIPAA-002", 
                    standard=ComplianceStandard.HIPAA,
                    title="Access Logging",
                    description="All access to PHI must be logged and auditable",
                    severity="high",
                    regions=[Region.NORTH_AMERICA],
                    data_categories=["phi", "medical_records", "patient_data"]
                ),
                ComplianceRule(
                    rule_id="HIPAA-003",
                    standard=ComplianceStandard.HIPAA, 
                    title="Minimum Necessary Rule",
                    description="Access to PHI should be limited to minimum necessary",
                    severity="high",
                    regions=[Region.NORTH_AMERICA],
                    data_categories=["phi", "medical_records", "patient_data"]
                )
            ],
            
            ComplianceStandard.GDPR: [
                ComplianceRule(
                    rule_id="GDPR-001",
                    standard=ComplianceStandard.GDPR,
                    title="Data Subject Rights",
                    description="Implement data subject access, rectification, and erasure rights",
                    severity="critical", 
                    regions=[Region.EUROPE],
                    data_categories=["personal_data", "sensitive_data", "biometric_data"]
                ),
                ComplianceRule(
                    rule_id="GDPR-002",
                    standard=ComplianceStandard.GDPR,
                    title="Data Processing Lawfulness",
                    description="Ensure lawful basis for processing personal data",
                    severity="critical",
                    regions=[Region.EUROPE],
                    data_categories=["personal_data", "sensitive_data"]
                ),
                ComplianceRule(
                    rule_id="GDPR-003",
                    standard=ComplianceStandard.GDPR,
                    title="Privacy by Design",
                    description="Implement privacy protection measures from the design stage",
                    severity="high",
                    regions=[Region.EUROPE],
                    data_categories=["personal_data", "sensitive_data"]
                )
            ],
            
            ComplianceStandard.PDPA: [
                ComplianceRule(
                    rule_id="PDPA-001",
                    standard=ComplianceStandard.PDPA,
                    title="Consent Management",
                    description="Obtain and manage individual consent for data processing",
                    severity="critical",
                    regions=[Region.ASIA_PACIFIC],
                    data_categories=["personal_data", "sensitive_data"]
                ),
                ComplianceRule(
                    rule_id="PDPA-002",
                    standard=ComplianceStandard.PDPA,
                    title="Data Breach Notification",
                    description="Notify authorities and individuals of data breaches",
                    severity="high",
                    regions=[Region.ASIA_PACIFIC],
                    data_categories=["personal_data", "sensitive_data"]
                )
            ]
        }
        return rules
    
    def _initialize_regional_requirements(self) -> Dict[Region, List[ComplianceStandard]]:
        """Map regions to applicable compliance standards."""
        return {
            Region.NORTH_AMERICA: [ComplianceStandard.HIPAA, ComplianceStandard.CCPA, ComplianceStandard.SOX],
            Region.EUROPE: [ComplianceStandard.GDPR],
            Region.ASIA_PACIFIC: [ComplianceStandard.PDPA], 
            Region.SOUTH_AMERICA: [],
            Region.AFRICA: [],
            Region.MIDDLE_EAST: []
        }
    
    def get_applicable_standards(self, region: Region) -> List[ComplianceStandard]:
        """Get compliance standards applicable to a region."""
        return self.regional_requirements.get(region, [])
    
    def get_rules_for_standard(self, standard: ComplianceStandard) -> List[ComplianceRule]:
        """Get all rules for a compliance standard."""
        return self.compliance_rules.get(standard, [])
    
    def get_rules_for_data_type(self, data_type: str, region: Optional[Region] = None) -> List[ComplianceRule]:
        """Get applicable rules for a specific data type and region."""
        applicable_rules = []
        
        # Get standards for region if specified
        if region:
            applicable_standards = self.get_applicable_standards(region)
        else:
            applicable_standards = list(self.compliance_rules.keys())
        
        # Find rules that apply to the data type
        for standard in applicable_standards:
            for rule in self.get_rules_for_standard(standard):
                if rule.applies_to_data(data_type):
                    # Filter by region if specified
                    if region is None or region in rule.regions:
                        applicable_rules.append(rule)
        
        return applicable_rules
    
    def validate_compliance(self, data_type: str, region: Region, 
                          implemented_controls: Set[str]) -> Dict:
        """Validate compliance for given data type and region."""
        applicable_rules = self.get_rules_for_data_type(data_type, region)
        
        compliance_results = {
            'total_rules': len(applicable_rules),
            'compliant_rules': 0,
            'non_compliant_rules': [],
            'compliance_score': 0.0,
            'recommendations': []
        }
        
        for rule in applicable_rules:
            # Simple check if rule is implemented (would be more complex in real implementation)
            if rule.rule_id.lower() in [control.lower() for control in implemented_controls]:
                compliance_results['compliant_rules'] += 1
            else:
                compliance_results['non_compliant_rules'].append({
                    'rule_id': rule.rule_id,
                    'title': rule.title,
                    'severity': rule.severity,
                    'description': rule.description
                })
                
                # Generate recommendations
                compliance_results['recommendations'].append(
                    f"Implement {rule.title}: {rule.description}"
                )
        
        # Calculate compliance score
        if compliance_results['total_rules'] > 0:
            compliance_results['compliance_score'] = (
                compliance_results['compliant_rules'] / compliance_results['total_rules']
            )
        else:
            compliance_results['compliance_score'] = 1.0
        
        return compliance_results
    
    def get_supported_regions(self) -> List[Region]:
        """Get list of supported regions."""
        return list(Region)
    
    def get_supported_standards(self) -> List[ComplianceStandard]:
        """Get list of supported compliance standards."""
        return list(ComplianceStandard)


# Global compliance manager instance
_global_compliance_manager = None

def get_compliance_manager() -> GlobalComplianceManager:
    """Get the global compliance manager instance."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = GlobalComplianceManager()
    return _global_compliance_manager

def validate_regional_compliance(data_type: str, region: Region, 
                               implemented_controls: Set[str]) -> Dict:
    """Validate compliance for data processing in a specific region."""
    return get_compliance_manager().validate_compliance(
        data_type, region, implemented_controls
    )

def get_regional_requirements(region: Region) -> List[ComplianceStandard]:
    """Get compliance requirements for a specific region."""
    return get_compliance_manager().get_applicable_standards(region)