"""Business rules engine for HIPAA compliance decision making."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..models.phi_entity import PHICategory
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Types of business rules."""
    VALIDATION = "validation"
    COMPLIANCE = "compliance"
    REDACTION = "redaction"
    RISK_ASSESSMENT = "risk_assessment"
    WORKFLOW = "workflow"


class RuleAction(str, Enum):
    """Actions that rules can trigger."""
    ALLOW = "allow"
    DENY = "deny"
    REDACT = "redact"
    FLAG = "flag"
    ESCALATE = "escalate"
    REQUIRE_REVIEW = "require_review"


@dataclass
class RuleCondition:
    """Represents a condition in a business rule."""
    
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any
    description: Optional[str] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against a context."""
        try:
            field_value = self._get_field_value(context, self.field)
            
            if self.operator == "eq":
                return field_value == self.value
            elif self.operator == "ne":
                return field_value != self.value
            elif self.operator == "gt":
                return field_value > self.value
            elif self.operator == "lt":
                return field_value < self.value
            elif self.operator == "gte":
                return field_value >= self.value
            elif self.operator == "lte":
                return field_value <= self.value
            elif self.operator == "in":
                return field_value in self.value
            elif self.operator == "not_in":
                return field_value not in self.value
            elif self.operator == "contains":
                return self.value in str(field_value)
            elif self.operator == "regex":
                import re
                return bool(re.search(self.value, str(field_value)))
            else:
                logger.warning(f"Unknown operator: {self.operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition {self.field} {self.operator} {self.value}: {e}")
            return False
    
    def _get_field_value(self, context: Dict[str, Any], field_path: str) -> Any:
        """Get field value from context using dot notation."""
        value = context
        for part in field_path.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value


@dataclass
class RuleResult:
    """Result of rule execution."""
    
    rule_name: str
    rule_type: RuleType
    action: RuleAction
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "action": self.action.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "executed_at": self.executed_at.isoformat()
        }


class Rule(ABC):
    """Base class for business rules."""
    
    def __init__(self, name: str, rule_type: RuleType, priority: int = 100,
                 enabled: bool = True, description: str = ""):
        """Initialize rule.
        
        Args:
            name: Rule name
            rule_type: Type of rule
            priority: Rule priority (lower numbers = higher priority)
            enabled: Whether rule is enabled
            description: Rule description
        """
        self.name = name
        self.rule_type = rule_type
        self.priority = priority
        self.enabled = enabled
        self.description = description
        self.execution_count = 0
        self.last_executed = None
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """Evaluate the rule against a context.
        
        Args:
            context: Context containing data to evaluate
            
        Returns:
            Rule result
        """
        pass
    
    def execute(self, context: Dict[str, Any]) -> Optional[RuleResult]:
        """Execute the rule if enabled.
        
        Args:
            context: Context to evaluate
            
        Returns:
            Rule result or None if disabled
        """
        if not self.enabled:
            return None
        
        try:
            self.execution_count += 1
            self.last_executed = datetime.utcnow()
            
            result = self.evaluate(context)
            result.metadata.update({
                "rule_priority": self.priority,
                "execution_count": self.execution_count
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Rule {self.name} execution failed: {e}")
            return RuleResult(
                rule_name=self.name,
                rule_type=self.rule_type,
                action=RuleAction.FLAG,
                passed=False,
                message=f"Rule execution error: {str(e)}",
                details={"error": str(e)},
                confidence=0.0
            )


class ComplianceRule(Rule):
    """Rule for HIPAA compliance validation."""
    
    def __init__(self, name: str, conditions: List[RuleCondition], 
                 action: RuleAction, priority: int = 100, **kwargs):
        super().__init__(name, RuleType.COMPLIANCE, priority, **kwargs)
        self.conditions = conditions
        self.action = action
    
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """Evaluate compliance rule."""
        # Evaluate all conditions
        condition_results = []
        for condition in self.conditions:
            result = condition.evaluate(context)
            condition_results.append({
                "condition": f"{condition.field} {condition.operator} {condition.value}",
                "passed": result,
                "description": condition.description
            })
        
        # All conditions must pass for compliance
        all_passed = all(r["passed"] for r in condition_results)
        
        message = f"Compliance rule '{self.name}': {'PASSED' if all_passed else 'FAILED'}"
        if not all_passed:
            failed_conditions = [r for r in condition_results if not r["passed"]]
            message += f" - {len(failed_conditions)} condition(s) failed"
        
        recommendations = []
        if not all_passed:
            if self.action == RuleAction.REDACT:
                recommendations.append("Redact identified PHI before proceeding")
            elif self.action == RuleAction.REQUIRE_REVIEW:
                recommendations.append("Manual review required before approval")
            elif self.action == RuleAction.ESCALATE:
                recommendations.append("Escalate to compliance officer")
        
        return RuleResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            action=self.action if not all_passed else RuleAction.ALLOW,
            passed=all_passed,
            message=message,
            details={
                "conditions_evaluated": condition_results,
                "total_conditions": len(self.conditions),
                "passed_conditions": sum(1 for r in condition_results if r["passed"])
            },
            recommendations=recommendations
        )


class RedactionRule(Rule):
    """Rule for determining redaction requirements."""
    
    def __init__(self, name: str, phi_categories: List[str], 
                 risk_threshold: float = 0.7, priority: int = 100, **kwargs):
        super().__init__(name, RuleType.REDACTION, priority, **kwargs)
        self.phi_categories = phi_categories
        self.risk_threshold = risk_threshold
    
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """Evaluate redaction rule."""
        phi_entities = context.get("phi_entities", [])
        risk_analysis = context.get("risk_analysis")
        
        # Check for target PHI categories
        found_categories = set()
        entities_to_redact = []
        
        for entity in phi_entities:
            entity_category = getattr(entity, 'category', None)
            if entity_category in self.phi_categories:
                found_categories.add(entity_category)
                entities_to_redact.append(entity)
        
        # Check risk threshold
        risk_score = getattr(risk_analysis, 'overall_risk_score', 0.0) if risk_analysis else 0.0
        risk_exceeds_threshold = risk_score >= self.risk_threshold
        
        # Determine if redaction is required
        requires_redaction = bool(found_categories) or risk_exceeds_threshold
        
        message = f"Redaction rule '{self.name}': "
        if requires_redaction:
            message += f"REDACTION REQUIRED - Found {len(entities_to_redact)} entities"
            if risk_exceeds_threshold:
                message += f", risk score {risk_score:.2f} exceeds threshold {self.risk_threshold}"
        else:
            message += "No redaction required"
        
        recommendations = []
        if requires_redaction:
            recommendations.append(f"Redact {len(entities_to_redact)} PHI entities")
            if found_categories:
                recommendations.append(f"Focus on categories: {', '.join(found_categories)}")
            if risk_exceeds_threshold:
                recommendations.append("Apply enhanced redaction due to high risk")
        
        return RuleResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            action=RuleAction.REDACT if requires_redaction else RuleAction.ALLOW,
            passed=not requires_redaction,  # "Passed" means no redaction needed
            message=message,
            details={
                "target_categories": self.phi_categories,
                "found_categories": list(found_categories),
                "entities_to_redact": len(entities_to_redact),
                "risk_score": risk_score,
                "risk_threshold": self.risk_threshold,
                "risk_exceeds_threshold": risk_exceeds_threshold
            },
            recommendations=recommendations,
            confidence=0.9 if found_categories else 0.7
        )


class ValidationRule(Rule):
    """Rule for document validation."""
    
    def __init__(self, name: str, validators: List[Callable], priority: int = 100, **kwargs):
        super().__init__(name, RuleType.VALIDATION, priority, **kwargs)
        self.validators = validators
    
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """Evaluate validation rule."""
        validation_results = []
        all_valid = True
        
        for i, validator in enumerate(self.validators):
            try:
                result = validator(context)
                if isinstance(result, bool):
                    validation_results.append({
                        "validator": f"validator_{i+1}",
                        "valid": result,
                        "message": "Validation passed" if result else "Validation failed"
                    })
                    if not result:
                        all_valid = False
                elif isinstance(result, dict):
                    validation_results.append(result)
                    if not result.get("valid", False):
                        all_valid = False
                        
            except Exception as e:
                validation_results.append({
                    "validator": f"validator_{i+1}",
                    "valid": False,
                    "message": f"Validator error: {str(e)}"
                })
                all_valid = False
        
        message = f"Validation rule '{self.name}': {'PASSED' if all_valid else 'FAILED'}"
        if not all_valid:
            failed_count = sum(1 for r in validation_results if not r["valid"])
            message += f" - {failed_count} validator(s) failed"
        
        return RuleResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            action=RuleAction.ALLOW if all_valid else RuleAction.DENY,
            passed=all_valid,
            message=message,
            details={
                "validation_results": validation_results,
                "total_validators": len(self.validators),
                "passed_validators": sum(1 for r in validation_results if r["valid"])
            }
        )


class BusinessRulesEngine:
    """Engine for executing business rules."""
    
    def __init__(self):
        """Initialize business rules engine."""
        self.rules: List[Rule] = []
        self.rule_history: List[RuleResult] = []
        self.max_history = 10000
    
    def add_rule(self, rule: Rule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Sort by priority (lower number = higher priority)
        self.rules.sort(key=lambda r: r.priority)
        logger.info(f"Added rule: {rule.name} (type: {rule.rule_type.value}, priority: {rule.priority})")
    
    def remove_rule(self, rule_name: str):
        """Remove a rule from the engine."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed rule: {rule_name}")
    
    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def enable_rule(self, rule_name: str):
        """Enable a rule."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
            logger.info(f"Enabled rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a rule."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
            logger.info(f"Disabled rule: {rule_name}")
    
    @trace_operation("business_rules_execution")
    def execute_rules(self, context: Dict[str, Any], 
                     rule_types: List[RuleType] = None) -> List[RuleResult]:
        """Execute all applicable rules.
        
        Args:
            context: Context to evaluate rules against
            rule_types: Optional filter for rule types to execute
            
        Returns:
            List of rule results
        """
        results = []
        
        # Filter rules by type if specified
        rules_to_execute = self.rules
        if rule_types:
            rules_to_execute = [r for r in self.rules if r.rule_type in rule_types]
        
        logger.info(f"Executing {len(rules_to_execute)} rules")
        
        for rule in rules_to_execute:
            try:
                result = rule.execute(context)
                if result:
                    results.append(result)
                    
                    # Store in history
                    self.rule_history.append(result)
                    if len(self.rule_history) > self.max_history:
                        self.rule_history = self.rule_history[-self.max_history:]
                    
                    logger.debug(f"Rule {rule.name}: {result.action.value} ({'passed' if result.passed else 'failed'})")
                    
            except Exception as e:
                logger.error(f"Failed to execute rule {rule.name}: {e}")
        
        logger.info(f"Executed {len(results)} rules, {sum(1 for r in results if not r.passed)} failed")
        return results
    
    def execute_compliance_rules(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Execute only compliance rules."""
        return self.execute_rules(context, [RuleType.COMPLIANCE])
    
    def execute_redaction_rules(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Execute only redaction rules."""
        return self.execute_rules(context, [RuleType.REDACTION])
    
    def execute_validation_rules(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Execute only validation rules."""
        return self.execute_rules(context, [RuleType.VALIDATION])
    
    def get_failed_rules(self, results: List[RuleResult]) -> List[RuleResult]:
        """Get rules that failed."""
        return [r for r in results if not r.passed]
    
    def get_rules_by_action(self, results: List[RuleResult], action: RuleAction) -> List[RuleResult]:
        """Get rules that triggered a specific action."""
        return [r for r in results if r.action == action]
    
    def should_block_processing(self, results: List[RuleResult]) -> bool:
        """Determine if processing should be blocked based on rule results."""
        blocking_actions = [RuleAction.DENY, RuleAction.ESCALATE]
        return any(r.action in blocking_actions for r in results)
    
    def should_require_redaction(self, results: List[RuleResult]) -> bool:
        """Determine if redaction is required based on rule results."""
        return any(r.action == RuleAction.REDACT for r in results)
    
    def should_require_review(self, results: List[RuleResult]) -> bool:
        """Determine if manual review is required based on rule results."""
        return any(r.action == RuleAction.REQUIRE_REVIEW for r in results)
    
    def get_recommendations(self, results: List[RuleResult]) -> List[str]:
        """Get all recommendations from rule results."""
        recommendations = []
        for result in results:
            recommendations.extend(result.recommendations)
        return list(set(recommendations))  # Remove duplicates
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of all rules."""
        enabled_rules = [r for r in self.rules if r.enabled]
        rule_types = {}
        
        for rule_type in RuleType:
            rule_types[rule_type.value] = len([r for r in enabled_rules if r.rule_type == rule_type])
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len(enabled_rules),
            "disabled_rules": len(self.rules) - len(enabled_rules),
            "rules_by_type": rule_types,
            "execution_history_size": len(self.rule_history)
        }
    
    def setup_default_rules(self):
        """Setup default HIPAA compliance rules."""
        
        # High-risk PHI redaction rule
        self.add_rule(RedactionRule(
            name="high_risk_phi_redaction",
            phi_categories=[
                PHICategory.SOCIAL_SECURITY_NUMBERS.value,
                PHICategory.BIOMETRIC_IDENTIFIERS.value,
                PHICategory.FULL_FACE_PHOTOS.value
            ],
            risk_threshold=0.8,
            priority=10,
            description="Require redaction of high-risk PHI categories"
        ))
        
        # Safe Harbor compliance rule
        safe_harbor_conditions = [
            RuleCondition("compliance_analysis.safe_harbor_compliance", "eq", True,
                         "Document must meet Safe Harbor requirements")
        ]
        
        self.add_rule(ComplianceRule(
            name="safe_harbor_compliance",
            conditions=safe_harbor_conditions,
            action=RuleAction.REQUIRE_REVIEW,
            priority=20,
            description="Ensure Safe Harbor de-identification compliance"
        ))
        
        # High compliance score validation
        compliance_score_conditions = [
            RuleCondition("compliance_analysis.overall_compliance_score", "gte", 95.0,
                         "Compliance score must be at least 95%")
        ]
        
        self.add_rule(ComplianceRule(
            name="minimum_compliance_score",
            conditions=compliance_score_conditions,
            action=RuleAction.REQUIRE_REVIEW,
            priority=30,
            description="Require minimum compliance score"
        ))
        
        # Critical risk escalation rule
        critical_risk_conditions = [
            RuleCondition("risk_analysis.risk_level", "ne", "critical",
                         "Documents with critical risk require escalation")
        ]
        
        self.add_rule(ComplianceRule(
            name="critical_risk_escalation",
            conditions=critical_risk_conditions,
            action=RuleAction.ESCALATE,
            priority=5,  # Highest priority
            description="Escalate documents with critical risk level"
        ))
        
        # Document validation rule
        def validate_document_content(context):
            content = context.get("content", "")
            return {
                "valid": len(content.strip()) > 0,
                "message": "Document must have content" if len(content.strip()) == 0 else "Content validation passed"
            }
        
        def validate_phi_detection(context):
            phi_entities = context.get("phi_entities", [])
            return {
                "valid": True,  # Always pass, but log if no PHI found
                "message": f"PHI detection completed: {len(phi_entities)} entities found"
            }
        
        self.add_rule(ValidationRule(
            name="document_content_validation",
            validators=[validate_document_content, validate_phi_detection],
            priority=100,
            description="Validate basic document requirements"
        ))
        
        logger.info("Setup default HIPAA compliance rules")