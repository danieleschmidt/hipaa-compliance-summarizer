"""Worker for compliance-specific background jobs."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .queue_manager import Job, JobResult
from ..rules.business_rules import BusinessRulesEngine, RuleType
from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


class ComplianceWorker:
    """Worker for compliance monitoring and enforcement jobs."""
    
    def __init__(self, rules_engine: BusinessRulesEngine):
        """Initialize compliance worker.
        
        Args:
            rules_engine: Business rules engine instance
        """
        self.rules_engine = rules_engine
        self.compliance_checks = 0
        self.violations_found = 0
    
    @trace_operation("compliance_check_worker")
    async def process_compliance_check_job(self, job: Job) -> Dict[str, Any]:
        """Process a compliance check job.
        
        Args:
            job: Job containing data for compliance check
            
        Returns:
            Compliance check result
        """
        logger.info(f"Processing compliance check job {job.job_id}")
        
        try:
            # Extract job data
            document_data = job.data.get("document_data")
            check_type = job.data.get("check_type", "full")  # full, compliance_only, validation_only
            
            if not document_data:
                raise ValueError("No document data provided for compliance check")
            
            # Prepare context for rules engine
            context = {
                "document_id": document_data.get("document_id"),
                "content": document_data.get("content"),
                "phi_entities": document_data.get("phi_entities", []),
                "compliance_analysis": document_data.get("compliance_analysis"),
                "risk_analysis": document_data.get("risk_analysis"),
                "document_analysis": document_data.get("document_analysis")
            }
            
            # Execute appropriate rules based on check type
            if check_type == "compliance_only":
                rule_results = self.rules_engine.execute_compliance_rules(context)
            elif check_type == "validation_only":
                rule_results = self.rules_engine.execute_validation_rules(context)
            else:  # full check
                rule_results = self.rules_engine.execute_rules(context)
            
            # Analyze results
            failed_rules = self.rules_engine.get_failed_rules(rule_results)
            critical_violations = [r for r in failed_rules if r.action.value == "escalate"]
            
            # Update statistics
            self.compliance_checks += 1
            if failed_rules:
                self.violations_found += len(failed_rules)
            
            # Determine overall compliance status
            compliance_status = "compliant"
            if critical_violations:
                compliance_status = "critical_violations"
            elif failed_rules:
                compliance_status = "violations_found"
            
            # Prepare result
            result = {
                "compliance_status": compliance_status,
                "total_rules_checked": len(rule_results),
                "passed_rules": len(rule_results) - len(failed_rules),
                "failed_rules": len(failed_rules),
                "critical_violations": len(critical_violations),
                "requires_redaction": self.rules_engine.should_require_redaction(rule_results),
                "requires_review": self.rules_engine.should_require_review(rule_results),
                "should_block": self.rules_engine.should_block_processing(rule_results),
                "rule_results": [r.to_dict() for r in rule_results],
                "recommendations": self.rules_engine.get_recommendations(rule_results),
                "check_type": check_type
            }
            
            logger.info(f"Compliance check job {job.job_id} completed: {compliance_status}")
            return result
            
        except Exception as e:
            logger.error(f"Compliance check job {job.job_id} failed: {e}")
            raise
    
    @trace_operation("audit_report_worker") 
    async def process_audit_report_job(self, job: Job) -> Dict[str, Any]:
        """Process an audit report generation job.
        
        Args:
            job: Job containing audit report parameters
            
        Returns:
            Audit report
        """
        logger.info(f"Processing audit report job {job.job_id}")
        
        try:
            # Extract job parameters
            report_type = job.data.get("report_type", "compliance")  # compliance, risk, summary
            time_period_hours = job.data.get("time_period_hours", 24)
            include_details = job.data.get("include_details", True)
            document_ids = job.data.get("document_ids", [])  # Optional filter
            
            # Get historical data (this would typically come from a database)
            # For now, we'll use the rules engine history
            cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
            
            # Filter rule history by time
            recent_rule_results = [
                r for r in self.rules_engine.rule_history
                if r.executed_at >= cutoff_time
            ]
            
            # Generate report based on type
            if report_type == "compliance":
                report = self._generate_compliance_report(recent_rule_results, time_period_hours, include_details)
            elif report_type == "risk":
                report = self._generate_risk_report(recent_rule_results, time_period_hours, include_details)
            else:  # summary
                report = self._generate_summary_report(recent_rule_results, time_period_hours)
            
            # Add metadata
            report["metadata"] = {
                "report_type": report_type,
                "time_period_hours": time_period_hours,
                "generated_at": datetime.utcnow().isoformat(),
                "total_rule_executions": len(recent_rule_results),
                "document_filter": document_ids if document_ids else "all"
            }
            
            logger.info(f"Audit report job {job.job_id} completed: {report_type} report generated")
            return report
            
        except Exception as e:
            logger.error(f"Audit report job {job.job_id} failed: {e}")
            raise
    
    def _generate_compliance_report(self, rule_results: List[Any], 
                                   time_period_hours: int, include_details: bool) -> Dict[str, Any]:
        """Generate compliance-focused audit report."""
        compliance_results = [r for r in rule_results if r.rule_type.value == "compliance"]
        
        total_checks = len(compliance_results)
        passed_checks = sum(1 for r in compliance_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Analyze violations by severity
        violations_by_action = {}
        for result in compliance_results:
            if not result.passed:
                action = result.action.value
                violations_by_action[action] = violations_by_action.get(action, 0) + 1
        
        # Get most common violations
        violation_rules = [r for r in compliance_results if not r.passed]
        common_violations = {}
        for violation in violation_rules:
            rule_name = violation.rule_name
            common_violations[rule_name] = common_violations.get(rule_name, 0) + 1
        
        report = {
            "summary": {
                "total_compliance_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "compliance_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 100,
                "time_period_hours": time_period_hours
            },
            "violations": {
                "by_severity": violations_by_action,
                "most_common": dict(sorted(common_violations.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        }
        
        if include_details:
            report["detailed_results"] = [r.to_dict() for r in compliance_results]
        
        return report
    
    def _generate_risk_report(self, rule_results: List[Any], 
                             time_period_hours: int, include_details: bool) -> Dict[str, Any]:
        """Generate risk-focused audit report."""
        risk_results = [r for r in rule_results if r.rule_type.value == "risk_assessment"]
        
        # Analyze risk patterns
        high_risk_events = [r for r in risk_results if not r.passed and r.confidence >= 0.8]
        escalation_events = [r for r in rule_results if r.action.value == "escalate"]
        
        report = {
            "summary": {
                "total_risk_assessments": len(risk_results),
                "high_risk_events": len(high_risk_events),
                "escalation_events": len(escalation_events),
                "time_period_hours": time_period_hours
            },
            "risk_patterns": {
                "escalation_rate": (len(escalation_events) / len(rule_results) * 100) if rule_results else 0,
                "high_confidence_risks": len(high_risk_events)
            }
        }
        
        if include_details:
            report["detailed_results"] = [r.to_dict() for r in risk_results]
        
        return report
    
    def _generate_summary_report(self, rule_results: List[Any], time_period_hours: int) -> Dict[str, Any]:
        """Generate summary audit report."""
        total_rules = len(rule_results)
        passed_rules = sum(1 for r in rule_results if r.passed)
        
        # Group by rule type
        by_type = {}
        for result in rule_results:
            rule_type = result.rule_type.value
            if rule_type not in by_type:
                by_type[rule_type] = {"total": 0, "passed": 0, "failed": 0}
            
            by_type[rule_type]["total"] += 1
            if result.passed:
                by_type[rule_type]["passed"] += 1
            else:
                by_type[rule_type]["failed"] += 1
        
        # Calculate pass rates
        for type_data in by_type.values():
            type_data["pass_rate"] = (type_data["passed"] / type_data["total"] * 100) if type_data["total"] > 0 else 0
        
        return {
            "summary": {
                "total_rule_executions": total_rules,
                "overall_pass_rate": (passed_rules / total_rules * 100) if total_rules > 0 else 100,
                "time_period_hours": time_period_hours
            },
            "by_rule_type": by_type,
            "recommendations": self._generate_audit_recommendations(rule_results)
        }
    
    def _generate_audit_recommendations(self, rule_results: List[Any]) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []
        
        failed_results = [r for r in rule_results if not r.passed]
        total_results = len(rule_results)
        
        if not total_results:
            return ["No rule executions found in the audit period"]
        
        failure_rate = len(failed_results) / total_results
        
        if failure_rate > 0.2:  # More than 20% failure rate
            recommendations.append("High rule failure rate detected - review compliance processes")
        
        if any(r.action.value == "escalate" for r in failed_results):
            recommendations.append("Critical violations found - immediate review required")
        
        # Check for repeated violations
        rule_failure_counts = {}
        for result in failed_results:
            rule_name = result.rule_name
            rule_failure_counts[rule_name] = rule_failure_counts.get(rule_name, 0) + 1
        
        frequent_failures = [rule for rule, count in rule_failure_counts.items() if count >= 3]
        if frequent_failures:
            recommendations.append(f"Frequent failures in rules: {', '.join(frequent_failures[:3])}")
        
        if not recommendations:
            recommendations.append("Compliance status is good - continue current practices")
        
        return recommendations
    
    async def process_violation_alert_job(self, job: Job) -> Dict[str, Any]:
        """Process a compliance violation alert job.
        
        Args:
            job: Job containing violation alert data
            
        Returns:
            Alert processing result
        """
        logger.info(f"Processing violation alert job {job.job_id}")
        
        try:
            # Extract alert data
            violation_data = job.data.get("violation_data")
            alert_config = job.data.get("alert_config", {})
            
            if not violation_data:
                raise ValueError("No violation data provided for alert")
            
            # Determine alert severity
            severity = self._determine_alert_severity(violation_data)
            
            # Generate alert message
            alert_message = self._generate_alert_message(violation_data, severity)
            
            # Prepare alert result
            result = {
                "alert_id": f"alert_{job.job_id}",
                "severity": severity,
                "message": alert_message,
                "violation_details": violation_data,
                "timestamp": datetime.utcnow().isoformat(),
                "requires_immediate_action": severity in ["critical", "high"],
                "recommended_actions": self._get_recommended_actions(violation_data, severity)
            }
            
            logger.info(f"Violation alert job {job.job_id} completed: {severity} severity")
            return result
            
        except Exception as e:
            logger.error(f"Violation alert job {job.job_id} failed: {e}")
            raise
    
    def _determine_alert_severity(self, violation_data: Dict[str, Any]) -> str:
        """Determine alert severity based on violation data."""
        rule_action = violation_data.get("action", "")
        confidence = violation_data.get("confidence", 0.0)
        
        if rule_action == "escalate":
            return "critical"
        elif rule_action in ["deny", "require_review"] and confidence >= 0.9:
            return "high"
        elif rule_action == "redact":
            return "medium"
        else:
            return "low"
    
    def _generate_alert_message(self, violation_data: Dict[str, Any], severity: str) -> str:
        """Generate human-readable alert message."""
        rule_name = violation_data.get("rule_name", "Unknown Rule")
        document_id = violation_data.get("document_id", "Unknown Document")
        
        base_message = f"HIPAA Compliance Violation Detected\\n"
        base_message += f"Rule: {rule_name}\\n"
        base_message += f"Document: {document_id}\\n"
        base_message += f"Severity: {severity.upper()}"
        
        if violation_data.get("message"):
            base_message += f"\\nDetails: {violation_data['message']}"
        
        return base_message
    
    def _get_recommended_actions(self, violation_data: Dict[str, Any], severity: str) -> List[str]:
        """Get recommended actions for violation."""
        actions = []
        rule_action = violation_data.get("action", "")
        
        if severity == "critical":
            actions.append("Stop all processing immediately")
            actions.append("Notify compliance officer")
            actions.append("Initiate incident response procedure")
        elif severity == "high":
            actions.append("Review document immediately")
            actions.append("Apply required redactions")
            actions.append("Document remediation steps")
        elif rule_action == "redact":
            actions.append("Apply PHI redaction")
            actions.append("Verify redaction completeness")
        
        actions.append("Update audit log")
        return actions
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "compliance_checks": self.compliance_checks,
            "violations_found": self.violations_found,
            "violation_rate": (
                (self.violations_found / self.compliance_checks * 100)
                if self.compliance_checks > 0 else 0
            ),
            "worker_type": "compliance"
        }