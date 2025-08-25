"""
Progressive Quality Gates - Autonomous SDLC Quality Enhancement System

This module implements progressive quality gates that automatically validate
and improve code quality throughout development cycles. It provides intelligent
quality assessment, automated remediation, and continuous improvement capabilities.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    print("PyYAML not available, using JSON fallback")
    yaml = None


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PASSED = "passed"
    FAILED = "failed" 
    WARNING = "warning"
    SKIPPED = "skipped"
    RUNNING = "running"


class QualityGateType(Enum):
    """Types of quality gates."""
    SYNTAX = "syntax"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float
    details: Dict[str, Any]
    duration: float
    timestamp: datetime
    remediation_suggestions: List[str]
    auto_fix_applied: bool = False


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    enabled: bool = True
    threshold: float = 0.8
    auto_fix: bool = True
    timeout: int = 300
    retry_count: int = 2
    severity: str = "error"
    custom_rules: Dict[str, Any] = None


class ProgressiveQualityGates:
    """
    Progressive Quality Gates system that provides intelligent quality validation
    and improvement capabilities for the HIPAA Compliance Summarizer project.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize progressive quality gates system."""
        self.config_path = config_path or "/root/repo/config/quality_gates.yml"
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.project_root = Path("/root/repo")
        self.results_history: List[QualityGateResult] = []
        
    def _load_config(self) -> Dict[str, QualityGateConfig]:
        """Load quality gate configurations."""
        default_config = {
            "syntax": QualityGateConfig(threshold=1.0, auto_fix=True),
            "testing": QualityGateConfig(threshold=0.85, auto_fix=False),
            "security": QualityGateConfig(threshold=0.9, auto_fix=True),
            "performance": QualityGateConfig(threshold=0.8, auto_fix=True),
            "compliance": QualityGateConfig(threshold=0.95, auto_fix=False),
            "documentation": QualityGateConfig(threshold=0.7, auto_fix=True),
            "dependency": QualityGateConfig(threshold=0.9, auto_fix=True),
        }
        
        if os.path.exists(self.config_path) and yaml is not None:
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                # Merge with defaults
                for gate_type, settings in config_data.get('gates', {}).items():
                    if gate_type in default_config:
                        for key, value in settings.items():
                            setattr(default_config[gate_type], key, value)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
        elif yaml is None:
            logging.info("YAML not available, using default configuration")
                
        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for quality gates."""
        logger = logging.getLogger("progressive_quality_gates")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    async def run_all_gates(self, target_path: Optional[str] = None) -> Dict[QualityGateType, QualityGateResult]:
        """Run all configured quality gates."""
        target_path = target_path or str(self.project_root)
        self.logger.info("🚀 Starting Progressive Quality Gates execution")
        
        results = {}
        
        # Run gates in parallel where possible
        syntax_task = self._run_syntax_gate(target_path)
        security_task = self._run_security_gate(target_path)
        dependency_task = self._run_dependency_gate(target_path)
        
        # Execute parallel tasks
        syntax_result, security_result, dependency_result = await asyncio.gather(
            syntax_task, security_task, dependency_task, return_exceptions=True
        )
        
        if not isinstance(syntax_result, Exception):
            results[QualityGateType.SYNTAX] = syntax_result
        if not isinstance(security_result, Exception):
            results[QualityGateType.SECURITY] = security_result  
        if not isinstance(dependency_result, Exception):
            results[QualityGateType.DEPENDENCY] = dependency_result
            
        # Run sequential gates that depend on syntax passing
        if results.get(QualityGateType.SYNTAX, QualityGateResult(
            QualityGateType.SYNTAX, QualityGateStatus.FAILED, 0, {}, 0, datetime.now(), []
        )).status == QualityGateStatus.PASSED:
            
            testing_result = await self._run_testing_gate(target_path)
            results[QualityGateType.TESTING] = testing_result
            
            performance_result = await self._run_performance_gate(target_path)
            results[QualityGateType.PERFORMANCE] = performance_result
            
        # Always run compliance and documentation gates
        compliance_result = await self._run_compliance_gate(target_path)
        results[QualityGateType.COMPLIANCE] = compliance_result
        
        documentation_result = await self._run_documentation_gate(target_path)
        results[QualityGateType.DOCUMENTATION] = documentation_result
        
        # Store results
        self.results_history.extend(results.values())
        
        # Generate summary report
        await self._generate_summary_report(results)
        
        return results

    async def _run_syntax_gate(self, target_path: str) -> QualityGateResult:
        """Run syntax quality gate using ruff."""
        start_time = time.time()
        self.logger.info("🔍 Running syntax quality gate")
        
        try:
            # Run ruff check
            result = subprocess.run(
                ["ruff", "check", target_path, "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=self.config["syntax"].timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return QualityGateResult(
                    gate_type=QualityGateType.SYNTAX,
                    status=QualityGateStatus.PASSED,
                    score=1.0,
                    details={"message": "All syntax checks passed"},
                    duration=duration,
                    timestamp=datetime.now(),
                    remediation_suggestions=[],
                )
            else:
                issues = []
                if result.stdout:
                    try:
                        issues = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        issues = [{"message": result.stdout}]
                
                # Attempt auto-fix if configured
                auto_fix_applied = False
                if self.config["syntax"].auto_fix and issues:
                    self.logger.info("🔧 Attempting automatic syntax fixes")
                    fix_result = subprocess.run(
                        ["ruff", "check", target_path, "--fix"],
                        capture_output=True,
                        text=True
                    )
                    auto_fix_applied = fix_result.returncode == 0
                
                score = max(0.0, 1.0 - (len(issues) / 100))  # Penalty for issues
                
                return QualityGateResult(
                    gate_type=QualityGateType.SYNTAX,
                    status=QualityGateStatus.WARNING if score > 0.5 else QualityGateStatus.FAILED,
                    score=score,
                    details={"issues": issues, "issue_count": len(issues)},
                    duration=duration,
                    timestamp=datetime.now(),
                    remediation_suggestions=[
                        "Run 'ruff check --fix' to automatically fix issues",
                        "Review and manually fix complex syntax issues",
                        "Consider updating ruff configuration if needed"
                    ],
                    auto_fix_applied=auto_fix_applied
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Syntax gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.SYNTAX,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Check ruff installation and configuration"],
            )

    async def _run_testing_gate(self, target_path: str) -> QualityGateResult:
        """Run testing quality gate using pytest."""
        start_time = time.time()
        self.logger.info("🧪 Running testing quality gate")
        
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=json", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config["testing"].timeout
            )
            
            duration = time.time() - start_time
            
            # Parse coverage results
            coverage_file = self.project_root / "coverage.json"
            coverage_score = 0.0
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    coverage_score = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
            
            # Determine status based on test results and coverage
            tests_passed = result.returncode == 0
            meets_threshold = coverage_score >= self.config["testing"].threshold
            
            if tests_passed and meets_threshold:
                status = QualityGateStatus.PASSED
            elif tests_passed:
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
                
            return QualityGateResult(
                gate_type=QualityGateType.TESTING,
                status=status,
                score=coverage_score if tests_passed else 0.0,
                details={
                    "coverage": coverage_score,
                    "tests_passed": tests_passed,
                    "output": result.stdout[-1000:],  # Last 1000 chars
                },
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=[
                    "Add more unit tests to increase coverage",
                    "Fix failing tests before proceeding",
                    "Consider integration and performance tests",
                ] if not meets_threshold or not tests_passed else [],
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Testing gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.TESTING,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Check pytest installation and test configuration"],
            )

    async def _run_security_gate(self, target_path: str) -> QualityGateResult:
        """Run security quality gate using bandit."""
        start_time = time.time()
        self.logger.info("🛡️ Running security quality gate")
        
        try:
            # Run bandit security scan
            result = subprocess.run(
                ["bandit", "-r", target_path, "-f", "json"],
                capture_output=True,
                text=True,
                timeout=self.config["security"].timeout
            )
            
            duration = time.time() - start_time
            
            if result.stdout:
                try:
                    security_data = json.loads(result.stdout)
                    issues = security_data.get("results", [])
                    metrics = security_data.get("metrics", {})
                    
                    # Calculate security score
                    total_lines = metrics.get("_totals", {}).get("loc", 1)
                    high_issues = len([i for i in issues if i.get("issue_severity") == "HIGH"])
                    medium_issues = len([i for i in issues if i.get("issue_severity") == "MEDIUM"])
                    
                    # Penalty-based scoring
                    penalty = (high_issues * 0.1) + (medium_issues * 0.05)
                    score = max(0.0, 1.0 - penalty)
                    
                    status = QualityGateStatus.PASSED if score >= self.config["security"].threshold else QualityGateStatus.WARNING
                    
                    return QualityGateResult(
                        gate_type=QualityGateType.SECURITY,
                        status=status,
                        score=score,
                        details={
                            "high_issues": high_issues,
                            "medium_issues": medium_issues,
                            "total_issues": len(issues),
                            "metrics": metrics
                        },
                        duration=duration,
                        timestamp=datetime.now(),
                        remediation_suggestions=[
                            "Review and fix high-severity security issues",
                            "Consider security best practices",
                            "Add security tests and validation"
                        ] if score < self.config["security"].threshold else [],
                    )
                    
                except json.JSONDecodeError:
                    pass
            
            # Fallback for non-JSON output
            score = 0.8 if result.returncode == 0 else 0.4
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                status=QualityGateStatus.PASSED if score >= self.config["security"].threshold else QualityGateStatus.WARNING,
                score=score,
                details={"output": result.stdout or result.stderr},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=[] if score >= self.config["security"].threshold else ["Review security scan output"],
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Security gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Check bandit installation"],
            )

    async def _run_dependency_gate(self, target_path: str) -> QualityGateResult:
        """Run dependency security gate using pip-audit."""
        start_time = time.time()
        self.logger.info("📦 Running dependency quality gate")
        
        try:
            # Run pip-audit for dependency vulnerabilities
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config["dependency"].timeout
            )
            
            duration = time.time() - start_time
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities = audit_data.get("vulnerabilities", [])
                    
                    high_vuln = len([v for v in vulnerabilities if v.get("fix_versions")])
                    total_vuln = len(vulnerabilities)
                    
                    # Score based on vulnerability severity
                    score = max(0.0, 1.0 - (total_vuln * 0.1))
                    
                    status = QualityGateStatus.PASSED if score >= self.config["dependency"].threshold else QualityGateStatus.WARNING
                    
                    return QualityGateResult(
                        gate_type=QualityGateType.DEPENDENCY,
                        status=status,
                        score=score,
                        details={
                            "total_vulnerabilities": total_vuln,
                            "fixable_vulnerabilities": high_vuln,
                            "vulnerabilities": vulnerabilities[:5]  # First 5 for brevity
                        },
                        duration=duration,
                        timestamp=datetime.now(),
                        remediation_suggestions=[
                            "Update vulnerable dependencies",
                            "Review dependency security advisories",
                            "Consider dependency pinning"
                        ] if total_vuln > 0 else [],
                    )
                    
                except json.JSONDecodeError:
                    pass
            
            # Fallback 
            score = 0.9 if result.returncode == 0 else 0.6
            return QualityGateResult(
                gate_type=QualityGateType.DEPENDENCY,
                status=QualityGateStatus.PASSED if score >= self.config["dependency"].threshold else QualityGateStatus.WARNING,
                score=score,
                details={"returncode": result.returncode, "output": result.stdout or result.stderr},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=[] if score >= self.config["dependency"].threshold else ["Review dependency audit output"],
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Dependency gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.DEPENDENCY,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Install pip-audit: pip install pip-audit"],
            )

    async def _run_performance_gate(self, target_path: str) -> QualityGateResult:
        """Run performance quality gate."""
        start_time = time.time()
        self.logger.info("⚡ Running performance quality gate")
        
        try:
            # Run basic performance checks
            perf_script = self.project_root / "performance_test_large.py"
            
            if perf_script.exists():
                result = subprocess.run(
                    ["python", str(perf_script)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=self.config["performance"].timeout
                )
                
                duration = time.time() - start_time
                
                # Basic performance scoring
                score = 0.8 if result.returncode == 0 else 0.4
                
                return QualityGateResult(
                    gate_type=QualityGateType.PERFORMANCE,
                    status=QualityGateStatus.PASSED if score >= self.config["performance"].threshold else QualityGateStatus.WARNING,
                    score=score,
                    details={
                        "test_output": result.stdout[-500:] if result.stdout else "",
                        "returncode": result.returncode
                    },
                    duration=duration,
                    timestamp=datetime.now(),
                    remediation_suggestions=[
                        "Optimize slow operations",
                        "Add caching where appropriate",
                        "Profile performance bottlenecks"
                    ] if score < self.config["performance"].threshold else [],
                )
            else:
                duration = time.time() - start_time
                return QualityGateResult(
                    gate_type=QualityGateType.PERFORMANCE,
                    status=QualityGateStatus.SKIPPED,
                    score=0.8,  # Assume good performance if no tests
                    details={"message": "No performance tests found"},
                    duration=duration,
                    timestamp=datetime.now(),
                    remediation_suggestions=["Create performance test suite"],
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Performance gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Check performance test setup"],
            )

    async def _run_compliance_gate(self, target_path: str) -> QualityGateResult:
        """Run compliance quality gate for HIPAA standards."""
        start_time = time.time()
        self.logger.info("📋 Running compliance quality gate")
        
        try:
            compliance_checks = {
                "config_files": self._check_compliance_config(),
                "documentation": self._check_compliance_docs(),
                "security_measures": self._check_security_compliance(),
                "audit_logging": self._check_audit_compliance(),
            }
            
            duration = time.time() - start_time
            
            # Calculate compliance score
            passed_checks = sum(1 for check in compliance_checks.values() if check)
            total_checks = len(compliance_checks)
            score = passed_checks / total_checks
            
            status = QualityGateStatus.PASSED if score >= self.config["compliance"].threshold else QualityGateStatus.WARNING
            
            return QualityGateResult(
                gate_type=QualityGateType.COMPLIANCE,
                status=status,
                score=score,
                details={
                    "checks": compliance_checks,
                    "passed": passed_checks,
                    "total": total_checks
                },
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=[
                    "Review HIPAA compliance documentation",
                    "Ensure proper configuration files",
                    "Implement missing security controls"
                ] if score < self.config["compliance"].threshold else [],
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Compliance gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.COMPLIANCE,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Review compliance requirements"],
            )

    def _check_compliance_config(self) -> bool:
        """Check if HIPAA compliance configuration exists."""
        config_file = self.project_root / "config" / "hipaa_config.yml"
        return config_file.exists()

    def _check_compliance_docs(self) -> bool:
        """Check if compliance documentation exists."""
        docs = [
            self.project_root / "docs" / "hipaa-compliance.md",
            self.project_root / "SECURITY.md",
            self.project_root / "docs" / "security.md"
        ]
        return any(doc.exists() for doc in docs)

    def _check_security_compliance(self) -> bool:
        """Check security compliance measures."""
        security_files = [
            self.project_root / "src" / "hipaa_compliance_summarizer" / "security.py",
            self.project_root / "src" / "hipaa_compliance_summarizer" / "advanced_security.py"
        ]
        return any(f.exists() for f in security_files)

    def _check_audit_compliance(self) -> bool:
        """Check audit logging compliance."""
        audit_files = [
            self.project_root / "src" / "hipaa_compliance_summarizer" / "audit_logger.py",
            self.project_root / "src" / "hipaa_compliance_summarizer" / "logging_framework.py"
        ]
        return any(f.exists() for f in audit_files)

    async def _run_documentation_gate(self, target_path: str) -> QualityGateResult:
        """Run documentation quality gate."""
        start_time = time.time()
        self.logger.info("📚 Running documentation quality gate")
        
        try:
            doc_checks = {
                "readme": (self.project_root / "README.md").exists(),
                "architecture": (self.project_root / "ARCHITECTURE.md").exists(),
                "api_docs": (self.project_root / "API_DOCUMENTATION.md").exists(),
                "contributing": (self.project_root / "CONTRIBUTING.md").exists(),
                "security": (self.project_root / "SECURITY.md").exists(),
            }
            
            # Check docstrings in Python files
            python_files = list(Path(target_path).rglob("*.py"))
            documented_files = 0
            for py_file in python_files[:10]:  # Sample first 10 files
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            documented_files += 1
                except:
                    continue
            
            docstring_ratio = documented_files / min(len(python_files), 10) if python_files else 0
            
            duration = time.time() - start_time
            
            # Calculate documentation score
            doc_score = sum(doc_checks.values()) / len(doc_checks)
            combined_score = (doc_score + docstring_ratio) / 2
            
            status = QualityGateStatus.PASSED if combined_score >= self.config["documentation"].threshold else QualityGateStatus.WARNING
            
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION,
                status=status,
                score=combined_score,
                details={
                    "doc_files": doc_checks,
                    "docstring_ratio": docstring_ratio,
                    "python_files_checked": min(len(python_files), 10)
                },
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=[
                    "Add missing documentation files",
                    "Improve docstring coverage",
                    "Update existing documentation"
                ] if combined_score < self.config["documentation"].threshold else [],
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Documentation gate failed: {e}")
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION,
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                timestamp=datetime.now(),
                remediation_suggestions=["Review documentation requirements"],
            )

    async def _generate_summary_report(self, results: Dict[QualityGateType, QualityGateResult]) -> None:
        """Generate comprehensive quality gates summary report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": sum(r.score for r in results.values()) / len(results) if results else 0,
            "gates_passed": sum(1 for r in results.values() if r.status == QualityGateStatus.PASSED),
            "gates_total": len(results),
            "results": {
                gate_type.value: {
                    "status": result.status.value,
                    "score": result.score,
                    "duration": result.duration,
                    "remediation_suggestions": result.remediation_suggestions,
                    "auto_fix_applied": result.auto_fix_applied
                }
                for gate_type, result in results.items()
            }
        }
        
        # Save report
        report_file = self.project_root / "quality_gates_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Quality Gates Report: {report['gates_passed']}/{report['gates_total']} passed, Overall Score: {report['overall_score']:.2f}")


def main():
    """Main entry point for progressive quality gates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates for HIPAA Compliance Summarizer")
    parser.add_argument("--config", help="Path to quality gates configuration file")
    parser.add_argument("--target", default="/root/repo", help="Target path for quality checks")
    
    args = parser.parse_args()
    
    async def run_gates():
        gates = ProgressiveQualityGates(config_path=args.config)
        results = await gates.run_all_gates(target_path=args.target)
        
        # Print summary
        passed = sum(1 for r in results.values() if r.status == QualityGateStatus.PASSED)
        total = len(results)
        overall_score = sum(r.score for r in results.values()) / total if total else 0
        
        print(f"\n{'='*60}")
        print(f"PROGRESSIVE QUALITY GATES SUMMARY")
        print(f"{'='*60}")
        print(f"Gates Passed: {passed}/{total}")
        print(f"Overall Score: {overall_score:.2%}")
        print(f"{'='*60}")
        
        return results
    
    return asyncio.run(run_gates())


if __name__ == "__main__":
    main()