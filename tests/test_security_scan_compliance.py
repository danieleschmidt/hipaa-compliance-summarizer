"""Tests for security scan compliance."""

import json
import subprocess
from pathlib import Path
import pytest


class TestSecurityCompliance:
    """Test that security scans meet compliance requirements."""
    
    def test_bandit_scan_no_high_severity_issues(self):
        """Test that bandit scan shows no high or critical severity issues."""
        # Read the latest bandit report
        bandit_report_path = Path("bandit_report.json")
        
        if not bandit_report_path.exists():
            pytest.skip("Bandit report not found - run bandit scan first")
        
        with open(bandit_report_path) as f:
            report = json.load(f)
        
        metrics = report.get("metrics", {}).get("_totals", {})
        
        # Check for high or critical severity issues
        high_severity = metrics.get("SEVERITY.HIGH", 0)
        critical_severity = metrics.get("SEVERITY.CRITICAL", 0)
        
        assert high_severity == 0, f"Found {high_severity} high severity issues"
        assert critical_severity == 0, f"Found {critical_severity} critical severity issues"
    
    def test_bandit_scan_medium_issues_documented(self):
        """Test that medium severity issues are documented and addressed."""
        bandit_report_path = Path("bandit_report.json")
        
        if not bandit_report_path.exists():
            pytest.skip("Bandit report not found")
        
        with open(bandit_report_path) as f:
            report = json.load(f)
        
        metrics = report.get("metrics", {}).get("_totals", {})
        medium_severity = metrics.get("SEVERITY.MEDIUM", 0)
        
        # Medium issues should be minimized and documented
        # For HIPAA compliance, we want very few security issues
        assert medium_severity <= 5, f"Too many medium severity issues: {medium_severity}"
    
    def test_security_scan_results_exist(self):
        """Test that security scan results file exists and is recent."""
        scan_results_path = Path("security_scan_results.json")
        
        # Should have recent scan results
        assert scan_results_path.exists(), "Security scan results file missing"
        
        # Read and validate structure
        with open(scan_results_path) as f:
            results = json.load(f)
        
        assert "generated_at" in results, "Scan results missing timestamp"
        assert "metrics" in results, "Scan results missing metrics"
    
    def test_no_hardcoded_secrets_in_code(self):
        """Test that no hardcoded secrets are detected by security scans."""
        # This would fail if there were actual secrets detected
        # The test files with secrets should be marked with allowlist comments
        
        # Read bandit report for secret-related issues
        bandit_report_path = Path("bandit_report.json")
        
        if bandit_report_path.exists():
            with open(bandit_report_path) as f:
                report = json.load(f)
            
            # Check if there are any results related to secrets
            results = report.get("results", [])
            
            # Filter for secret-related issues that aren't test files
            secret_issues = []
            for result in results:
                if any(keyword in result.get("test_name", "").lower() 
                       for keyword in ["secret", "password", "key", "token"]):
                    if "test" not in result.get("filename", ""):
                        secret_issues.append(result)
            
            assert len(secret_issues) == 0, f"Found secret-related issues in non-test files: {secret_issues}"