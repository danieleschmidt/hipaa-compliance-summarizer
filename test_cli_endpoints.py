#!/usr/bin/env python3
"""Test CLI endpoints functionality."""

import subprocess
import tempfile
import os

def test_cli_endpoints():
    """Test all CLI entry points are working."""
    
    # Create a temporary test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Patient John Doe, DOB: 01/01/1980, SSN: 123-45-6789")
        test_file = f.name
    
    try:
        # Test hipaa-summarize
        result = subprocess.run([
            'hipaa-summarize', '--file', test_file, '--compliance-level', 'standard'
        ], capture_output=True, text=True, cwd='/root/repo')
        
        print("hipaa-summarize output:", result.stdout)
        print("hipaa-summarize errors:", result.stderr)
        assert result.returncode == 0, f"CLI failed with code {result.returncode}"
        
        # Test hipaa-compliance-report  
        result = subprocess.run([
            'hipaa-compliance-report', '--audit-period', '2024-Q1'
        ], capture_output=True, text=True, cwd='/root/repo')
        
        print("compliance-report output:", result.stdout) 
        print("compliance-report errors:", result.stderr)
        assert result.returncode == 0, f"CLI failed with code {result.returncode}"
        
        print("✅ All CLI endpoints working correctly")
        
    finally:
        os.unlink(test_file)

if __name__ == "__main__":
    test_cli_endpoints()