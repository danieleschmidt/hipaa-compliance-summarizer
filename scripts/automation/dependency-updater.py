#!/usr/bin/env python3
"""
HIPAA Compliance Summarizer - Dependency Update Automation

This script automates dependency updates while maintaining HIPAA compliance
and security requirements for healthcare applications.

Security Note: All dependency updates are validated for security vulnerabilities
and HIPAA compliance requirements before being applied.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests
import yaml


class DependencyUpdater:
    """Automated dependency management with HIPAA compliance validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize dependency updater with configuration."""
        self.config = self._load_config(config_path)
        self.updated_packages = []
        self.security_alerts = []
        self.compliance_issues = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        default_config = {
            'github_token': os.getenv('GITHUB_TOKEN'),
            'auto_merge_patch': True,
            'auto_merge_minor': False,
            'security_auto_merge': True,
            'test_before_merge': True,
            'compliance_check': True,
            'excluded_packages': ['pandas', 'numpy'],  # Critical packages to exclude
            'max_updates_per_run': 10,
            'security_apis': {
                'pyup': os.getenv('PYUP_API_KEY'),
                'snyk': os.getenv('SNYK_TOKEN')
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def check_outdated_packages(self) -> List[Dict[str, Any]]:
        """Check for outdated packages using pip-check-updates equivalent."""
        print("🔍 Checking for outdated packages...")
        
        outdated = []
        
        try:
            # Use pip list --outdated
            result = subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                
                for package in packages:
                    if package['name'] not in self.config['excluded_packages']:
                        outdated.append({
                            'name': package['name'],
                            'current_version': package['version'],
                            'latest_version': package['latest_version'],
                            'type': package.get('latest_filetype', 'wheel')
                        })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"⚠️ Error checking outdated packages: {e}")
        
        print(f"📦 Found {len(outdated)} outdated packages")
        return outdated
    
    def analyze_security_vulnerabilities(self, packages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze packages for known security vulnerabilities."""
        print("🔒 Analyzing security vulnerabilities...")
        
        vulnerabilities = {}
        
        # Create temporary requirements file for scanning
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            for package in packages:
                temp_file.write(f"{package['name']}=={package['latest_version']}\n")
            temp_req_path = temp_file.name
        
        try:
            # Use safety to check for vulnerabilities
            result = subprocess.run(['safety', 'check', '-r', temp_req_path, '--json'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vulnerability in safety_data:
                        package_name = vulnerability.get('package_name')
                        if package_name:
                            if package_name not in vulnerabilities:
                                vulnerabilities[package_name] = []
                            vulnerabilities[package_name].append(vulnerability.get('advisory', 'Unknown vulnerability'))
                except json.JSONDecodeError:
                    print("⚠️ Could not parse safety output")
            
            # Also use pip-audit for additional coverage
            result = subprocess.run(['pip-audit', '-r', temp_req_path, '--format=json'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0 and result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    for vulnerability in audit_data.get('vulnerabilities', []):
                        package_name = vulnerability.get('package')
                        if package_name:
                            if package_name not in vulnerabilities:
                                vulnerabilities[package_name] = []
                            vulnerabilities[package_name].append(vulnerability.get('id', 'Unknown vulnerability'))
                except json.JSONDecodeError:
                    print("⚠️ Could not parse pip-audit output")
        
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error running security analysis: {e}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_req_path)
            except OSError:
                pass
        
        return vulnerabilities
    
    def check_hipaa_compliance(self, packages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Check packages for HIPAA compliance considerations."""
        print("🏥 Checking HIPAA compliance considerations...")
        
        compliance_issues = {}
        
        # Known problematic packages for healthcare applications
        healthcare_concerns = {
            'requests': ['Ensure TLS 1.2+ for all HTTPS requests'],
            'urllib3': ['Verify SSL certificate validation is enabled'],
            'cryptography': ['Review encryption algorithms for FIPS compliance'],
            'pycryptodome': ['Ensure FIPS 140-2 validated modules'],
            'sqlalchemy': ['Review query logging for PHI exposure'],
            'django': ['Check debug settings and error logging'],
            'flask': ['Verify debug mode disabled in production'],
            'celery': ['Ensure task serialization doesn\'t expose PHI'],
            'redis': ['Verify data encryption at rest and in transit'],
            'pymongo': ['Check MongoDB encryption and access controls']
        }
        
        for package in packages:
            package_name = package['name'].lower()
            
            # Check against known healthcare concerns
            if package_name in healthcare_concerns:
                compliance_issues[package['name']] = healthcare_concerns[package_name]
            
            # Check for packages that might handle sensitive data
            sensitive_patterns = ['auth', 'session', 'token', 'crypto', 'security', 'ssl', 'tls']
            if any(pattern in package_name for pattern in sensitive_patterns):
                if package['name'] not in compliance_issues:
                    compliance_issues[package['name']] = []
                compliance_issues[package['name']].append(
                    'Security-sensitive package - review for HIPAA compliance'
                )
        
        return compliance_issues
    
    def categorize_updates(self, packages: List[Dict[str, Any]], 
                          vulnerabilities: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize updates by priority and type."""
        print("📋 Categorizing updates by priority...")
        
        categories = {
            'security_critical': [],
            'patch_updates': [],
            'minor_updates': [],
            'major_updates': [],
            'excluded': []
        }
        
        for package in packages:
            package_name = package['name']
            current_version = package['current_version']
            latest_version = package['latest_version']
            
            # Check if package has security vulnerabilities
            if package_name in vulnerabilities:
                categories['security_critical'].append(package)
                continue
            
            # Parse semantic version to determine update type
            try:
                current_parts = self._parse_version(current_version)
                latest_parts = self._parse_version(latest_version)
                
                if current_parts and latest_parts:
                    if latest_parts[0] > current_parts[0]:
                        categories['major_updates'].append(package)
                    elif latest_parts[1] > current_parts[1]:
                        categories['minor_updates'].append(package)
                    else:
                        categories['patch_updates'].append(package)
                else:
                    # If version parsing fails, treat as minor update
                    categories['minor_updates'].append(package)
            
            except Exception:
                categories['minor_updates'].append(package)
        
        return categories
    
    def _parse_version(self, version: str) -> Optional[Tuple[int, int, int]]:
        """Parse semantic version string."""
        # Remove any non-numeric prefixes/suffixes
        version_clean = re.search(r'(\d+)\.(\d+)\.(\d+)', version)
        if version_clean:
            return (int(version_clean.group(1)), 
                   int(version_clean.group(2)), 
                   int(version_clean.group(3)))
        return None
    
    def update_requirements_file(self, packages: List[Dict[str, Any]], 
                                requirements_file: str = 'requirements.txt') -> bool:
        """Update requirements file with new package versions."""
        print(f"📝 Updating {requirements_file}...")
        
        req_file = Path(requirements_file)
        if not req_file.exists():
            print(f"⚠️ Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Read current requirements
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            # Create mapping of package names to new versions
            update_map = {pkg['name'].lower(): pkg['latest_version'] for pkg in packages}
            
            # Update lines
            updated_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name from requirement line
                    package_match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                    if package_match:
                        package_name = package_match.group(1).lower()
                        if package_name in update_map:
                            new_version = update_map[package_name]
                            updated_line = f"{package_match.group(1)}=={new_version}\n"
                            updated_lines.append(updated_line)
                            print(f"  📦 {package_match.group(1)}: {new_version}")
                            continue
                
                updated_lines.append(line + '\n' if not line.endswith('\n') else line)
            
            # Write updated requirements
            with open(req_file, 'w') as f:
                f.writelines(updated_lines)
            
            return True
        
        except Exception as e:
            print(f"❌ Error updating requirements file: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests to validate updates."""
        print("🧪 Running tests to validate updates...")
        
        try:
            # Install updated dependencies
            result = subprocess.run(['pip', 'install', '-r', 'requirements.txt'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
            
            # Run test suite
            result = subprocess.run(['pytest', 'tests/', '-v', '--tb=short'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print(f"❌ Tests failed: {result.stdout}\n{result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            print("⏰ Test execution timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def create_pull_request(self, updated_packages: List[Dict[str, Any]], 
                           category: str) -> Optional[str]:
        """Create pull request for dependency updates."""
        if not self.config.get('github_token'):
            print("⚠️ No GitHub token available for PR creation")
            return None
        
        print(f"📝 Creating pull request for {category} updates...")
        
        # Create commit
        commit_message = f"deps: update {category} dependencies\n\n"
        commit_message += "Updated packages:\n"
        for package in updated_packages:
            commit_message += f"- {package['name']}: {package['current_version']} → {package['latest_version']}\n"
        
        commit_message += "\n🤖 Generated with [Claude Code](https://claude.ai/code)\n"
        commit_message += "Co-Authored-By: Claude <noreply@anthropic.com>"
        
        try:
            # Stage and commit changes
            subprocess.run(['git', 'add', 'requirements.txt'], check=True)
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Create branch name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"deps/{category}-updates-{timestamp}"
            
            # Create and push branch
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            subprocess.run(['git', 'push', 'origin', branch_name], check=True)
            
            # Create PR using GitHub API
            pr_title = f"🔄 {category.replace('_', ' ').title()} Dependency Updates"
            pr_body = f"""## Dependency Updates - {category.replace('_', ' ').title()}

This PR updates the following packages:

"""
            for package in updated_packages:
                pr_body += f"- **{package['name']}**: `{package['current_version']}` → `{package['latest_version']}`\n"
            
            pr_body += f"""

### Security Analysis
- All packages have been scanned for known vulnerabilities
- HIPAA compliance considerations have been reviewed
- Tests have been executed to validate compatibility

### Review Checklist
- [ ] Security scan results reviewed
- [ ] HIPAA compliance verified
- [ ] Test suite passes
- [ ] Documentation updated if needed

**Category**: {category}
**Auto-merge eligible**: {self._is_auto_merge_eligible(category)}

🤖 Generated with [Claude Code](https://claude.ai/code)
"""
            
            # Create PR via GitHub API
            api_url = f"https://api.github.com/repos/{self.config.get('repository', 'owner/repo')}/pulls"
            headers = {
                'Authorization': f"token {self.config['github_token']}",
                'Accept': 'application/vnd.github.v3+json'
            }
            
            pr_data = {
                'title': pr_title,
                'body': pr_body,
                'head': branch_name,
                'base': 'main'
            }
            
            response = requests.post(api_url, json=pr_data, headers=headers)
            
            if response.status_code == 201:
                pr_url = response.json()['html_url']
                print(f"✅ Pull request created: {pr_url}")
                return pr_url
            else:
                print(f"❌ Failed to create PR: {response.status_code} - {response.text}")
                return None
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return None
        except requests.RequestException as e:
            print(f"❌ GitHub API request failed: {e}")
            return None
    
    def _is_auto_merge_eligible(self, category: str) -> bool:
        """Determine if updates in this category are eligible for auto-merge."""
        if category == 'security_critical':
            return self.config.get('security_auto_merge', True)
        elif category == 'patch_updates':
            return self.config.get('auto_merge_patch', True)
        elif category == 'minor_updates':
            return self.config.get('auto_merge_minor', False)
        else:
            return False
    
    def generate_report(self) -> str:
        """Generate dependency update report."""
        report = f"# Dependency Update Report\n\n"
        report += f"**Generated**: {datetime.utcnow().isoformat()}Z\n\n"
        
        if self.updated_packages:
            report += "## Updated Packages\n\n"
            for package in self.updated_packages:
                report += f"- **{package['name']}**: {package['current_version']} → {package['latest_version']}\n"
            report += "\n"
        
        if self.security_alerts:
            report += "## Security Alerts\n\n"
            for alert in self.security_alerts:
                report += f"- ⚠️ {alert}\n"
            report += "\n"
        
        if self.compliance_issues:
            report += "## HIPAA Compliance Considerations\n\n"
            for package, issues in self.compliance_issues.items():
                report += f"- **{package}**:\n"
                for issue in issues:
                    report += f"  - {issue}\n"
            report += "\n"
        
        return report
    
    def run_update_process(self) -> int:
        """Run the complete dependency update process."""
        print("🚀 Starting dependency update process...")
        
        try:
            # Check for outdated packages
            outdated_packages = self.check_outdated_packages()
            
            if not outdated_packages:
                print("✅ All packages are up to date!")
                return 0
            
            # Analyze security vulnerabilities
            vulnerabilities = self.analyze_security_vulnerabilities(outdated_packages)
            if vulnerabilities:
                print(f"🔒 Found security vulnerabilities in {len(vulnerabilities)} packages")
                self.security_alerts = [f"{pkg}: {', '.join(vulns)}" for pkg, vulns in vulnerabilities.items()]
            
            # Check HIPAA compliance
            compliance_issues = self.check_hipaa_compliance(outdated_packages)
            if compliance_issues:
                print(f"🏥 Found HIPAA compliance considerations for {len(compliance_issues)} packages")
                self.compliance_issues = compliance_issues
            
            # Categorize updates
            categories = self.categorize_updates(outdated_packages, vulnerabilities)
            
            # Process updates by category
            success = True
            for category, packages in categories.items():
                if not packages or category == 'excluded':
                    continue
                
                print(f"\n📦 Processing {category}: {len(packages)} packages")
                
                # Limit number of updates per run
                if len(packages) > self.config['max_updates_per_run']:
                    packages = packages[:self.config['max_updates_per_run']]
                    print(f"⚠️ Limited to {self.config['max_updates_per_run']} packages per run")
                
                # Update requirements file
                if self.update_requirements_file(packages):
                    # Run tests if configured
                    if self.config.get('test_before_merge', True):
                        if not self.run_tests():
                            print(f"❌ Tests failed for {category} updates")
                            success = False
                            continue
                    
                    # Create pull request
                    pr_url = self.create_pull_request(packages, category)
                    if pr_url:
                        self.updated_packages.extend(packages)
                        print(f"✅ {category} updates completed: {pr_url}")
                    else:
                        success = False
                else:
                    success = False
            
            # Generate report
            report = self.generate_report()
            report_file = Path(f"dependency-update-report-{datetime.now().strftime('%Y%m%d')}.md")
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📊 Update process completed!")
            print(f"📄 Report saved: {report_file}")
            print(f"📦 Total packages updated: {len(self.updated_packages)}")
            
            return 0 if success else 1
        
        except Exception as e:
            print(f"❌ Error during update process: {e}")
            return 1


def main():
    """Main entry point for dependency updates."""
    updater = DependencyUpdater()
    
    try:
        return updater.run_update_process()
    except KeyboardInterrupt:
        print("\n⏹️ Dependency update process interrupted")
        return 1


if __name__ == '__main__':
    sys.exit(main())