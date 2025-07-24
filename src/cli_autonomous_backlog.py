#!/usr/bin/env python3
"""
CLI interface for the Autonomous Backlog Assistant

Provides commands to:
- Start autonomous execution
- View backlog status  
- Generate reports
- Manage scope and configuration
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from autonomous_backlog_assistant import AutonomousBacklogAssistant, TaskStatus, RiskTier


def load_automation_scope(repo_root: Path) -> Dict[str, Any]:
    """Load automation scope configuration"""
    scope_file = repo_root / ".automation-scope.yaml"
    
    if not scope_file.exists():
        return {}
    
    with open(scope_file, 'r') as f:
        return yaml.safe_load(f)


def check_scope_permissions(path: str, scope_config: Dict[str, Any]) -> bool:
    """Check if path is within automation scope"""
    allowed_paths = scope_config.get('allowed_paths', [])
    restricted_paths = scope_config.get('restricted_paths', [])
    
    # Check if path is explicitly restricted
    for restricted in restricted_paths:
        if path.startswith(restricted.replace('**', '')):
            return False
    
    # Check if path is explicitly allowed
    for allowed in allowed_paths:
        if path.startswith(allowed.replace('**', '')):
            return True
    
    # Default to restricted if not explicitly allowed
    return False


def cmd_start(args) -> None:
    """Start autonomous execution"""
    print("🚀 Starting Autonomous Backlog Assistant")
    
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    
    # Load scope configuration
    scope_config = load_automation_scope(repo_root)
    
    if not scope_config:
        print("⚠️  No automation scope configuration found")
        print("   Creating default .automation-scope.yaml")
        # Default scope would be created here
    
    # Initialize assistant
    assistant = AutonomousBacklogAssistant(repo_root)
    
    if args.dry_run:
        print("📋 DRY RUN MODE - No changes will be made")
        # Discover and report what would be done
        items = assistant.discover_backlog_items()
        assistant.apply_aging_multiplier(items)
        sorted_items = assistant.sort_by_wsjf(items)
        
        print(f"\nDiscovered {len(items)} backlog items")
        print("\nTop 5 items by WSJF:")
        for i, item in enumerate(sorted_items[:5], 1):
            print(f"{i}. {item.id}: {item.title} (WSJF: {item.wsjf_score:.2f})")
    else:
        # Run autonomous execution
        assistant.execute_macro_loop()
    
    print("✅ Execution completed")


def cmd_status(args) -> None:
    """Show backlog status"""
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    assistant = AutonomousBacklogAssistant(repo_root)
    
    # Discover current backlog
    items = assistant.discover_backlog_items()
    assistant.apply_aging_multiplier(items)
    sorted_items = assistant.sort_by_wsjf(items)
    
    print(f"📊 Backlog Status")
    print(f"Total items: {len(items)}")
    
    # Count by status
    status_counts = {}
    for status in TaskStatus:
        count = len([item for item in items if item.status == status])
        if count > 0:
            status_counts[status.value] = count
    
    print("\nStatus distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Count by risk tier
    risk_counts = {}
    for risk in RiskTier:
        count = len([item for item in items if item.risk_tier == risk])
        if count > 0:
            risk_counts[risk.value] = count
    
    print("\nRisk distribution:")
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count}")
    
    # Show top WSJF items
    print(f"\nTop 10 items by WSJF:")
    for i, item in enumerate(sorted_items[:10], 1):
        status_icon = "🔥" if item.risk_tier == RiskTier.CRITICAL else "⚡" if item.risk_tier == RiskTier.HIGH else "📈"
        print(f"{i:2d}. {status_icon} {item.id}: {item.title}")
        print(f"     WSJF: {item.wsjf_score:.2f} | Status: {item.status.value} | Risk: {item.risk_tier.value}")


def cmd_report(args) -> None:
    """Generate detailed report"""
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    assistant = AutonomousBacklogAssistant(repo_root)
    
    # Generate metrics and report
    items = assistant.discover_backlog_items()
    assistant.backlog_items = items
    assistant.update_metrics()
    assistant.generate_status_report()
    
    print("📄 Report generated in docs/status/")
    
    if args.format == 'json':
        # Output JSON for programmatic use
        report_data = {
            'items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'wsjf_score': item.wsjf_score,
                    'status': item.status.value,
                    'risk_tier': item.risk_tier.value
                }
                for item in items
            ],
            'metrics': assistant.metrics
        }
        print(json.dumps(report_data, indent=2))


def cmd_scope(args) -> None:
    """Manage automation scope"""
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    scope_file = repo_root / ".automation-scope.yaml"
    
    if args.action == 'show':
        if scope_file.exists():
            with open(scope_file, 'r') as f:
                config = yaml.safe_load(f)
            print("📋 Current automation scope:")
            print(yaml.dump(config, default_flow_style=False))
        else:
            print("❌ No automation scope configuration found")
    
    elif args.action == 'check' and args.path:
        scope_config = load_automation_scope(repo_root)
        allowed = check_scope_permissions(args.path, scope_config)
        print(f"Path '{args.path}': {'✅ ALLOWED' if allowed else '❌ RESTRICTED'}")
    
    elif args.action == 'approve' and args.target:
        print(f"🔓 APPROVE_SCOPE: {args.target}")
        # This would be used by the autonomous assistant to gain approval
        # for specific restricted operations


def cmd_config(args) -> None:
    """Manage configuration"""
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    
    if args.action == 'init':
        # Initialize configuration files
        print("🔧 Initializing autonomous backlog assistant configuration...")
        
        # Check if files already exist
        backlog_file = repo_root / "backlog.yml"
        scope_file = repo_root / ".automation-scope.yaml"
        
        if backlog_file.exists() and scope_file.exists():
            print("✅ Configuration files already exist")
        else:
            print("📄 Configuration files would be created here")
            print("   (Implementation would create default configs)")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Backlog Assistant - Discover, Prioritize, Execute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start autonomous execution
  python -m cli_autonomous_backlog start
  
  # Dry run to see what would be done
  python -m cli_autonomous_backlog start --dry-run
  
  # Show current backlog status
  python -m cli_autonomous_backlog status
  
  # Generate detailed report
  python -m cli_autonomous_backlog report
  
  # Check automation scope
  python -m cli_autonomous_backlog scope show
  python -m cli_autonomous_backlog scope check ./src/new_file.py
        """
    )
    
    parser.add_argument(
        '--repo-root',
        type=str,
        help='Repository root path (default: current directory)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start autonomous execution')
    start_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    start_parser.set_defaults(func=cmd_start)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show backlog status')
    status_parser.set_defaults(func=cmd_status)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate detailed report')
    report_parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='Report format'
    )
    report_parser.set_defaults(func=cmd_report)
    
    # Scope command
    scope_parser = subparsers.add_parser('scope', help='Manage automation scope')
    scope_parser.add_argument(
        'action',
        choices=['show', 'check', 'approve'],
        help='Scope action'
    )
    scope_parser.add_argument('--path', help='Path to check')
    scope_parser.add_argument('--target', help='Target to approve')
    scope_parser.set_defaults(func=cmd_scope)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument(
        'action',
        choices=['init', 'validate'],
        help='Configuration action'
    )
    config_parser.set_defaults(func=cmd_config)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())