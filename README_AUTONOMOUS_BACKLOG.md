# 🤖 Autonomous Backlog Assistant

An autonomous senior coding assistant that discovers, prioritizes, and executes backlog items using WSJF (Weighted Shortest Job First) methodology.

## 🎯 Overview

The Autonomous Backlog Assistant implements the complete workflow described in the AUTONOMOUS_BACKLOG.md specification:

1. **Discovers** tasks from multiple sources (backlog files, TODO comments, failing tests, security alerts)
2. **Prioritizes** using WSJF scoring with aging multipliers
3. **Executes** tasks using TDD micro-cycles with built-in safety checks
4. **Reports** progress and metrics automatically

## 🚀 Quick Start

### Basic Usage

```bash
# Check current backlog status
python3 src/cli_autonomous_backlog.py status

# See what would be executed (dry run)
python3 src/cli_autonomous_backlog.py start --dry-run

# Generate detailed report
python3 src/cli_autonomous_backlog.py report

# Start autonomous execution (with safety limits)
python3 src/cli_autonomous_backlog.py start
```

### Configuration

The assistant uses two main configuration files:

- `backlog.yml` - Structured backlog items with WSJF scores
- `.automation-scope.yaml` - Defines what the assistant can/cannot modify

## 📊 WSJF Scoring

Items are prioritized using the formula:
```
WSJF = (Business Value + Time Criticality + Risk Reduction) × Aging Multiplier / Effort
```

**Scale**: 1, 2, 3, 5, 8, 13 (Fibonacci)

**Current Top Priorities** (as of latest run):
1. **CR003**: Handle empty except blocks (WSJF: 31.00)
2. **DM001**: Security vulnerability scan (WSJF: 26.00)  
3. **CR004**: Add docstrings to CLI main() functions (WSJF: 18.00)
4. **CR002**: Implement empty SecurityError exception classes (WSJF: 17.00)
5. **DM002**: Dependency audit (WSJF: 16.00)

## 🔒 Safety Features

### Scope Control
- **Allowed Paths**: `./src/**`, `./tests/**`, `./docs/**`, `./config/**`
- **Restricted Paths**: `./.github/**`, `./pyproject.toml`, `./requirements.txt`
- **Forbidden Patterns**: `*.key`, `*.pem`, `*.env`, secrets

### Quality Gates
- All tests must pass
- Security scans must show 0 high/critical issues
- Code coverage must not decrease
- Documentation updated for API changes

### Risk Assessment
- High-risk items (>8 effort, auth/crypto/security) require human approval
- Automatic rollback preparation
- Maximum 10 iterations per execution cycle

## 📁 File Structure

```
src/
├── autonomous_backlog_assistant.py  # Core autonomous assistant logic
├── cli_autonomous_backlog.py        # Command-line interface
backlog.yml                          # Structured backlog items
.automation-scope.yaml               # Automation permissions
docs/status/                         # Generated reports
```

## 🔄 Execution Cycle

### Macro Loop
1. Sync repository and CI
2. Discover new tasks from all sources
3. Apply aging multipliers and WSJF scoring
4. Select highest priority READY task
5. Execute TDD micro-cycle
6. Update metrics and generate reports

### TDD Micro-Cycle (per task)
1. **RED**: Write failing test
2. **GREEN**: Implement minimum code to pass
3. **REFACTOR**: Clean up while keeping tests green
4. **Security check**: Validate against security checklist
5. **Documentation**: Update relevant docs
6. **CI gates**: Run all quality checks

## 📈 Metrics & Reporting

The assistant generates detailed reports including:
- Backlog size and status distribution
- Average WSJF scores and trends
- Security and critical item counts
- Completion rates and cycle times

Reports are automatically saved to `docs/status/autonomous_execution_report_YYYY-MM-DD.md`

## 🚨 Emergency Controls

### Stop Execution
```bash
# Interrupt with Ctrl+C - the assistant will complete the current task and stop gracefully
```

### Scope Violations
If the assistant encounters a path outside its scope, it will:
1. Log the violation
2. Request human approval via `APPROVE_SCOPE: <target>`
3. Continue with next task if no approval

### Blocked Tasks
Tasks are automatically marked as BLOCKED if:
- Quality gates fail
- Security issues detected
- Scope violations occur
- High-risk patterns detected

## 📋 Commands Reference

### Status Commands
```bash
# View backlog status
python3 src/cli_autonomous_backlog.py status

# Generate reports
python3 src/cli_autonomous_backlog.py report
python3 src/cli_autonomous_backlog.py report --format json
```

### Execution Commands
```bash
# Dry run (safe preview)
python3 src/cli_autonomous_backlog.py start --dry-run

# Full execution
python3 src/cli_autonomous_backlog.py start
```

### Scope Management
```bash
# Show automation scope
python3 src/cli_autonomous_backlog.py scope show

# Check if path is allowed
python3 src/cli_autonomous_backlog.py scope check ./src/new_file.py

# Approve restricted operation
python3 src/cli_autonomous_backlog.py scope approve target-id
```

### Configuration
```bash
# Initialize configuration
python3 src/cli_autonomous_backlog.py config init
```

## 🔧 Customization

### Adding New Task Sources
Extend `discover_backlog_items()` in `autonomous_backlog_assistant.py`:

```python
def _scan_custom_source(self) -> List[BacklogItem]:
    # Your custom task discovery logic
    pass
```

### Custom WSJF Weights
Modify the scoring in `BacklogItem.wsjf_score` property or adjust the aging multiplier logic.

### Quality Gates
Add new quality checks in `.automation-scope.yaml`:

```yaml
quality_gates:
  - name: "custom_check"
    command: "your-custom-command"
    required: true
```

## 🎯 Best Practices

1. **Run dry-run first** to understand what will be executed
2. **Monitor reports** in `docs/status/` for progress tracking  
3. **Review blocked items** regularly to unblock high-value work
4. **Calibrate WSJF scores** based on actual delivery outcomes
5. **Update automation scope** as project structure evolves

## 🤝 Contributing

The autonomous assistant follows the same development practices it enforces:
- TDD approach for all changes
- Security-first mindset
- Comprehensive documentation
- Automated quality gates

## 📞 Support

For issues or questions:
1. Check the generated reports in `docs/status/`
2. Review the automation scope configuration
3. Run with `--dry-run` to debug task selection
4. Examine logs for detailed execution information

---

**Remember**: The assistant is designed to be conservative and safe. It will err on the side of caution and request human input rather than risk making inappropriate changes.

*Last Updated: 2025-07-24*