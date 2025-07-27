# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the HIPAA Compliance Summarizer project.

## ADR Format

Each ADR follows the format:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The situation that necessitates a decision
- **Decision**: The architecture decision made
- **Consequences**: The implications of the decision

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-python-technology-stack.md) | Python Technology Stack Selection | Accepted | 2025-07-27 |
| [ADR-002](002-cli-interface-design.md) | CLI Interface Design | Accepted | 2025-07-27 |
| [ADR-003](003-security-framework.md) | Security Framework Selection | Accepted | 2025-07-27 |
| [ADR-004](004-testing-strategy.md) | Testing Strategy and Framework | Accepted | 2025-07-27 |
| [ADR-005](005-ci-cd-pipeline.md) | CI/CD Pipeline Architecture | Accepted | 2025-07-27 |

## Guidelines

1. **Create new ADRs** for significant architectural decisions
2. **Number sequentially** starting from 001
3. **Use descriptive titles** that clearly indicate the decision area
4. **Include context** explaining why the decision was needed
5. **Document alternatives** that were considered
6. **Explain consequences** both positive and negative
7. **Update status** when decisions are superseded or deprecated