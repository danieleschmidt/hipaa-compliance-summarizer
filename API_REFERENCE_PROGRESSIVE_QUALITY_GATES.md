# Progressive Quality Gates API Reference

## Overview

The Progressive Quality Gates system provides a comprehensive API for autonomous software quality assurance. This reference documents all public interfaces, classes, and methods available for integration and customization.

## Core API Classes

### ProgressiveQualityGates

Main orchestrator class for quality gate execution.

```python
from hipaa_compliance_summarizer.progressive_quality_gates import ProgressiveQualityGates

class ProgressiveQualityGates:
    """Main class for progressive quality gate execution."""
```

#### Constructor

```python
def __init__(self, config_path: Optional[str] = None):
    """
    Initialize progressive quality gates system.
    
    Args:
        config_path: Path to quality gates configuration file.
                    Defaults to "/root/repo/config/quality_gates.yml"
    """
```

#### Methods

##### `run_all_gates(target_path: str) -> Dict[QualityGateType, QualityGateResult]`

Execute all configured quality gates.

```python
async def run_all_gates(
    self, 
    target_path: Optional[str] = None
) -> Dict[QualityGateType, QualityGateResult]:
    """
    Run all configured quality gates.
    
    Args:
        target_path: Path to analyze. Defaults to project root.
        
    Returns:
        Dictionary mapping gate types to their results.
        
    Example:
        gates = ProgressiveQualityGates()
        results = await gates.run_all_gates("/path/to/project")
        
        for gate_type, result in results.items():
            print(f"{gate_type.value}: {result.status.value}")
    """
```

##### Individual Gate Methods

```python
async def _run_syntax_gate(self, target_path: str) -> QualityGateResult:
    """Run syntax quality gate using ruff."""

async def _run_testing_gate(self, target_path: str) -> QualityGateResult:
    """Run testing quality gate using pytest."""

async def _run_security_gate(self, target_path: str) -> QualityGateResult:
    """Run security quality gate using bandit."""

async def _run_performance_gate(self, target_path: str) -> QualityGateResult:
    """Run performance quality gate."""

async def _run_compliance_gate(self, target_path: str) -> QualityGateResult:
    """Run compliance quality gate for HIPAA standards."""

async def _run_documentation_gate(self, target_path: str) -> QualityGateResult:
    """Run documentation quality gate."""

async def _run_dependency_gate(self, target_path: str) -> QualityGateResult:
    """Run dependency security gate using pip-audit."""
```

### Quality Gate Data Models

#### QualityGateType

Enumeration of available quality gate types.

```python
from hipaa_compliance_summarizer.progressive_quality_gates import QualityGateType

class QualityGateType(Enum):
    """Types of quality gates."""
    SYNTAX = "syntax"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
```

#### QualityGateStatus

Enumeration of quality gate execution statuses.

```python
class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PASSED = "passed"
    FAILED = "failed" 
    WARNING = "warning"
    SKIPPED = "skipped"
    RUNNING = "running"
```

#### QualityGateResult

Result object for quality gate execution.

```python
@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float                    # 0.0 - 1.0
    details: Dict[str, Any]
    duration: float                 # Seconds
    timestamp: datetime
    remediation_suggestions: List[str]
    auto_fix_applied: bool = False
    
    # Example usage:
    if result.status == QualityGateStatus.PASSED:
        print(f"Gate passed with score: {result.score}")
    else:
        print(f"Remediation needed: {result.remediation_suggestions}")
```

#### QualityGateConfig

Configuration object for quality gate settings.

```python
@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    enabled: bool = True
    threshold: float = 0.8          # Pass/fail threshold
    auto_fix: bool = True           # Enable automatic fixes
    timeout: int = 300              # Timeout in seconds
    retry_count: int = 2            # Number of retries
    severity: str = "error"         # error, warning, info
    custom_rules: Dict[str, Any] = None
```

## Resilient Quality System API

### ResilientQualitySystem

Main class for fault-tolerant quality operations.

```python
from hipaa_compliance_summarizer.resilient_quality_system import ResilientQualitySystem

class ResilientQualitySystem:
    """Comprehensive resilient quality system."""
```

#### Methods

##### `create_circuit_breaker(name: str, config: CircuitBreakerConfig) -> CircuitBreaker`

Create and register a circuit breaker.

```python
def create_circuit_breaker(
    self, 
    name: str, 
    config: CircuitBreakerConfig
) -> CircuitBreaker:
    """
    Create and register a circuit breaker.
    
    Args:
        name: Unique name for the circuit breaker
        config: Circuit breaker configuration
        
    Returns:
        Configured CircuitBreaker instance
        
    Example:
        system = ResilientQualitySystem()
        breaker = system.create_circuit_breaker(
            "security_scan",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
        )
    """
```

##### `resilient_operation(name: str, **configs) -> Callable`

Decorator for applying resilience patterns.

```python
def resilient_operation(
    self,
    operation_name: str,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    bulkhead_config: Optional[BulkheadConfig] = None,
    fallback_config: Optional[FallbackConfig] = None,
) -> Callable:
    """
    Decorator that applies multiple resilience patterns.
    
    Example:
        @system.resilient_operation(
            "critical_operation",
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
            retry_config=RetryConfig(max_attempts=3),
            fallback_config=FallbackConfig(fallback_value={"status": "degraded"})
        )
        async def critical_operation():
            # Your operation implementation
            pass
    """
```

### Resilience Configuration Classes

#### CircuitBreakerConfig

```python
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before attempting reset
    expected_recovery_time: float = 30.0 # Expected time to recover
    monitor_requests: int = 10          # Requests to monitor in half-open
```

#### RetryConfig

```python
@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    initial_delay: float = 1.0          # Initial retry delay
    max_delay: float = 60.0             # Maximum retry delay
    backoff_factor: float = 2.0         # Backoff multiplier
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: tuple = (Exception,)
```

#### RetryStrategy

```python
class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI = "fibonacci"
```

### Convenience Decorators

#### `resilient_quality_gate`

Global decorator for quality gate resilience.

```python
from hipaa_compliance_summarizer.resilient_quality_system import resilient_quality_gate

@resilient_quality_gate(
    "syntax_validation",
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
    retry_config=RetryConfig(max_attempts=2),
    fallback_config=FallbackConfig(fallback_value={"status": "unknown", "score": 0.5})
)
async def validate_syntax():
    """Quality gate with full resilience protection."""
    # Implementation here
    pass
```

## Adaptive Learning Engine API

### AdaptiveLearningEngine

Main class for ML-driven quality optimization.

```python
from hipaa_compliance_summarizer.adaptive_learning_engine import AdaptiveLearningEngine

class AdaptiveLearningEngine:
    """Main adaptive learning engine for quality optimization."""
```

#### Constructor

```python
def __init__(self, model_path: Optional[Path] = None):
    """
    Initialize adaptive learning engine.
    
    Args:
        model_path: Path to saved model. Defaults to adaptive_quality_model.pkl
    """
```

#### Methods

##### `learn_from_execution(gate_type: str, execution_result: Dict, context: Dict)`

Learn from quality gate execution results.

```python
def learn_from_execution(
    self, 
    gate_type: str, 
    execution_result: Dict[str, Any], 
    context: Dict[str, Any]
):
    """
    Learn from a quality gate execution result.
    
    Args:
        gate_type: Type of quality gate (e.g., "security", "testing")
        execution_result: Result dictionary with score, duration, status
        context: Execution context (project size, complexity, etc.)
        
    Example:
        engine = AdaptiveLearningEngine()
        engine.learn_from_execution(
            gate_type="security",
            execution_result={
                "score": 0.92,
                "duration": 45.2,
                "status": "passed",
                "remediation_applied": True
            },
            context={
                "project_size": "large",
                "code_complexity": 0.7,
                "change_velocity": "medium"
            }
        )
    """
```

##### `predict_and_optimize(gate_type: str, current_config: Dict, context: Dict)`

Predict quality outcome and optimize configuration.

```python
def predict_and_optimize(
    self, 
    gate_type: str, 
    current_config: Dict[str, Any], 
    context: Dict[str, Any]
) -> Tuple[PredictionResult, Dict[str, Any]]:
    """
    Predict quality outcome and optimize configuration.
    
    Args:
        gate_type: Type of quality gate
        current_config: Current gate configuration
        context: Execution context
        
    Returns:
        Tuple of (prediction_result, optimized_config)
        
    Example:
        prediction, optimized_config = engine.predict_and_optimize(
            gate_type="performance",
            current_config={"timeout": 300, "parallel_jobs": 4},
            context={"project_size": "medium", "load": "high"}
        )
        
        print(f"Predicted score: {prediction.predicted_score}")
        print(f"Confidence: {prediction.confidence}")
        print(f"Risk level: {prediction.risk_level}")
        print(f"Optimized timeout: {optimized_config['timeout']}")
    """
```

### Learning Data Models

#### PredictionResult

```python
@dataclass
class PredictionResult:
    """Result of quality prediction."""
    predicted_score: float              # 0.0 - 1.0
    confidence: float                   # 0.0 - 1.0
    risk_level: str                     # low, medium, high
    recommendations: List[str]
    optimal_parameters: Dict[str, Any]
```

#### QualityDataPoint

```python
@dataclass
class QualityDataPoint:
    """A single quality measurement data point."""
    timestamp: datetime
    gate_type: str
    score: float
    duration: float
    status: str
    context: Dict[str, Any] = field(default_factory=dict)
    remediation_applied: bool = False
```

### Learning Decorators

#### `learning_enabled_quality_gate`

Enable adaptive learning for a quality gate.

```python
from hipaa_compliance_summarizer.adaptive_learning_engine import learning_enabled_quality_gate

@learning_enabled_quality_gate("custom_quality_check")
async def custom_quality_check(context: Dict[str, Any]):
    """
    Quality gate with adaptive learning enabled.
    
    The decorator will:
    1. Predict optimal configuration
    2. Execute the gate
    3. Learn from the results
    4. Optimize future executions
    """
    # Implementation here
    return {"score": 0.95, "duration": 30.0, "status": "passed"}
```

## Autonomous Quality Orchestrator API

### AutonomousQualityOrchestrator

Main orchestrator for autonomous quality lifecycle management.

```python
from hipaa_compliance_summarizer.autonomous_quality_orchestrator import AutonomousQualityOrchestrator

class AutonomousQualityOrchestrator:
    """Autonomous quality orchestration system."""
```

#### Methods

##### `orchestrate_quality_lifecycle() -> Dict[str, Any]`

Execute the complete autonomous quality lifecycle.

```python
async def orchestrate_quality_lifecycle(self) -> Dict[str, Any]:
    """
    Orchestrate the complete autonomous quality lifecycle.
    
    Phases:
    1. Initialization and intelligence gathering
    2. Intelligent analysis
    3. Autonomous execution
    4. Intelligent validation
    5. Autonomous remediation (if needed)
    6. Continuous optimization
    7. Comprehensive reporting
    
    Returns:
        Comprehensive orchestration result with metrics and insights
        
    Example:
        orchestrator = AutonomousQualityOrchestrator()
        result = await orchestrator.orchestrate_quality_lifecycle()
        
        print(f"Status: {result['status']}")
        print(f"Phases completed: {len(result['phases_completed'])}")
        print(f"Overall quality score: {result['final_metrics']['overall_score']}")
    """
```

## Intelligent Performance Optimizer API

### IntelligentPerformanceOptimizer

Main class for intelligent performance optimization.

```python
from hipaa_compliance_summarizer.intelligent_performance_optimizer import IntelligentPerformanceOptimizer

class IntelligentPerformanceOptimizer:
    """Intelligent performance optimization system."""
```

#### Methods

##### `start_intelligent_optimization(optimization_interval: float = 300.0)`

Start intelligent performance optimization.

```python
async def start_intelligent_optimization(
    self, 
    optimization_interval: float = 300.0
):
    """
    Start intelligent performance optimization system.
    
    Args:
        optimization_interval: Optimization cycle interval in seconds
        
    Features:
    - Real-time performance monitoring
    - Automatic bottleneck detection
    - ML-driven optimization recommendations
    - Automated performance tuning
    """
```

##### `performance_optimize(operation_name: str)`

Decorator for performance optimization.

```python
def performance_optimize(self, operation_name: str):
    """
    Decorator for automatic performance optimization.
    
    Example:
        optimizer = IntelligentPerformanceOptimizer()
        
        @optimizer.performance_optimize("quality_gate_execution")
        async def run_quality_gates():
            # Automatically optimized based on performance history
            pass
    """
```

### Performance Data Models

#### PerformanceMetric

```python
@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    operation_name: str
    duration: float                     # Seconds
    cpu_usage: float                    # Percentage
    memory_usage: float                 # Percentage
    io_operations: int
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
```

#### OptimizationRecommendation

```python
@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str                       # cpu, memory, io, algorithm
    priority: str                       # high, medium, low
    description: str
    implementation: str
    expected_improvement: float         # 0.0 - 1.0
    estimated_effort: str              # hours, days, weeks
    code_location: Optional[str] = None
```

### Global Performance Decorator

#### `performance_optimized`

Global decorator for performance optimization.

```python
from hipaa_compliance_summarizer.intelligent_performance_optimizer import performance_optimized

@performance_optimized("data_processing")
async def process_large_dataset():
    """
    Function with intelligent performance optimization:
    - Automatic profiling
    - Resource monitoring
    - Performance metrics collection
    - ML-driven optimization suggestions
    """
    # Implementation here
    pass
```

## Global Compliance Framework API

### GlobalComplianceFramework

Main class for multi-jurisdiction compliance management.

```python
from hipaa_compliance_summarizer.global_compliance_framework import GlobalComplianceFramework

class GlobalComplianceFramework:
    """Comprehensive global compliance framework."""
```

#### Methods

##### `assess_compliance(data_types, processing_activities, jurisdictions, safeguards)`

Conduct comprehensive compliance assessment.

```python
def assess_compliance(
    self, 
    data_types: List[str], 
    processing_activities: List[str],
    jurisdictions: List[ComplianceJurisdiction],
    current_safeguards: Dict[str, bool]
) -> List[ComplianceAssessment]:
    """
    Conduct comprehensive compliance assessment.
    
    Args:
        data_types: Types of data being processed
        processing_activities: List of processing activities
        jurisdictions: Jurisdictions to assess
        current_safeguards: Current security/privacy safeguards
        
    Returns:
        List of compliance assessments for each jurisdiction
        
    Example:
        framework = GlobalComplianceFramework()
        assessments = framework.assess_compliance(
            data_types=["health_data", "biometric_data"],
            processing_activities=["collection", "analysis", "storage"],
            jurisdictions=[ComplianceJurisdiction.UNITED_STATES, 
                          ComplianceJurisdiction.EUROPEAN_UNION],
            current_safeguards={
                "data_encryption": True,
                "audit_logging": True,
                "access_controls": True,
                "multi_factor_authentication": False
            }
        )
    """
```

### Compliance Data Models

#### ComplianceJurisdiction

```python
class ComplianceJurisdiction(Enum):
    """Global compliance jurisdictions."""
    UNITED_STATES = "US"
    EUROPEAN_UNION = "EU"
    UNITED_KINGDOM = "UK"
    CANADA = "CA"
    SINGAPORE = "SG"
    AUSTRALIA = "AU"
    JAPAN = "JP"
    SOUTH_KOREA = "KR"
    BRAZIL = "BR"
    INDIA = "IN"
```

#### ComplianceStandard

```python
class ComplianceStandard(Enum):
    """International compliance standards."""
    HIPAA = "hipaa"                    # US
    GDPR = "gdpr"                      # EU
    PDPA_SG = "pdpa_sg"                # Singapore
    PIPEDA = "pipeda"                  # Canada
    DPA_UK = "dpa_uk"                  # UK
    PRIVACY_ACT = "privacy_act_au"      # Australia
    APPI = "appi"                      # Japan
    PIPA_KR = "pipa_kr"                # South Korea
    LGPD = "lgpd"                      # Brazil
    DPDPA = "dpdpa"                    # India
```

#### ComplianceAssessment

```python
@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    jurisdiction: ComplianceJurisdiction
    standards_evaluated: List[ComplianceStandard]
    compliance_score: float             # 0.0 - 1.0
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str                     # low, medium, high, critical
    assessment_date: datetime
    next_review_date: datetime
```

### Convenience Functions

#### `assess_multi_jurisdiction_compliance`

Simplified compliance assessment function.

```python
from hipaa_compliance_summarizer.global_compliance_framework import assess_multi_jurisdiction_compliance

assessments = assess_multi_jurisdiction_compliance(
    data_types=["health_data", "direct_identifiers"],
    processing_activities=["collection", "storage", "analysis"],
    target_jurisdictions=["US", "EU", "SG"],
    current_safeguards={
        "data_encryption": True,
        "audit_logging": True,
        "access_controls": True,
        "multi_factor_authentication": False
    }
)
```

## Multi-Region Deployment API

### MultiRegionDeploymentOrchestrator

Main class for global deployment management.

```python
from hipaa_compliance_summarizer.multi_region_deployment import MultiRegionDeploymentOrchestrator

class MultiRegionDeploymentOrchestrator:
    """Multi-region deployment orchestrator."""
```

#### Methods

##### `deploy_globally(services, version, strategy)`

Execute global deployment across all regions.

```python
async def deploy_globally(
    self, 
    services: List[str], 
    version: str, 
    deployment_strategy: str = "rolling"
) -> Dict[str, Any]:
    """
    Execute global deployment across all configured regions.
    
    Args:
        services: List of service names to deploy
        version: Version to deploy
        deployment_strategy: Deployment strategy (rolling, blue_green, canary)
        
    Returns:
        Comprehensive deployment result
        
    Example:
        orchestrator = MultiRegionDeploymentOrchestrator()
        result = await orchestrator.deploy_globally(
            services=["api", "processor", "validator"],
            version="v1.2.3",
            deployment_strategy="rolling"
        )
    """
```

### Deployment Configuration

#### RegionConfig

```python
@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    cloud_provider: CloudProvider
    jurisdiction: ComplianceJurisdiction
    data_residency_required: bool
    primary_region: bool = False
    disaster_recovery_target: bool = True
    latency_target_ms: int = 100
    availability_target: float = 99.9
    compliance_requirements: List[str] = field(default_factory=list)
```

#### DeploymentRegion

```python
class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    # ... additional regions
```

### Convenience Functions

#### `configure_global_regions`

Configure global deployment regions.

```python
from hipaa_compliance_summarizer.multi_region_deployment import configure_global_regions

configure_global_regions([
    ("us-east-1", "aws", "US", False),
    ("eu-west-1", "aws", "EU", True),
    ("ap-southeast-1", "aws", "SG", True)
])
```

#### `deploy_globally_with_compliance`

Global deployment with compliance validation.

```python
from hipaa_compliance_summarizer.multi_region_deployment import deploy_globally_with_compliance

result = await deploy_globally_with_compliance(
    services=["api", "processor", "validator"],
    version="v1.2.3",
    strategy="rolling"
)
```

## Usage Examples

### Complete Integration Example

```python
import asyncio
from hipaa_compliance_summarizer.progressive_quality_gates import ProgressiveQualityGates
from hipaa_compliance_summarizer.resilient_quality_system import resilient_quality_gate, CircuitBreakerConfig, RetryConfig
from hipaa_compliance_summarizer.adaptive_learning_engine import learning_enabled_quality_gate
from hipaa_compliance_summarizer.intelligent_performance_optimizer import performance_optimized

class AdvancedQualitySystem:
    """Advanced quality system with all features integrated."""
    
    def __init__(self):
        self.gates = ProgressiveQualityGates()
    
    @resilient_quality_gate(
        "comprehensive_quality_check",
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
        retry_config=RetryConfig(max_attempts=2)
    )
    @learning_enabled_quality_gate("comprehensive_quality_check")
    @performance_optimized("comprehensive_quality_check")
    async def run_comprehensive_quality_check(self, project_path: str):
        """
        Run comprehensive quality check with:
        - Resilience patterns (circuit breaker, retry)
        - Adaptive learning (ML optimization)
        - Performance optimization (intelligent monitoring)
        """
        results = await self.gates.run_all_gates(project_path)
        
        # Process results
        passed = sum(1 for r in results.values() if r.status.value == 'passed')
        total = len(results)
        
        return {
            "overall_score": passed / total,
            "gates_passed": passed,
            "total_gates": total,
            "detailed_results": {
                gate_type.value: {
                    "status": result.status.value,
                    "score": result.score,
                    "duration": result.duration,
                    "recommendations": result.remediation_suggestions
                }
                for gate_type, result in results.items()
            }
        }

# Usage
async def main():
    system = AdvancedQualitySystem()
    result = await system.run_comprehensive_quality_check("/path/to/project")
    print(f"Quality check completed with {result['overall_score']:.2%} success rate")

asyncio.run(main())
```

### CI/CD Pipeline Integration

```python
import os
import sys
from pathlib import Path

class CIPipelineQualityGates:
    """CI/CD pipeline integration for quality gates."""
    
    def __init__(self):
        self.gates = ProgressiveQualityGates()
        self.workspace = Path(os.environ.get('CI_WORKSPACE', '.'))
    
    async def run_ci_quality_gates(self):
        """Run quality gates in CI/CD environment."""
        try:
            results = await self.gates.run_all_gates(str(self.workspace))
            
            # Generate CI-friendly output
            self._generate_ci_output(results)
            
            # Check if any gates failed
            failed_gates = [
                gate_type for gate_type, result in results.items()
                if result.status.value == 'failed'
            ]
            
            if failed_gates:
                print(f"❌ {len(failed_gates)} quality gates failed: {[g.value for g in failed_gates]}")
                sys.exit(1)
            else:
                print(f"✅ All {len(results)} quality gates passed")
                
        except Exception as e:
            print(f"💥 Quality gates system error: {e}")
            sys.exit(2)
    
    def _generate_ci_output(self, results):
        """Generate CI-friendly output and artifacts."""
        # Console output
        print("=" * 60)
        print("PROGRESSIVE QUALITY GATES RESULTS")
        print("=" * 60)
        
        for gate_type, result in results.items():
            status_icon = "✅" if result.status.value == "passed" else "❌"
            print(f"{status_icon} {gate_type.value.upper()}: {result.status.value} ({result.score:.2f})")
            
            if result.remediation_suggestions:
                for suggestion in result.remediation_suggestions[:3]:  # Top 3
                    print(f"   💡 {suggestion}")
        
        # Generate artifacts
        self._save_quality_report(results)
    
    def _save_quality_report(self, results):
        """Save quality report as CI artifact."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": "ci_cd",
            "workspace": str(self.workspace),
            "results": {
                gate_type.value: {
                    "status": result.status.value,
                    "score": result.score,
                    "duration": result.duration,
                    "remediation_suggestions": result.remediation_suggestions
                }
                for gate_type, result in results.items()
            }
        }
        
        with open("quality_gates_report.json", "w") as f:
            json.dump(report, f, indent=2)

# For CI/CD usage
if __name__ == "__main__":
    pipeline = CIPipelineQualityGates()
    asyncio.run(pipeline.run_ci_quality_gates())
```

## Error Handling

### Standard Exceptions

The Progressive Quality Gates system defines several custom exceptions:

```python
from hipaa_compliance_summarizer.progressive_quality_gates import (
    QualityGateError,
    ConfigurationError,
    ExecutionError
)

from hipaa_compliance_summarizer.resilient_quality_system import (
    ResilientQualityError,
    CircuitBreakerOpenError,
    BulkheadFullError,
    RetryExhaustedError
)

try:
    gates = ProgressiveQualityGates()
    results = await gates.run_all_gates("/invalid/path")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except CircuitBreakerOpenError as e:
    print(f"Circuit breaker is open: {e}")
except RetryExhaustedError as e:
    print(f"Retries exhausted: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Best Practices

1. **Always use try-catch blocks** when calling quality gate methods
2. **Check result status** before assuming success
3. **Handle CircuitBreakerOpenError** gracefully in production
4. **Log execution details** for debugging and optimization
5. **Set appropriate timeouts** based on your project size

## Version Compatibility

| Version | Python | Dependencies | Features |
|---------|--------|--------------|----------|
| 4.0.x | 3.8+ | Optional (fallbacks included) | All features |
| 3.x.x | 3.8+ | PyYAML, psutil required | Core features |
| 2.x.x | 3.7+ | Limited feature set | Basic gates |

## Migration Guide

### From Version 3.x to 4.0

```python
# Old (3.x)
from quality_gates import QualityGates
gates = QualityGates()
results = gates.run_gates()

# New (4.0)
from hipaa_compliance_summarizer.progressive_quality_gates import ProgressiveQualityGates
gates = ProgressiveQualityGates()
results = await gates.run_all_gates()
```

## Support and Resources

- **Documentation**: This API reference and deployment guide
- **Examples**: See `/examples` directory in the repository  
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join our Slack channel for support

---

**API Reference Version**: 4.0  
**Last Updated**: August 25, 2025  
**Status**: ✅ Production Ready

🤖 *Generated with Autonomous SDLC Progressive Quality Gates System*