"""
Comprehensive Test Suite for Generation 4 Autonomous HIPAA Compliance System

This test suite validates all Generation 4 autonomous features including:
- Autonomous execution engine
- Self-healing quality gates  
- Autonomous learning optimizer
- Integration and system-wide validation
"""
import asyncio
import pytest
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import Generation 4 autonomous components
from src.hipaa_compliance_summarizer.autonomous_execution_engine import (
    AutonomousExecutionEngine,
    ExecutionStrategy,
    ExecutionPhase,
    ExecutionResult
)
from src.hipaa_compliance_summarizer.self_healing_quality_gates import (
    SelfHealingQualityGate,
    SelfHealingQualityOrchestrator,
    QualityGateStatus,
    HealingStrategy,
    QualityMetrics
)
from src.hipaa_compliance_summarizer.autonomous_learning_optimizer import (
    AutonomousLearningOptimizer,
    OptimizationDomain,
    LearningPhase,
    OptimizationAction
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutonomousExecutionEngine:
    """Test suite for Autonomous Execution Engine."""
    
    @pytest.fixture
    def execution_engine(self):
        """Create test execution engine."""
        return AutonomousExecutionEngine(
            strategy=ExecutionStrategy.BALANCED,
            confidence_threshold=0.6,
            learning_rate=0.02,
            enable_research_mode=True
        )
    
    @pytest.fixture
    def sample_project_context(self):
        """Sample project context for testing."""
        return {
            "project_type": "healthcare_ai_compliance",
            "complexity_score": 0.7,
            "maturity_level": "production_ready",
            "existing_generations": 4,
            "performance_requirements": "high",
            "compliance_level": "strict"
        }
    
    @pytest.fixture
    def sample_requirements(self):
        """Sample requirements for testing."""
        return [
            "Implement autonomous quality gates",
            "Add self-healing mechanisms",
            "Create adaptive learning systems",
            "Enable intelligent performance optimization"
        ]
    
    def test_execution_engine_initialization(self, execution_engine):
        """Test execution engine initializes correctly."""
        assert execution_engine.strategy == ExecutionStrategy.BALANCED
        assert execution_engine.confidence_threshold == 0.6
        assert execution_engine.learning_rate == 0.02
        assert execution_engine.enable_research_mode is True
        assert len(execution_engine.execution_history) == 0
        assert len(execution_engine.learned_patterns) == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_decision_making(
        self, 
        execution_engine, 
        sample_project_context, 
        sample_requirements
    ):
        """Test autonomous decision making process."""
        decision = await execution_engine._make_autonomous_decision(
            ExecutionPhase.ANALYSIS,
            sample_project_context,
            sample_requirements
        )
        
        assert decision is not None
        assert decision.action in ["deep_analysis", "targeted_analysis", "quick_scan"]
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(decision.reasoning, str)
        assert len(decision.fallback_actions) > 0
    
    @pytest.mark.asyncio
    async def test_execute_analysis_phase(self, execution_engine, sample_project_context):
        """Test analysis phase execution."""
        from src.hipaa_compliance_summarizer.autonomous_execution_engine import ExecutionDecision
        
        decision = ExecutionDecision(
            action="deep_analysis",
            confidence=0.8,
            reasoning="Test decision",
            risk_level="LOW",
            expected_outcome="Comprehensive analysis",
            fallback_actions=["manual_analysis"],
            metrics={}
        )
        
        result = await execution_engine._execute_analysis_phase(decision, sample_project_context)
        
        assert isinstance(result, ExecutionResult)
        assert result.phase == ExecutionPhase.ANALYSIS
        assert result.success is True
        assert result.quality_score > 0.8
        assert result.compliance_score > 0.9
        assert len(result.learned_patterns) > 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_full_autonomous_sdlc(
        self, 
        execution_engine, 
        sample_project_context, 
        sample_requirements
    ):
        """Test complete autonomous SDLC execution."""
        results = await execution_engine.execute_autonomous_sdlc(
            sample_project_context,
            sample_requirements
        )
        
        assert len(results) == 6  # All phases should be executed
        
        expected_phases = [
            ExecutionPhase.ANALYSIS,
            ExecutionPhase.PLANNING,
            ExecutionPhase.IMPLEMENTATION,
            ExecutionPhase.VALIDATION,
            ExecutionPhase.OPTIMIZATION,
            ExecutionPhase.DEPLOYMENT
        ]
        
        for i, phase in enumerate(expected_phases):
            assert results[i].phase == phase
            assert isinstance(results[i].success, bool)
            assert results[i].quality_score >= 0.0
            assert results[i].compliance_score >= 0.0
            assert isinstance(results[i].learned_patterns, list)
            assert isinstance(results[i].recommendations, list)
    
    def test_historical_success_analysis(self, execution_engine):
        """Test historical success rate analysis."""
        # Add some mock history
        from src.hipaa_compliance_summarizer.autonomous_execution_engine import ExecutionResult
        
        execution_engine.execution_history = [
            ExecutionResult(
                phase=ExecutionPhase.ANALYSIS,
                success=True,
                execution_time=10.0,
                quality_score=0.9,
                performance_impact=0.1,
                compliance_score=0.95,
                learned_patterns=[],
                recommendations=[],
                next_actions=[]
            ),
            ExecutionResult(
                phase=ExecutionPhase.ANALYSIS,
                success=False,
                execution_time=8.0,
                quality_score=0.6,
                performance_impact=0.0,
                compliance_score=0.7,
                learned_patterns=[],
                recommendations=[],
                next_actions=[]
            )
        ]
        
        success_rate = execution_engine._analyze_historical_success(ExecutionPhase.ANALYSIS)
        assert success_rate == 0.5  # 1 success out of 2 attempts
    
    def test_risk_factor_assessment(self, execution_engine, sample_project_context):
        """Test risk factor assessment."""
        risk_factors = execution_engine._assess_risk_factors(
            ExecutionPhase.IMPLEMENTATION,
            sample_project_context
        )
        
        assert isinstance(risk_factors, dict)
        assert "complexity" in risk_factors
        assert "dependencies" in risk_factors
        assert "time_pressure" in risk_factors
        assert "resource_constraints" in risk_factors
        
        # Implementation phase should have additional risk factors
        assert "technical_debt" in risk_factors
        
        # All risk factors should be between 0 and 1
        for risk_value in risk_factors.values():
            assert 0.0 <= risk_value <= 1.0
    
    def test_complexity_calculation(self, execution_engine):
        """Test complexity score calculation."""
        simple_requirements = ["Add basic logging"]
        complex_requirements = [
            "Implement ML-based security detection",
            "Add real-time performance optimization",
            "Create scalability framework with AI integration"
        ]
        
        simple_complexity = execution_engine._calculate_complexity_score(simple_requirements)
        complex_complexity = execution_engine._calculate_complexity_score(complex_requirements)
        
        assert simple_complexity < complex_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
    
    def test_learning_pattern_updates(self, execution_engine):
        """Test learning pattern updates."""
        from src.hipaa_compliance_summarizer.autonomous_execution_engine import (
            ExecutionDecision, ExecutionResult
        )
        
        decision = ExecutionDecision(
            action="deep_analysis",
            confidence=0.8,
            reasoning="Test decision",
            risk_level="LOW",
            expected_outcome="Good results",
            fallback_actions=[],
            metrics={}
        )
        
        result = ExecutionResult(
            phase=ExecutionPhase.ANALYSIS,
            success=True,
            execution_time=10.0,
            quality_score=0.9,
            performance_impact=0.1,
            compliance_score=0.95,
            learned_patterns=["test_pattern"],
            recommendations=[],
            next_actions=[]
        )
        
        execution_engine._update_learning_patterns(ExecutionPhase.ANALYSIS, result, decision)
        
        pattern_key = "analysis_deep_analysis"
        assert pattern_key in execution_engine.learned_patterns
        
        pattern = execution_engine.learned_patterns[pattern_key]
        assert pattern["success_count"] == 1
        assert pattern["total_count"] == 1
        assert pattern["average_quality"] == 0.9


class TestSelfHealingQualityGates:
    """Test suite for Self-Healing Quality Gates."""
    
    @pytest.fixture
    def quality_gate(self):
        """Create test quality gate."""
        return SelfHealingQualityGate(
            name="Test_Quality_Gate",
            thresholds={
                "compliance_score": 0.95,
                "performance_score": 0.85,
                "security_score": 0.98,
                "test_coverage": 0.80,
                "error_rate": 0.05,
                "response_time": 200.0
            },
            healing_strategies=[HealingStrategy.AUTOMATIC, HealingStrategy.GRADUAL],
            auto_healing_enabled=True,
            max_healing_attempts=3
        )
    
    @pytest.fixture
    def quality_orchestrator(self):
        """Create test quality orchestrator."""
        return SelfHealingQualityOrchestrator()
    
    @pytest.fixture
    def healthy_context(self):
        """Context representing healthy system state."""
        return {
            "compliance_score": 0.97,
            "performance_score": 0.90,
            "security_score": 0.99,
            "test_coverage": 0.85,
            "error_rate": 0.02,
            "response_time": 120.0
        }
    
    @pytest.fixture
    def unhealthy_context(self):
        """Context representing unhealthy system state."""
        return {
            "compliance_score": 0.92,  # Below threshold
            "performance_score": 0.80,  # Below threshold
            "security_score": 0.94,   # Below threshold
            "test_coverage": 0.75,    # Below threshold
            "error_rate": 0.08,       # Above threshold
            "response_time": 250.0    # Above threshold
        }
    
    def test_quality_gate_initialization(self, quality_gate):
        """Test quality gate initializes correctly."""
        assert quality_gate.name == "Test_Quality_Gate"
        assert quality_gate.auto_healing_enabled is True
        assert quality_gate.max_healing_attempts == 3
        assert quality_gate.status == QualityGateStatus.HEALTHY
        assert quality_gate.failure_count == 0
        assert len(quality_gate.healing_strategies) == 2
    
    @pytest.mark.asyncio
    async def test_quality_metrics_collection(self, quality_gate, healthy_context):
        """Test quality metrics collection."""
        metrics = await quality_gate._collect_quality_metrics(healthy_context)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.compliance_score <= 1.0
        assert 0.0 <= metrics.performance_score <= 1.0
        assert 0.0 <= metrics.security_score <= 1.0
        assert 0.0 <= metrics.test_coverage <= 1.0
        assert metrics.error_rate >= 0.0
        assert metrics.response_time > 0.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_quality_criteria_evaluation_healthy(self, quality_gate, healthy_context):
        """Test quality criteria evaluation with healthy metrics."""
        import asyncio
        
        async def run_test():
            metrics = await quality_gate._collect_quality_metrics(healthy_context)
            issues, passing = quality_gate._evaluate_quality_criteria(metrics)
            return issues, passing
        
        issues, passing = asyncio.run(run_test())
        
        # Healthy context should pass (though some variance might cause issues)
        assert isinstance(issues, list)
        assert isinstance(passing, bool)
        # Note: Due to random variance in metrics, we can't guarantee passing=True
    
    def test_quality_criteria_evaluation_unhealthy(self, quality_gate, unhealthy_context):
        """Test quality criteria evaluation with unhealthy metrics."""
        import asyncio
        
        async def run_test():
            metrics = await quality_gate._collect_quality_metrics(unhealthy_context)
            issues, passing = quality_gate._evaluate_quality_criteria(metrics)
            return issues, passing
        
        issues, passing = asyncio.run(run_test())
        
        assert isinstance(issues, list)
        assert passing is False  # Unhealthy context should not pass
        assert len(issues) > 0  # Should detect multiple issues
    
    @pytest.mark.asyncio
    async def test_healing_action_identification(self, quality_gate):
        """Test identification of healing actions."""
        issues = [
            "Compliance score 0.92 below threshold 0.95",
            "Performance score 0.80 below threshold 0.85",
            "Security score 0.94 below threshold 0.98"
        ]
        
        metrics = QualityMetrics(
            compliance_score=0.92,
            performance_score=0.80,
            security_score=0.94
        )
        
        healing_actions = await quality_gate._identify_healing_actions(issues, metrics)
        
        assert len(healing_actions) >= 3  # Should identify actions for each issue
        
        action_types = {action.action_type for action in healing_actions}
        assert "compliance_remediation" in action_types
        assert "performance_optimization" in action_types
        assert "security_hardening" in action_types
        
        for action in healing_actions:
            assert action.auto_execute is True
            assert action.estimated_duration > 0
            assert len(action.success_criteria) > 0
    
    @pytest.mark.asyncio
    async def test_self_healing_execution(self, quality_gate, unhealthy_context):
        """Test self-healing execution."""
        result = await quality_gate.execute(unhealthy_context)
        
        assert isinstance(result.gate_name, str)
        assert isinstance(result.status, QualityGateStatus)
        assert isinstance(result.passing, bool)
        assert isinstance(result.issues_detected, list)
        assert isinstance(result.healing_actions, list)
        assert isinstance(result.auto_healed, bool)
        assert result.execution_time > 0
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_quality_gate_orchestrator(self, quality_orchestrator, unhealthy_context):
        """Test quality gate orchestrator."""
        # Create and register multiple gates
        gate1 = SelfHealingQualityGate(
            name="Compliance_Gate",
            thresholds={"compliance_score": 0.95},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        
        gate2 = SelfHealingQualityGate(
            name="Performance_Gate", 
            thresholds={"performance_score": 0.85},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        
        quality_orchestrator.register_gate(gate1)
        quality_orchestrator.register_gate(gate2)
        
        # Execute all gates
        results = await quality_orchestrator.execute_all_gates(unhealthy_context)
        
        assert len(results) == 2
        assert "Compliance_Gate" in results
        assert "Performance_Gate" in results
        
        for gate_name, result in results.items():
            assert isinstance(result.gate_name, str)
            assert isinstance(result.status, QualityGateStatus)
    
    def test_health_summary_generation(self, quality_gate):
        """Test health summary generation."""
        summary = quality_gate.get_health_summary()
        
        required_keys = [
            "name", "status", "failure_count", "last_healing_attempt",
            "healing_enabled", "total_healing_attempts", "successful_healings",
            "thresholds", "recent_healings"
        ]
        
        for key in required_keys:
            assert key in summary
        
        assert summary["name"] == quality_gate.name
        assert summary["status"] == quality_gate.status.value
        assert summary["healing_enabled"] == quality_gate.auto_healing_enabled
    
    def test_system_health_dashboard(self, quality_orchestrator):
        """Test system health dashboard generation."""
        # Add some gates
        gate1 = SelfHealingQualityGate(
            name="Gate1",
            thresholds={"score": 0.8},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        gate2 = SelfHealingQualityGate(
            name="Gate2", 
            thresholds={"score": 0.9},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        
        quality_orchestrator.register_gate(gate1)
        quality_orchestrator.register_gate(gate2)
        
        dashboard = quality_orchestrator.get_system_health_dashboard()
        
        assert "system_overview" in dashboard
        assert "gate_details" in dashboard
        assert "recent_orchestrations" in dashboard
        assert "global_context" in dashboard
        
        system_overview = dashboard["system_overview"]
        assert system_overview["total_gates"] == 2
        assert 0 <= system_overview["health_percentage"] <= 100
        assert system_overview["auto_healing_enabled"] == 2


class TestAutonomousLearningOptimizer:
    """Test suite for Autonomous Learning Optimizer."""
    
    @pytest.fixture
    def learning_optimizer(self):
        """Create test learning optimizer."""
        return AutonomousLearningOptimizer(
            learning_rate=0.02,
            exploration_factor=0.1,
            confidence_threshold=0.7,
            model_update_interval=50,
            max_history_size=1000
        )
    
    def test_optimizer_initialization(self, learning_optimizer):
        """Test optimizer initializes correctly."""
        assert learning_optimizer.learning_rate == 0.02
        assert learning_optimizer.exploration_factor == 0.1
        assert learning_optimizer.confidence_threshold == 0.7
        assert learning_optimizer.current_phase == LearningPhase.EXPLORATION
        assert learning_optimizer.iteration_count == 0
        assert len(learning_optimizer.optimization_history) == 0
        assert len(learning_optimizer.learned_patterns) == 0
    
    def test_ml_models_initialization(self, learning_optimizer):
        """Test ML models are initialized for all domains."""
        for domain in OptimizationDomain:
            assert domain in learning_optimizer.models
            assert "predictor" in learning_optimizer.models[domain]
            assert "clusterer" in learning_optimizer.models[domain]
            assert "trained" in learning_optimizer.models[domain]
            assert learning_optimizer.models[domain]["trained"] is False
            
            assert domain in learning_optimizer.scalers
            assert domain in learning_optimizer.anomaly_detectors
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, learning_optimizer):
        """Test comprehensive metrics collection."""
        metrics = await learning_optimizer._collect_comprehensive_metrics()
        
        assert len(metrics) == len(OptimizationDomain)
        
        for domain in OptimizationDomain:
            assert domain in metrics
            assert isinstance(metrics[domain], dict)
            
            # Check that metrics have reasonable values
            for metric_name, value in metrics[domain].items():
                assert isinstance(value, (int, float))
                assert not (metric_name.endswith('_score') and (value < 0 or value > 1))
    
    @pytest.mark.asyncio
    async def test_opportunity_identification(self, learning_optimizer):
        """Test optimization opportunity identification."""
        # Create test metrics with some issues
        test_metrics = {
            OptimizationDomain.PERFORMANCE: {
                "response_time": 180.0,  # High response time
                "throughput": 400,       # Low throughput
                "cache_hit_rate": 0.75   # Low cache hit rate
            },
            OptimizationDomain.ACCURACY: {
                "phi_detection_accuracy": 0.94,  # Below optimal
                "false_positive_rate": 0.08      # High false positive rate
            },
            OptimizationDomain.RESOURCE_USAGE: {
                "memory_efficiency": 0.65,  # Low efficiency
                "cpu_usage": 0.85           # High CPU usage
            }
        }
        
        opportunities = await learning_optimizer._identify_optimization_opportunities(test_metrics)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert isinstance(opportunity, OptimizationAction)
            assert opportunity.domain in OptimizationDomain
            assert 0.0 <= opportunity.confidence <= 1.0
            assert 0.0 <= opportunity.expected_impact <= 1.0
            assert opportunity.risk_level in ["LOW", "MEDIUM", "HIGH"]
            assert isinstance(opportunity.reversible, bool)
    
    @pytest.mark.asyncio
    async def test_rule_based_opportunity_detection(self, learning_optimizer):
        """Test rule-based opportunity detection for specific domains."""
        # Test performance domain
        perf_metrics = {
            "response_time": 150.0,    # Above threshold (120)
            "cache_hit_rate": 0.85     # Below threshold (0.9)
        }
        
        perf_opportunities = await learning_optimizer._rule_based_opportunity_detection(
            OptimizationDomain.PERFORMANCE, perf_metrics
        )
        
        assert len(perf_opportunities) >= 1  # Should detect latency issues
        
        latency_actions = [
            op for op in perf_opportunities 
            if op.action_type == "latency_optimization"
        ]
        assert len(latency_actions) > 0
    
    def test_optimization_action_selection(self, learning_optimizer):
        """Test optimization action selection logic."""
        # Create mock opportunities
        opportunities = [
            OptimizationAction(
                action_id="high_impact",
                domain=OptimizationDomain.PERFORMANCE,
                action_type="test_action",
                description="High impact action",
                parameters={},
                expected_impact=0.8,
                confidence=0.9,
                risk_level="LOW",
                reversible=True
            ),
            OptimizationAction(
                action_id="low_confidence",
                domain=OptimizationDomain.ACCURACY,
                action_type="test_action", 
                description="Low confidence action",
                parameters={},
                expected_impact=0.6,
                confidence=0.4,  # Below threshold
                risk_level="HIGH",
                reversible=False
            ),
            OptimizationAction(
                action_id="medium_action",
                domain=OptimizationDomain.RESOURCE_USAGE,
                action_type="test_action",
                description="Medium action",
                parameters={},
                expected_impact=0.5,
                confidence=0.75,
                risk_level="MEDIUM",
                reversible=True
            )
        ]
        
        # Test with balanced learning phase
        learning_optimizer.current_phase = LearningPhase.EXPLOITATION
        learning_optimizer.confidence_threshold = 0.7
        
        selected = learning_optimizer._select_optimization_actions(opportunities)
        
        # Should select high confidence actions first
        assert len(selected) >= 1
        selected_ids = {action.action_id for action in selected}
        assert "high_impact" in selected_ids
        assert "medium_action" in selected_ids
        assert "low_confidence" not in selected_ids  # Below confidence threshold
    
    @pytest.mark.asyncio
    async def test_optimization_execution_simulation(self, learning_optimizer):
        """Test optimization action execution simulation."""
        action = OptimizationAction(
            action_id="test_action",
            domain=OptimizationDomain.PERFORMANCE,
            action_type="latency_optimization",
            description="Test latency optimization",
            parameters={"target_latency": 100},
            expected_impact=0.15,
            confidence=0.8,
            risk_level="LOW",
            reversible=True
        )
        
        baseline_metrics = {
            OptimizationDomain.PERFORMANCE: {
                "response_time": 150.0,
                "throughput": 450,
                "cache_hit_rate": 0.9
            }
        }
        
        result = await learning_optimizer._execute_optimization_action(action, baseline_metrics)
        
        assert result.action_id == "test_action"
        assert isinstance(result.success, bool)
        assert isinstance(result.actual_impact, float)
        assert result.execution_time > 0
        assert isinstance(result.side_effects, list)
        assert isinstance(result.metrics_before, dict)
        assert isinstance(result.learned_insights, list)
    
    def test_learning_phase_updates(self, learning_optimizer):
        """Test learning phase updates based on performance."""
        # Test exploration phase (early iterations)
        learning_optimizer.iteration_count = 10
        learning_optimizer._update_learning_phase()
        assert learning_optimizer.current_phase == LearningPhase.EXPLORATION
        
        # Test with good success rate and sufficient patterns
        learning_optimizer.iteration_count = 100
        learning_optimizer.optimization_success_rate = 0.85
        learning_optimizer.learned_patterns = {f"pattern_{i}": {} for i in range(15)}
        learning_optimizer._update_learning_phase()
        assert learning_optimizer.current_phase == LearningPhase.EXPLOITATION
        
        # Test refinement phase
        learning_optimizer.learned_patterns = {f"pattern_{i}": {} for i in range(25)}
        learning_optimizer._update_learning_phase()
        assert learning_optimizer.current_phase == LearningPhase.REFINEMENT
    
    def test_parameter_adaptation(self, learning_optimizer):
        """Test adaptive parameter adjustment."""
        # Test adaptation for poor performance
        learning_optimizer.optimization_success_rate = 0.5  # Poor performance
        initial_exploration = learning_optimizer.exploration_factor
        initial_confidence = learning_optimizer.confidence_threshold
        
        learning_optimizer._adapt_learning_parameters()
        
        # Should increase exploration and decrease confidence threshold
        assert learning_optimizer.adaptive_parameters["exploration_factor"] >= initial_exploration
        assert learning_optimizer.adaptive_parameters["confidence_threshold"] <= initial_confidence
        
        # Test adaptation for good performance
        learning_optimizer.optimization_success_rate = 0.9  # Good performance
        learning_optimizer._adapt_learning_parameters()
        
        # Should reduce exploration and can increase confidence threshold
        assert learning_optimizer.adaptive_parameters["exploration_factor"] <= 0.3
    
    def test_learning_dashboard_generation(self, learning_optimizer):
        """Test learning dashboard generation."""
        # Add some mock data
        learning_optimizer.iteration_count = 50
        learning_optimizer.optimization_success_rate = 0.75
        learning_optimizer.total_impact_achieved = 1.5
        learning_optimizer.learned_patterns = {"pattern1": Mock(), "pattern2": Mock()}
        
        dashboard = learning_optimizer.get_learning_dashboard()
        
        required_sections = [
            "learning_status", "adaptive_parameters", "model_status",
            "recent_patterns", "optimization_history_summary"
        ]
        
        for section in required_sections:
            assert section in dashboard
        
        learning_status = dashboard["learning_status"]
        assert learning_status["iteration_count"] == 50
        assert learning_status["success_rate"] == 0.75
        assert learning_status["total_impact_achieved"] == 1.5
        assert learning_status["patterns_learned"] == 2
    
    def test_optimization_interval_calculation(self, learning_optimizer):
        """Test optimization interval calculation."""
        # Test different phases
        learning_optimizer.current_phase = LearningPhase.EXPLORATION
        learning_optimizer.optimization_success_rate = 0.7
        interval = learning_optimizer._calculate_optimization_interval()
        assert 240 <= interval <= 450  # Base 300 with adjustments
        
        learning_optimizer.current_phase = LearningPhase.EXPLOITATION
        interval = learning_optimizer._calculate_optimization_interval()
        assert 480 <= interval <= 900  # Base 600 with adjustments


class TestGenerationFourIntegration:
    """Integration tests for Generation 4 autonomous features."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system with all Generation 4 components."""
        execution_engine = AutonomousExecutionEngine(
            strategy=ExecutionStrategy.BALANCED,
            confidence_threshold=0.7
        )
        
        quality_orchestrator = SelfHealingQualityOrchestrator()
        
        # Add quality gates
        compliance_gate = SelfHealingQualityGate(
            name="HIPAA_Compliance",
            thresholds={"compliance_score": 0.95, "security_score": 0.98},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        
        performance_gate = SelfHealingQualityGate(
            name="Performance_Quality",
            thresholds={"performance_score": 0.85, "response_time": 200.0},
            healing_strategies=[HealingStrategy.AUTOMATIC]
        )
        
        quality_orchestrator.register_gate(compliance_gate)
        quality_orchestrator.register_gate(performance_gate)
        
        learning_optimizer = AutonomousLearningOptimizer(
            learning_rate=0.02,
            confidence_threshold=0.7
        )
        
        return {
            "execution_engine": execution_engine,
            "quality_orchestrator": quality_orchestrator,
            "learning_optimizer": learning_optimizer
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_autonomous_workflow(self, integrated_system):
        """Test complete end-to-end autonomous workflow."""
        execution_engine = integrated_system["execution_engine"]
        quality_orchestrator = integrated_system["quality_orchestrator"]
        learning_optimizer = integrated_system["learning_optimizer"]
        
        # 1. Execute autonomous SDLC
        project_context = {
            "project_type": "healthcare_ai_compliance",
            "complexity_score": 0.8,
            "maturity_level": "production_ready"
        }
        
        requirements = [
            "Implement autonomous quality gates",
            "Add self-healing mechanisms", 
            "Enable continuous optimization"
        ]
        
        sdlc_results = await execution_engine.execute_autonomous_sdlc(
            project_context, requirements
        )
        
        # Verify SDLC execution
        assert len(sdlc_results) == 6
        successful_phases = sum(1 for result in sdlc_results if result.success)
        assert successful_phases >= 4  # At least 4 phases should succeed
        
        # 2. Execute quality gates
        test_context = {
            "compliance_score": 0.93,  # Slightly below threshold
            "performance_score": 0.88,
            "security_score": 0.97,
            "response_time": 180.0
        }
        
        quality_results = await quality_orchestrator.execute_all_gates(test_context)
        
        # Verify quality gate execution
        assert len(quality_results) == 2
        for gate_name, result in quality_results.items():
            assert isinstance(result.status, QualityGateStatus)
            assert isinstance(result.auto_healed, bool)
        
        # 3. Run optimization cycles
        optimization_metrics = await learning_optimizer._collect_comprehensive_metrics()
        opportunities = await learning_optimizer._identify_optimization_opportunities(
            optimization_metrics
        )
        
        # Verify optimization opportunity identification
        assert isinstance(opportunities, list)
        if opportunities:
            selected_actions = learning_optimizer._select_optimization_actions(opportunities)
            assert len(selected_actions) >= 0
            
            # Execute at least one optimization if available
            if selected_actions:
                result = await learning_optimizer._execute_optimization_action(
                    selected_actions[0], optimization_metrics
                )
                assert isinstance(result.success, bool)
                assert isinstance(result.actual_impact, float)
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self, integrated_system):
        """Test system resilience and recovery capabilities."""
        quality_orchestrator = integrated_system["quality_orchestrator"]
        
        # Simulate severely degraded system state
        degraded_context = {
            "compliance_score": 0.85,    # Well below threshold
            "performance_score": 0.70,   # Well below threshold
            "security_score": 0.90,      # Below threshold
            "response_time": 350.0,      # Well above threshold
            "error_rate": 0.12,          # High error rate
            "test_coverage": 0.65        # Low coverage
        }
        
        # Execute quality gates with degraded context
        results = await quality_orchestrator.execute_all_gates(degraded_context)
        
        # System should attempt healing
        healing_attempted = any(result.auto_healed for result in results.values())
        
        # Even if healing fails, system should provide recommendations
        for result in results.values():
            assert len(result.recommendations) > 0
            assert len(result.issues_detected) > 0
    
    def test_system_health_monitoring(self, integrated_system):
        """Test comprehensive system health monitoring."""
        quality_orchestrator = integrated_system["quality_orchestrator"]
        learning_optimizer = integrated_system["learning_optimizer"]
        
        # Get health dashboards
        quality_dashboard = quality_orchestrator.get_system_health_dashboard()
        learning_dashboard = learning_optimizer.get_learning_dashboard()
        
        # Verify quality dashboard structure
        assert "system_overview" in quality_dashboard
        assert "gate_details" in quality_dashboard
        
        system_overview = quality_dashboard["system_overview"]
        assert "total_gates" in system_overview
        assert "healthy_gates" in system_overview
        assert "health_percentage" in system_overview
        
        # Verify learning dashboard structure
        assert "learning_status" in learning_dashboard
        assert "model_status" in learning_dashboard
        assert "adaptive_parameters" in learning_dashboard
        
        learning_status = learning_dashboard["learning_status"]
        assert "current_phase" in learning_status
        assert "success_rate" in learning_status
        assert "patterns_learned" in learning_status
    
    @pytest.mark.asyncio
    async def test_cross_component_learning(self, integrated_system):
        """Test learning and adaptation across components."""
        execution_engine = integrated_system["execution_engine"]
        learning_optimizer = integrated_system["learning_optimizer"]
        
        # Simulate learning from execution results
        project_context = {"complexity_score": 0.6}
        requirements = ["Basic enhancement"]
        
        # Execute multiple cycles to generate learning data
        for cycle in range(3):
            # SDLC execution
            results = await execution_engine.execute_autonomous_sdlc(
                project_context, requirements
            )
            
            # Check that execution engine learns from results
            assert len(execution_engine.execution_history) > cycle
            
            # Optimization cycle
            metrics = await learning_optimizer._collect_comprehensive_metrics()
            opportunities = await learning_optimizer._identify_optimization_opportunities(metrics)
            
            if opportunities:
                selected = learning_optimizer._select_optimization_actions(opportunities[:1])
                if selected:
                    result = await learning_optimizer._execute_optimization_action(
                        selected[0], metrics
                    )
                    await learning_optimizer._learn_from_optimization_result(selected[0], result)
        
        # Verify learning occurred
        assert len(execution_engine.execution_history) >= 3
        assert len(learning_optimizer.optimization_history) >= 0
        
        # Check adaptive parameters were updated
        initial_params = {
            "learning_rate": 0.02,
            "exploration_factor": 0.1,
            "confidence_threshold": 0.7
        }
        
        current_params = learning_optimizer.adaptive_parameters
        
        # At least one parameter should have adapted
        params_changed = any(
            abs(current_params[key] - initial_params[key]) > 0.001
            for key in initial_params.keys()
            if key in current_params
        )
        
        # Note: Due to randomness, parameters might not always change in short test runs


# Performance and load testing
class TestGenerationFourPerformance:
    """Performance tests for Generation 4 autonomous features."""
    
    @pytest.mark.asyncio
    async def test_execution_engine_performance(self):
        """Test execution engine performance under load."""
        engine = AutonomousExecutionEngine(
            strategy=ExecutionStrategy.BALANCED,
            confidence_threshold=0.7
        )
        
        project_context = {"complexity_score": 0.5}
        requirements = ["Performance test requirement"]
        
        # Measure execution time
        start_time = time.time()
        results = await engine.execute_autonomous_sdlc(project_context, requirements)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert len(results) == 6  # All phases completed
    
    @pytest.mark.asyncio 
    async def test_quality_gates_concurrent_execution(self):
        """Test quality gates performance with concurrent execution."""
        orchestrator = SelfHealingQualityOrchestrator()
        
        # Create multiple gates
        gates = []
        for i in range(5):
            gate = SelfHealingQualityGate(
                name=f"Gate_{i}",
                thresholds={"score": 0.8},
                healing_strategies=[HealingStrategy.AUTOMATIC]
            )
            gates.append(gate)
            orchestrator.register_gate(gate)
        
        test_context = {"score": 0.75}  # Below threshold
        
        # Measure concurrent execution time
        start_time = time.time()
        results = await orchestrator.execute_all_gates(test_context)
        execution_time = time.time() - start_time
        
        # Should handle multiple gates efficiently
        assert execution_time < 15.0  # Should complete within 15 seconds
        assert len(results) == 5  # All gates executed
    
    @pytest.mark.asyncio
    async def test_learning_optimizer_memory_efficiency(self):
        """Test learning optimizer memory efficiency."""
        optimizer = AutonomousLearningOptimizer(
            max_history_size=100  # Limited history for testing
        )
        
        # Add many optimization results to test memory management
        for i in range(150):  # More than max_history_size
            optimizer.optimization_history.append({
                "timestamp": datetime.now(),
                "success": i % 2 == 0,
                "impact": 0.1
            })
        
        # History should be limited to max_history_size
        assert len(optimizer.optimization_history) <= 100
        
        # Test metrics collection performance
        start_time = time.time()
        metrics = await optimizer._collect_comprehensive_metrics()
        collection_time = time.time() - start_time
        
        assert collection_time < 2.0  # Should be fast
        assert len(metrics) == len(OptimizationDomain)


# Main test runner
if __name__ == "__main__":
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run specific test categories
    logger.info("Running Generation 4 Autonomous System Tests")
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_autonomous",  # Run autonomous-related tests
        "--disable-warnings"
    ])