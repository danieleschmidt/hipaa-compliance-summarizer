"""Comprehensive tests for research framework components."""

import pytest
import asyncio
import json
import statistics
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.hipaa_compliance_summarizer.research_framework import (
    NovelPHIDetectionMethod,
    ResearchExperimentRunner,
    ResearchHypothesis,
    ExperimentConfig,
    ExperimentResult,
    ComparativeAnalysis,
    ResearchPhase,
    ExperimentType,
    initialize_research_framework
)


@pytest.fixture
def research_hypothesis():
    """Create research hypothesis for testing."""
    return ResearchHypothesis(
        hypothesis_id="hyp_001",
        title="Novel PHI Detection Improves Accuracy",
        description="Testing whether novel contextual embedding approach improves PHI detection accuracy",
        null_hypothesis="Novel method shows no improvement over baseline",
        alternative_hypothesis="Novel method shows statistically significant improvement",
        success_criteria=[
            "F1-score improvement > 5%",
            "Processing time increase < 50%",
            "Statistical significance p < 0.05"
        ],
        measurable_outcomes=[
            "precision", "recall", "f1_score", "processing_time_ms"
        ],
        significance_threshold=0.05,
        power_threshold=0.8
    )


@pytest.fixture
def experiment_config(research_hypothesis):
    """Create experiment configuration for testing."""
    return ExperimentConfig(
        experiment_id="exp_001",
        experiment_type=ExperimentType.COMPARATIVE_STUDY,
        hypothesis=research_hypothesis,
        dataset_config={"size": 1000, "phi_categories": ["names", "ssn", "dates"]},
        baseline_methods=["pattern_based_detection"],
        novel_methods=["novel_phi_detection"],
        evaluation_metrics=["precision", "recall", "f1_score", "processing_time_ms"],
        sample_size=500,
        significance_level=0.05,
        random_seed=42,
        cross_validation_folds=5
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    return {
        "samples": [
            {
                "id": "sample_001",
                "text": "Patient John Doe, SSN: 123-45-6789, was admitted on 01/15/2024.",
                "ground_truth_phi": [
                    {"text": "John Doe", "category": "names", "start": 8, "end": 16},
                    {"text": "123-45-6789", "category": "ssn", "start": 23, "end": 34},
                    {"text": "01/15/2024", "category": "dates", "start": 51, "end": 61}
                ]
            },
            {
                "id": "sample_002", 
                "text": "Dr. Smith prescribed medication for patient ID HSP-456789.",
                "ground_truth_phi": [
                    {"text": "Dr. Smith", "category": "names", "start": 0, "end": 9},
                    {"text": "HSP-456789", "category": "medical_ids", "start": 49, "end": 59}
                ]
            }
        ]
    }


@pytest.fixture
def novel_phi_method():
    """Create novel PHI detection method for testing."""
    config = {
        "use_contextual_embeddings": True,
        "ensemble_size": 3,
        "confidence_threshold": 0.8
    }
    return NovelPHIDetectionMethod(config)


@pytest.fixture
def experiment_runner():
    """Create experiment runner for testing."""
    return ResearchExperimentRunner()


class TestNovelPHIDetectionMethod:
    """Test cases for Novel PHI Detection Method."""
    
    def test_initialization(self, novel_phi_method):
        """Test novel PHI method initialization."""
        assert novel_phi_method.method_name == "novel_phi_detection"
        assert novel_phi_method.use_contextual_embeddings is True
        assert novel_phi_method.ensemble_size == 3
        assert novel_phi_method.confidence_threshold == 0.8
        assert not novel_phi_method.is_initialized
    
    def test_initialize(self, novel_phi_method):
        """Test method initialization."""
        result = novel_phi_method.initialize()
        assert result is True
        assert novel_phi_method.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_run_experiment(self, novel_phi_method, experiment_config, sample_dataset):
        """Test running experiment with novel method."""
        novel_phi_method.initialize()
        
        result = await novel_phi_method.run_experiment(sample_dataset, experiment_config)
        
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == experiment_config.experiment_id
        assert result.method_name == "novel_phi_detection"
        assert len(result.metrics) > 0
        assert result.processing_time_ms > 0
        assert result.reproducibility_score > 0
        
        # Check metrics structure
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1_score" in result.metrics
        assert "accuracy" in result.metrics
        
        # Metrics should be in valid range
        for metric_name, value in result.metrics.items():
            if metric_name not in ["true_positives", "false_positives", "false_negatives"]:
                assert 0 <= value <= 1, f"Metric {metric_name} out of range: {value}"
    
    def test_contextual_embedding_detection(self, novel_phi_method):
        """Test contextual embedding detection."""
        text = "Patient Mary Johnson was diagnosed with diabetes by Dr. Smith."
        
        entities = novel_phi_method._contextual_embedding_detection(text)
        
        assert isinstance(entities, list)
        assert len(entities) >= 0  # May or may not find entities
        
        # Check entity structure
        for entity in entities:
            assert isinstance(entity, dict)
            assert "text" in entity
            assert "category" in entity
            assert "confidence" in entity
            assert "start_position" in entity
            assert "end_position" in entity
            assert 0 <= entity["confidence"] <= 1
    
    def test_calculate_context_confidence(self, novel_phi_method):
        """Test context confidence calculation."""
        # Text with healthcare indicators
        text = "The patient was seen by the doctor at the hospital for treatment."
        entity_start = 4  # "patient"
        entity_end = 11
        
        boost = novel_phi_method._calculate_context_confidence(text, entity_start, entity_end)
        
        assert isinstance(boost, float)
        assert 0 <= boost <= 0.3  # Should be capped at 0.3
        assert boost > 0  # Should get some boost from healthcare context
    
    def test_ensemble_scoring(self, novel_phi_method):
        """Test ensemble scoring mechanism."""
        contextual_entities = [
            {
                "text": "John Doe",
                "category": "names",
                "confidence": 0.9,
                "start_position": 0,
                "end_position": 8,
                "detection_method": "contextual"
            }
        ]
        
        relationship_entities = [
            {
                "text": "John Doe",
                "category": "names", 
                "confidence": 0.85,
                "start_position": 0,
                "end_position": 8,
                "detection_method": "relationship"
            }
        ]
        
        final_entities = novel_phi_method._ensemble_scoring(
            contextual_entities, 
            relationship_entities
        )
        
        assert isinstance(final_entities, list)
        # Should only keep entities above confidence threshold
        for entity in final_entities:
            assert entity["confidence"] >= novel_phi_method.confidence_threshold
    
    def test_calculate_reproducibility_score(self, novel_phi_method):
        """Test reproducibility score calculation."""
        # Create multiple results with varying metrics
        results = [
            ExperimentResult(
                experiment_id="test_1",
                method_name="test_method",
                metrics={"accuracy": 0.95, "f1_score": 0.92},
                confidence_intervals={},
                statistical_tests={},
                raw_predictions=[],
                processing_time_ms=100.0,
                memory_usage_mb=50.0,
                reproducibility_score=0.0,
                metadata={}
            ),
            ExperimentResult(
                experiment_id="test_2",
                method_name="test_method", 
                metrics={"accuracy": 0.94, "f1_score": 0.93},
                confidence_intervals={},
                statistical_tests={},
                raw_predictions=[],
                processing_time_ms=105.0,
                memory_usage_mb=52.0,
                reproducibility_score=0.0,
                metadata={}
            ),
            ExperimentResult(
                experiment_id="test_3",
                method_name="test_method",
                metrics={"accuracy": 0.96, "f1_score": 0.91},
                confidence_intervals={},
                statistical_tests={},
                raw_predictions=[], 
                processing_time_ms=98.0,
                memory_usage_mb=48.0,
                reproducibility_score=0.0,
                metadata={}
            )
        ]
        
        score = novel_phi_method.calculate_reproducibility_score(results)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        # With low variance, should have high reproducibility
        assert score > 0.8


class TestResearchExperimentRunner:
    """Test cases for Research Experiment Runner."""
    
    def test_initialization(self, experiment_runner):
        """Test experiment runner initialization."""
        assert isinstance(experiment_runner.experiments, dict)
        assert len(experiment_runner.experiments) == 0
        assert isinstance(experiment_runner.results, dict) 
        assert len(experiment_runner.results) == 0
        assert isinstance(experiment_runner.methods, dict)
        assert len(experiment_runner.methods) == 0
    
    def test_register_method(self, experiment_runner, novel_phi_method):
        """Test method registration."""
        experiment_runner.register_method(novel_phi_method)
        
        assert "novel_phi_detection" in experiment_runner.methods
        assert experiment_runner.methods["novel_phi_detection"] == novel_phi_method
    
    def test_create_experiment(self, experiment_runner, experiment_config):
        """Test experiment creation."""
        experiment_id = experiment_runner.create_experiment(experiment_config)
        
        assert experiment_id == experiment_config.experiment_id
        assert experiment_id in experiment_runner.experiments
        assert experiment_runner.experiments[experiment_id] == experiment_config
    
    @pytest.mark.asyncio
    async def test_run_comparative_study(self, experiment_runner, novel_phi_method, experiment_config, sample_dataset):
        """Test running comparative study."""
        # Register method
        experiment_runner.register_method(novel_phi_method)
        
        # Create baseline method mock
        baseline_method = Mock()
        baseline_method.method_name = "pattern_based_detection"
        baseline_method.initialize.return_value = True
        baseline_method.is_initialized = True
        
        # Mock baseline run_experiment
        async def mock_baseline_experiment(dataset, config):
            return ExperimentResult(
                experiment_id=config.experiment_id,
                method_name="pattern_based_detection",
                metrics={
                    "precision": 0.85,
                    "recall": 0.80,
                    "f1_score": 0.825,
                    "accuracy": 0.82,
                    "true_positives": 40,
                    "false_positives": 7,
                    "false_negatives": 10
                },
                confidence_intervals={},
                statistical_tests={},
                raw_predictions=[],
                processing_time_ms=50.0,
                memory_usage_mb=25.0,
                reproducibility_score=0.92,
                metadata={}
            )
        
        baseline_method.run_experiment = mock_baseline_experiment
        experiment_runner.register_method(baseline_method)
        
        # Create experiment
        experiment_runner.create_experiment(experiment_config)
        
        # Run comparative study
        analysis = await experiment_runner.run_comparative_study(
            experiment_config.experiment_id,
            sample_dataset,
            runs_per_method=1  # Reduce for testing
        )
        
        assert isinstance(analysis, ComparativeAnalysis)
        assert analysis.experiment_id == experiment_config.experiment_id
        assert analysis.baseline_method == "pattern_based_detection"
        assert analysis.novel_method == "novel_phi_detection"
        
        # Check performance improvement metrics
        assert isinstance(analysis.performance_improvement, dict)
        assert len(analysis.performance_improvement) > 0
        
        # Check statistical significance
        assert isinstance(analysis.statistical_significance, dict)
        assert isinstance(analysis.p_values, dict)
        assert isinstance(analysis.effect_sizes, dict)
        
        # Check conclusions and recommendations
        assert isinstance(analysis.conclusions, list)
        assert len(analysis.conclusions) > 0
        assert isinstance(analysis.recommendations, list) 
        assert len(analysis.recommendations) > 0
    
    def test_perform_t_test(self, experiment_runner):
        """Test t-test implementation."""
        group1 = [0.85, 0.87, 0.84, 0.86, 0.88]
        group2 = [0.92, 0.94, 0.91, 0.93, 0.95]
        
        p_value = experiment_runner._perform_t_test(group1, group2)
        
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        # Groups are clearly different, so p-value should be relatively small
        assert p_value < 0.5
    
    def test_calculate_cohens_d(self, experiment_runner):
        """Test Cohen's d effect size calculation."""
        group1 = [0.80, 0.82, 0.81, 0.83, 0.79]
        group2 = [0.90, 0.92, 0.91, 0.93, 0.89]
        
        cohens_d = experiment_runner._calculate_cohens_d(group1, group2)
        
        assert isinstance(cohens_d, float)
        # Groups are separated, should have large effect size
        assert abs(cohens_d) > 2.0  # Large effect size
    
    def test_generate_conclusions(self, experiment_runner):
        """Test conclusions generation."""
        improvements = {
            "precision": 15.0,  # 15% improvement
            "recall": -5.0,     # 5% decrease
            "f1_score": 8.0     # 8% improvement
        }
        
        significance = {
            "precision": True,
            "recall": False,
            "f1_score": True
        }
        
        effect_sizes = {
            "precision": 1.2,   # Large effect
            "recall": -0.3,     # Small effect
            "f1_score": 0.8     # Medium effect
        }
        
        conclusions = experiment_runner._generate_conclusions(
            improvements, significance, effect_sizes
        )
        
        assert isinstance(conclusions, list)
        assert len(conclusions) == 3  # One for each metric
        
        # Check content
        precision_conclusion = conclusions[0]
        assert "15.0% improvement" in precision_conclusion
        assert "precision" in precision_conclusion
        assert "statistically significant" in precision_conclusion
    
    def test_generate_recommendations(self, experiment_runner):
        """Test recommendations generation.""" 
        improvements = {
            "precision": 10.0,
            "recall": 8.0,
            "f1_score": 12.0
        }
        
        significance = {
            "precision": True,
            "recall": True, 
            "f1_score": True
        }
        
        recommendations = experiment_runner._generate_recommendations(
            improvements, significance
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend deployment due to significant improvements
        deployment_rec = any("deployment" in rec.lower() for rec in recommendations)
        assert deployment_rec
        
        # Should mention peer review
        peer_review_rec = any("peer review" in rec.lower() for rec in recommendations)
        assert peer_review_rec
    
    def test_generate_publication_report(self, experiment_runner, experiment_config, novel_phi_method):
        """Test publication report generation."""
        # Set up experiment
        experiment_runner.register_method(novel_phi_method)
        experiment_runner.create_experiment(experiment_config)
        
        # Add some fake results
        fake_result = ExperimentResult(
            experiment_id=experiment_config.experiment_id,
            method_name="novel_phi_detection",
            metrics={"precision": 0.95, "recall": 0.92, "f1_score": 0.935},
            confidence_intervals={"precision": (0.93, 0.97)},
            statistical_tests={"normality_test": {"p_value": 0.15}},
            raw_predictions=[],
            processing_time_ms=150.0,
            memory_usage_mb=75.0,
            reproducibility_score=0.94,
            metadata={}
        )
        experiment_runner.results["novel_phi_detection"] = [fake_result]
        
        report = experiment_runner.generate_publication_report(experiment_config.experiment_id)
        
        assert isinstance(report, dict)
        
        # Check main sections
        assert "experiment_metadata" in report
        assert "methodology" in report
        assert "results" in report
        assert "statistical_analysis" in report
        assert "conclusions" in report
        assert "reproducibility" in report
        
        # Check experiment metadata
        metadata = report["experiment_metadata"]
        assert metadata["experiment_id"] == experiment_config.experiment_id
        assert metadata["experiment_type"] == experiment_config.experiment_type.value
        assert metadata["dataset_size"] == experiment_config.sample_size
        
        # Check methodology
        methodology = report["methodology"]
        assert methodology["baseline_methods"] == experiment_config.baseline_methods
        assert methodology["novel_methods"] == experiment_config.novel_methods
        
        # Check results
        results = report["results"]
        assert "raw_results" in results
        assert "summary_statistics" in results
        assert "reproducibility_analysis" in results
        
        # Check conclusions
        conclusions = report["conclusions"]
        assert "key_findings" in conclusions
        assert "implications" in conclusions
        assert "limitations" in conclusions
        assert "future_work" in conclusions
        
        # Check reproducibility
        reproducibility = report["reproducibility"]
        assert "code_version" in reproducibility
        assert "environment_details" in reproducibility
        assert "reproducibility_checklist" in reproducibility


class TestResearchHypothesis:
    """Test cases for Research Hypothesis."""
    
    def test_research_hypothesis_creation(self, research_hypothesis):
        """Test research hypothesis creation."""
        assert research_hypothesis.hypothesis_id == "hyp_001"
        assert research_hypothesis.title == "Novel PHI Detection Improves Accuracy"
        assert research_hypothesis.significance_threshold == 0.05
        assert research_hypothesis.power_threshold == 0.8
        assert len(research_hypothesis.success_criteria) == 3
        assert len(research_hypothesis.measurable_outcomes) == 4
        assert research_hypothesis.created_at is not None
        
        # Check that timestamp is valid ISO format
        datetime.fromisoformat(research_hypothesis.created_at)
    
    def test_hypothesis_auto_timestamp(self):
        """Test automatic timestamp generation."""
        hypothesis1 = ResearchHypothesis(
            hypothesis_id="test_1",
            title="Test Hypothesis",
            description="Test description",
            null_hypothesis="No effect",
            alternative_hypothesis="Has effect",
            success_criteria=["criteria 1"],
            measurable_outcomes=["outcome 1"]
        )
        
        hypothesis2 = ResearchHypothesis(
            hypothesis_id="test_2", 
            title="Test Hypothesis 2",
            description="Test description 2",
            null_hypothesis="No effect",
            alternative_hypothesis="Has effect",
            success_criteria=["criteria 1"],
            measurable_outcomes=["outcome 1"]
        )
        
        # Timestamps should be different (assuming not created at exact same time)
        assert hypothesis1.created_at != hypothesis2.created_at


class TestExperimentConfig:
    """Test cases for Experiment Config."""
    
    def test_experiment_config_creation(self, experiment_config):
        """Test experiment configuration creation."""
        assert experiment_config.experiment_id == "exp_001"
        assert experiment_config.experiment_type == ExperimentType.COMPARATIVE_STUDY
        assert experiment_config.sample_size == 500
        assert experiment_config.significance_level == 0.05
        assert experiment_config.random_seed == 42
        assert experiment_config.cross_validation_folds == 5
        assert len(experiment_config.baseline_methods) == 1
        assert len(experiment_config.novel_methods) == 1
        assert len(experiment_config.evaluation_metrics) == 4
        assert experiment_config.metadata == {}  # Default empty dict
    
    def test_experiment_config_with_metadata(self, research_hypothesis):
        """Test experiment config with metadata."""
        config = ExperimentConfig(
            experiment_id="exp_002",
            experiment_type=ExperimentType.PERFORMANCE_BENCHMARK,
            hypothesis=research_hypothesis,
            dataset_config={"test": True},
            baseline_methods=["method1"],
            novel_methods=["method2"],
            evaluation_metrics=["metric1"],
            sample_size=100,
            metadata={"custom_param": "value"}
        )
        
        assert config.metadata["custom_param"] == "value"


class TestExperimentResult:
    """Test cases for Experiment Result."""
    
    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        metrics = {"accuracy": 0.95, "precision": 0.92}
        confidence_intervals = {"accuracy": (0.93, 0.97)}
        
        result = ExperimentResult(
            experiment_id="test_exp",
            method_name="test_method",
            metrics=metrics,
            confidence_intervals=confidence_intervals,
            statistical_tests={},
            raw_predictions=[],
            processing_time_ms=100.0,
            memory_usage_mb=50.0,
            reproducibility_score=0.95,
            metadata={"test": True}
        )
        
        assert result.experiment_id == "test_exp"
        assert result.method_name == "test_method"
        assert result.metrics == metrics
        assert result.confidence_intervals == confidence_intervals
        assert result.processing_time_ms == 100.0
        assert result.memory_usage_mb == 50.0
        assert result.reproducibility_score == 0.95
        assert result.metadata["test"] is True
        assert result.timestamp is not None
        
        # Check timestamp format
        datetime.fromisoformat(result.timestamp)


class TestComparativeAnalysis:
    """Test cases for Comparative Analysis."""
    
    def test_comparative_analysis_creation(self):
        """Test comparative analysis creation."""
        analysis = ComparativeAnalysis(
            experiment_id="exp_001",
            baseline_method="baseline",
            novel_method="novel",
            performance_improvement={"accuracy": 10.0},
            statistical_significance={"accuracy": True},
            effect_sizes={"accuracy": 1.2},
            confidence_level=0.95,
            p_values={"accuracy": 0.01},
            conclusions=["Novel method is better"],
            recommendations=["Deploy novel method"]
        )
        
        assert analysis.experiment_id == "exp_001"
        assert analysis.baseline_method == "baseline"
        assert analysis.novel_method == "novel"
        assert analysis.performance_improvement["accuracy"] == 10.0
        assert analysis.statistical_significance["accuracy"] is True
        assert analysis.effect_sizes["accuracy"] == 1.2
        assert analysis.confidence_level == 0.95
        assert analysis.p_values["accuracy"] == 0.01
        assert len(analysis.conclusions) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.timestamp is not None


class TestIntegrationScenarios:
    """Integration test scenarios for research framework."""
    
    @pytest.mark.asyncio
    async def test_complete_research_pipeline(self):
        """Test complete research pipeline from hypothesis to publication."""
        # Initialize research framework
        runner = initialize_research_framework()
        
        # Define research hypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id="integration_test",
            title="Integration Test Hypothesis",
            description="Testing complete research pipeline",
            null_hypothesis="No improvement in PHI detection",
            alternative_hypothesis="Significant improvement in PHI detection",
            success_criteria=["F1-score improvement > 3%"],
            measurable_outcomes=["f1_score", "precision", "recall"]
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id="integration_experiment",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            hypothesis=hypothesis,
            dataset_config={"size": 100},
            baseline_methods=["novel_phi_detection"],  # Use same method as both for testing
            novel_methods=["novel_phi_detection"],
            evaluation_metrics=["precision", "recall", "f1_score"],
            sample_size=50,
            runs_per_method=2
        )
        
        # Create experiment
        runner.create_experiment(config)
        
        # Create test dataset
        test_dataset = {
            "samples": [
                {
                    "id": f"sample_{i}",
                    "text": f"Patient Sample {i}, SSN: 123-45-678{i % 10}",
                }
                for i in range(10)
            ],
            "ground_truth": [
                {
                    "entities": [
                        {"start_position": 8, "end_position": 16, "category": "names"},
                        {"start_position": 23, "end_position": 35, "category": "ssn"}
                    ]
                }
                for _ in range(10)
            ]
        }
        
        # Run comparative study
        analysis = await runner.run_comparative_study(
            config.experiment_id,
            test_dataset,
            runs_per_method=1  # Single run for testing
        )
        
        # Validate analysis results
        assert isinstance(analysis, ComparativeAnalysis)
        assert analysis.experiment_id == config.experiment_id
        
        # Generate publication report
        report = runner.generate_publication_report(config.experiment_id)
        
        # Validate report structure
        assert isinstance(report, dict)
        assert "experiment_metadata" in report
        assert "methodology" in report
        assert "results" in report
        assert "conclusions" in report
        
        # Check that report contains research details
        metadata = report["experiment_metadata"]
        assert metadata["experiment_id"] == config.experiment_id
        assert metadata["sample_size"] == config.sample_size
    
    @pytest.mark.asyncio
    async def test_multiple_method_comparison(self):
        """Test comparison of multiple research methods."""
        runner = initialize_research_framework()
        
        # Create multiple methods with different configurations
        method1 = NovelPHIDetectionMethod({
            "ensemble_size": 1,
            "confidence_threshold": 0.7
        })
        
        method2 = NovelPHIDetectionMethod({
            "ensemble_size": 3,
            "confidence_threshold": 0.8
        })
        
        method3 = NovelPHIDetectionMethod({
            "ensemble_size": 5,
            "confidence_threshold": 0.9
        })
        
        # Register methods with different names
        method1.method_name = "low_confidence_method"
        method2.method_name = "medium_confidence_method"
        method3.method_name = "high_confidence_method"
        
        runner.register_method(method1)
        runner.register_method(method2)
        runner.register_method(method3)
        
        # Create dataset
        dataset = {
            "samples": [
                {"id": f"sample_{i}", "text": f"Test text {i} with PHI"}
                for i in range(5)
            ]
        }
        
        # Test each method
        methods_to_test = ["low_confidence_method", "medium_confidence_method", "high_confidence_method"]
        
        for method_name in methods_to_test:
            # Create experiment config for each method
            hypothesis = ResearchHypothesis(
                hypothesis_id=f"hyp_{method_name}",
                title=f"Testing {method_name}",
                description=f"Evaluating performance of {method_name}",
                null_hypothesis="Method shows standard performance",
                alternative_hypothesis="Method shows improved performance",
                success_criteria=["Acceptable performance"],
                measurable_outcomes=["precision", "recall"]
            )
            
            config = ExperimentConfig(
                experiment_id=f"exp_{method_name}",
                experiment_type=ExperimentType.ALGORITHM_EVALUATION,
                hypothesis=hypothesis,
                dataset_config={},
                baseline_methods=[method_name],
                novel_methods=[method_name],
                evaluation_metrics=["precision", "recall", "f1_score"],
                sample_size=5
            )
            
            runner.create_experiment(config)
            
            # Run single method evaluation
            method = runner.methods[method_name]
            method.initialize()
            result = await method.run_experiment(dataset, config)
            
            # Validate result
            assert isinstance(result, ExperimentResult)
            assert result.method_name == method_name
            assert len(result.metrics) > 0
    
    def test_statistical_power_analysis(self, experiment_runner):
        """Test statistical power analysis capabilities."""
        # Simulate effect size analysis
        small_effect_group1 = [0.80, 0.81, 0.79, 0.82, 0.80]
        small_effect_group2 = [0.82, 0.83, 0.81, 0.84, 0.82]  # Small difference
        
        large_effect_group1 = [0.70, 0.71, 0.69, 0.72, 0.70]
        large_effect_group2 = [0.90, 0.91, 0.89, 0.92, 0.90]  # Large difference
        
        # Test effect size calculation
        small_effect = experiment_runner._calculate_cohens_d(small_effect_group1, small_effect_group2)
        large_effect = experiment_runner._calculate_cohens_d(large_effect_group1, large_effect_group2)
        
        # Large effect should be much larger than small effect
        assert abs(large_effect) > abs(small_effect)
        assert abs(large_effect) > 2.0  # Large effect size threshold
        assert abs(small_effect) < 1.0   # Small effect size threshold
        
        # Test statistical significance
        small_p_value = experiment_runner._perform_t_test(small_effect_group1, small_effect_group2)
        large_p_value = experiment_runner._perform_t_test(large_effect_group1, large_effect_group2)
        
        # Large effect should have smaller p-value (more significant)
        assert large_p_value < small_p_value
    
    @pytest.mark.asyncio
    async def test_reproducibility_validation(self):
        """Test reproducibility validation across multiple runs."""
        runner = initialize_research_framework()
        
        # Create method
        method = NovelPHIDetectionMethod({"random_seed_variant": True})
        runner.register_method(method)
        
        # Create simple dataset
        dataset = {
            "samples": [
                {"id": "sample_1", "text": "Patient John Doe with condition"}
            ]
        }
        
        # Run multiple times with same configuration
        results = []
        for i in range(3):
            hypothesis = ResearchHypothesis(
                hypothesis_id=f"repro_{i}",
                title="Reproducibility Test",
                description="Testing reproducibility",
                null_hypothesis="No reproducible results",
                alternative_hypothesis="Results are reproducible",
                success_criteria=["Consistent results"],
                measurable_outcomes=["precision"]
            )
            
            config = ExperimentConfig(
                experiment_id=f"repro_exp_{i}",
                experiment_type=ExperimentType.REPRODUCIBILITY_STUDY,
                hypothesis=hypothesis,
                dataset_config={},
                baseline_methods=["novel_phi_detection"],
                novel_methods=["novel_phi_detection"],
                evaluation_metrics=["precision", "recall"],
                sample_size=1,
                random_seed=42  # Same seed for reproducibility
            )
            
            result = await method.run_experiment(dataset, config)
            results.append(result)
        
        # Calculate reproducibility score
        reproducibility_score = method.calculate_reproducibility_score(results)
        
        # Should have high reproducibility with same seed
        assert reproducibility_score > 0.8
        assert isinstance(reproducibility_score, float)
        assert 0 <= reproducibility_score <= 1


@pytest.mark.asyncio
async def test_initialize_research_framework():
    """Test research framework initialization."""
    config = {
        "novel_phi": {
            "ensemble_size": 5,
            "confidence_threshold": 0.85,
            "use_contextual_embeddings": True
        }
    }
    
    runner = initialize_research_framework(config)
    
    assert isinstance(runner, ResearchExperimentRunner)
    assert "novel_phi_detection" in runner.methods
    
    # Check method configuration
    method = runner.methods["novel_phi_detection"]
    assert method.ensemble_size == 5
    assert method.confidence_threshold == 0.85
    assert method.use_contextual_embeddings is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])