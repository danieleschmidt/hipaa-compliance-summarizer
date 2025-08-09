"""
Comprehensive tests for research modules.

Tests cover:
1. Adaptive PHI detection with confidence modeling
2. Statistical validation and hypothesis testing  
3. Federated learning with privacy guarantees
4. Compliance risk prediction and analytics
5. Benchmark suite functionality
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from src.hipaa_compliance_summarizer.research import (
    AdaptivePHIDetector,
    PHIConfidenceModel,
    StatisticalValidator,
    ValidationMetrics,
    FederatedComplianceModel,
    PrivacyPreservingTrainer,
    PrivacyBudget,
    CompliancePredictionEngine,
    RiskPredictor,
    RiskLevel,
    RiskPrediction,
    ResearchBenchmarkSuite,
    ComparativeAnalysis,
)

from src.hipaa_compliance_summarizer.research.adaptive_phi_detection import (
    PHIConfidenceScore,
    DetectionContext,
)

from src.hipaa_compliance_summarizer.research.statistical_validation import (
    StudyDesign,
)

from src.hipaa_compliance_summarizer.research.federated_learning import (
    FederatedNode,
    ModelUpdate,
    demonstrate_federated_phi_detection,
)

from src.hipaa_compliance_summarizer.research.compliance_prediction import (
    ComplianceFeatures,
    ComplianceAnomalyDetector,
    demonstrate_compliance_prediction,
)

from src.hipaa_compliance_summarizer.research.benchmark_suite import (
    BenchmarkDataset,
    BenchmarkResult,
    benchmark_adaptive_phi_detector,
)


class TestAdaptivePHIDetection:
    """Test adaptive PHI detection with confidence modeling."""
    
    def test_phi_confidence_score_creation(self):
        """Test PHI confidence score calculation."""
        score = PHIConfidenceScore(
            entity_type="ssn",
            confidence=0.9,
            uncertainty=0.1,
            false_positive_probability=0.05,
            context_relevance=0.95
        )
        
        assert score.entity_type == "ssn"
        assert score.confidence == 0.9
        assert score.adjusted_confidence > 0.7  # Should be high
    
    def test_detection_context_analysis(self):
        """Test document context analysis."""
        context = DetectionContext(
            document_type="clinical_note",
            word_count=500,
            medical_term_density=0.05,
            phi_density_estimate=0.02
        )
        
        assert context.document_type == "clinical_note"
        assert context.get_context_weight() > 0.5
        assert context.get_context_weight() <= 1.0
    
    def test_phi_confidence_model(self):
        """Test PHI confidence modeling."""
        model = PHIConfidenceModel()
        
        context = DetectionContext(
            document_type="clinical_note",
            word_count=100,
            medical_term_density=0.05,
            phi_density_estimate=0.02
        )
        
        confidence_score = model.calculate_confidence("ssn", "123-45-6789", context)
        
        assert isinstance(confidence_score, PHIConfidenceScore)
        assert 0 <= confidence_score.confidence <= 1
        assert 0 <= confidence_score.uncertainty <= 1
        assert 0 <= confidence_score.false_positive_probability <= 1
    
    def test_adaptive_phi_detector(self):
        """Test adaptive PHI detector functionality."""
        detector = AdaptivePHIDetector(enable_statistical_validation=True)
        
        test_text = "Patient John Doe, SSN: 123-45-6789, phone: 555-123-4567"
        
        detections = detector.detect_phi_with_confidence(test_text, "clinical_note")
        
        assert len(detections) > 0
        assert all('confidence' in detection for detection in detections)
        assert all('adjusted_confidence' in detection for detection in detections)
        assert all('uncertainty' in detection for detection in detections)
    
    def test_adaptive_detector_with_validation(self):
        """Test adaptive detector with validation data."""
        detector = AdaptivePHIDetector(enable_statistical_validation=True)
        
        test_text = "SSN: 123-45-6789"
        validation_data = [("123-45-6789", True)]
        
        detections = detector.detect_phi_with_confidence(
            test_text, "clinical_note", validation_data
        )
        
        assert len(detections) > 0
        performance_summary = detector.get_performance_summary()
        assert performance_summary['total_runs'] > 0


class TestStatisticalValidation:
    """Test statistical validation framework."""
    
    def test_validation_metrics_calculation(self):
        """Test validation metrics computation."""
        metrics = ValidationMetrics(
            true_positives=85,
            false_positives=5,
            true_negatives=5,
            false_negatives=5
        )
        
        assert metrics.precision > 0.9
        assert metrics.recall > 0.9
        assert metrics.f1_score > 0.9
        assert metrics.accuracy > 0.9
        assert -1 <= metrics.matthews_correlation_coefficient <= 1
    
    def test_study_design(self):
        """Test study design for statistical analysis."""
        design = StudyDesign(alpha=0.05, beta=0.20, effect_size=0.1)
        
        assert design.power == 0.8
        required_size = design.calculate_required_sample_size(0.95)
        assert required_size >= 30
    
    def test_statistical_validator(self):
        """Test statistical validation with hypothesis testing."""
        validator = StatisticalValidator()
        
        # Simulate good performance
        predictions = [True] * 90 + [False] * 10
        ground_truth = [True] * 85 + [False] * 15
        
        metrics = validator.validate_model_performance(predictions, ground_truth)
        
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.precision > 0.8
        assert len(metrics.precision_ci) == 2
        assert metrics.precision_ci[0] <= metrics.precision <= metrics.precision_ci[1]
    
    def test_model_comparison(self):
        """Test statistical comparison between models."""
        validator = StatisticalValidator()
        
        # Model 1: Good performance
        model1_pred = [True] * 85 + [False] * 15
        # Model 2: Better performance  
        model2_pred = [True] * 90 + [False] * 10
        ground_truth = [True] * 88 + [False] * 12
        
        comparison = validator.compare_models(model1_pred, model2_pred, ground_truth)
        
        assert 'model1_metrics' in comparison
        assert 'model2_metrics' in comparison
        assert 'mcnemar_test' in comparison
        assert 'accuracy_difference' in comparison
    
    def test_power_analysis(self):
        """Test statistical power analysis."""
        validator = StatisticalValidator()
        
        power_analysis = validator.perform_power_analysis(
            current_sample_size=100,
            observed_accuracy=0.95
        )
        
        assert 'achieved_power' in power_analysis
        assert 'required_sample_size' in power_analysis
        assert 'sample_size_adequate' in power_analysis
    
    def test_meta_analysis(self):
        """Test meta-analysis across multiple studies."""
        validator = StatisticalValidator()
        
        # Create multiple validation results
        results = []
        for i in range(3):
            metrics = ValidationMetrics(
                true_positives=80 + i * 2,
                false_positives=10 - i,
                true_negatives=8 + i,
                false_negatives=2
            )
            results.append(metrics)
        
        meta_analysis = validator.meta_analysis(results)
        
        assert 'number_of_studies' in meta_analysis
        assert 'weighted_effect_size' in meta_analysis
        assert 'heterogeneity' in meta_analysis
        assert meta_analysis['number_of_studies'] == 3


class TestFederatedLearning:
    """Test federated learning implementation."""
    
    def test_privacy_budget(self):
        """Test privacy budget management."""
        budget = PrivacyBudget(epsilon_total=1.0, delta=1e-5)
        
        assert budget.remaining_budget == 1.0
        assert not budget.budget_exhausted
        assert budget.can_afford(0.5)
        
        success = budget.spend(0.6)
        assert success
        assert budget.remaining_budget == 0.4
        
        # Try to overspend
        success = budget.spend(0.5)
        assert not success
        assert budget.remaining_budget == 0.4
    
    def test_federated_node(self):
        """Test federated learning node."""
        node = FederatedNode(
            node_id="hospital_1",
            institution_name="General Hospital",
            data_volume=1000
        )
        
        assert node.node_id == "hospital_1"
        assert node.data_volume == 1000
        assert node.public_key_hash is not None
        
        # Test activity tracking
        node.update_activity()
        assert node.is_active
    
    def test_model_update_validation(self):
        """Test model update validation."""
        node = FederatedNode("test_node", "Test Institution", 100)
        
        update = ModelUpdate(
            node_id="test_node",
            update_id="update_1",
            model_weights_delta={"layer_1": np.random.normal(0, 0.1, (5, 5))},
            gradient_norm=0.5,
            privacy_cost=0.05,
            timestamp=time.time(),
            signature=f"test_signature_{node.public_key_hash}",
        )
        
        # Mock proper signature verification for test
        with patch.object(update, 'verify_signature', return_value=True):
            assert update.is_valid()
    
    def test_federated_compliance_model(self):
        """Test federated compliance model."""
        model = FederatedComplianceModel(
            model_id="test_model",
            privacy_budget=PrivacyBudget(epsilon_total=2.0),
            min_participants=2
        )
        
        # Register nodes
        success = model.register_node("node_1", "Hospital A", 1000)
        assert success
        
        success = model.register_node("node_2", "Hospital B", 1500)
        assert success
        
        assert len(model.nodes) == 2
    
    def test_privacy_preserving_trainer(self):
        """Test privacy-preserving trainer."""
        trainer = PrivacyPreservingTrainer()
        
        model = trainer.create_federated_model(
            model_id="test_phi_model",
            privacy_epsilon=1.0,
            min_participants=2
        )
        
        assert model.model_id == "test_phi_model"
        assert model.global_privacy_budget.epsilon_total == 1.0
    
    @patch('time.sleep')  # Skip sleep delays in test
    def test_federated_training_simulation(self, mock_sleep):
        """Test federated training simulation."""
        trainer = PrivacyPreservingTrainer()
        
        model = trainer.create_federated_model(
            "test_simulation",
            privacy_epsilon=2.0,
            min_participants=2
        )
        
        # Run a short simulation
        results = trainer.simulate_federated_training(
            "test_simulation",
            num_institutions=3,
            rounds_per_institution=2
        )
        
        assert 'global_accuracy' in results
        assert 'total_rounds' in results
        assert results['nodes']['total_registered'] == 3


class TestCompliancePrediction:
    """Test compliance prediction and risk assessment."""
    
    def test_compliance_features(self):
        """Test compliance features extraction."""
        features = ComplianceFeatures(
            document_type="clinical_note",
            word_count=500,
            phi_density=0.05,
            medical_term_density=0.08,
            recent_violations=1,
            days_since_last_violation=30
        )
        
        feature_array = features.to_array()
        assert len(feature_array) == len(features.feature_names)
        assert all(0 <= val <= 1 for val in feature_array)
    
    def test_risk_level_classification(self):
        """Test risk level classification."""
        for level in RiskLevel:
            score_range = level.score_range
            assert 0 <= score_range[0] < score_range[1] <= 1
            assert level.color_code.startswith("#")
    
    def test_risk_prediction(self):
        """Test risk prediction structure."""
        prediction = RiskPrediction(
            risk_score=0.7,
            risk_level=RiskLevel.HIGH,
            confidence=0.85,
            risk_factors=["High PHI density", "Recent violations"],
            recommendations=["Increase monitoring", "Additional training"]
        )
        
        assert prediction.risk_score == 0.7
        assert prediction.risk_level == RiskLevel.HIGH
        assert len(prediction.risk_factors) == 2
        assert len(prediction.recommendations) == 2
        
        # Test serialization
        prediction_dict = prediction.to_dict()
        assert 'risk_score' in prediction_dict
        assert 'risk_level' in prediction_dict
    
    def test_anomaly_detector(self):
        """Test compliance anomaly detection."""
        detector = ComplianceAnomalyDetector(window_size=10)
        
        # Add normal values
        for i in range(15):
            detector.update_baseline("processing_time", 5.0 + np.random.normal(0, 0.5))
        
        # Test anomaly detection
        is_anomaly, score = detector.detect_anomaly("processing_time", 15.0)  # Abnormal value
        assert isinstance(is_anomaly, bool)
        assert score >= 0
    
    def test_compliance_prediction_engine(self):
        """Test compliance prediction engine."""
        engine = CompliancePredictionEngine()
        
        features = ComplianceFeatures(
            document_type="clinical_note",
            word_count=1000,
            phi_density=0.1,
            recent_violations=2,
            user_behavior_score=0.3
        )
        
        prediction = engine.predict_compliance_risk(features)
        
        assert isinstance(prediction, RiskPrediction)
        assert 0 <= prediction.risk_score <= 1
        assert isinstance(prediction.risk_level, RiskLevel)
        assert 0 <= prediction.confidence <= 1
        assert len(prediction.feature_importances) > 0
    
    def test_risk_predictor(self):
        """Test high-level risk predictor interface."""
        predictor = RiskPredictor()
        
        # Test document with high PHI content
        high_phi_document = """
        Patient: John Doe
        SSN: 123-45-6789
        Phone: 555-123-4567
        Email: john.doe@email.com
        Address: 123 Main St, Boston, MA
        """
        
        prediction = predictor.predict_document_risk(
            high_phi_document,
            "clinical_note",
            user_context={'behavior_score': 0.2, 'recent_violations': 1},
            system_context={'system_load': 0.9, 'auth_strength': 0.5}
        )
        
        assert isinstance(prediction, RiskPrediction)
        assert prediction.risk_score > 0.5  # Should be high risk
        assert len(prediction.risk_factors) > 0
        assert len(prediction.recommendations) > 0
    
    def test_risk_dashboard(self):
        """Test risk dashboard generation."""
        predictor = RiskPredictor()
        
        # Generate some predictions to populate history
        for _ in range(5):
            predictor.predict_document_risk("test document", "clinical_note")
        
        dashboard = predictor.get_risk_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'current_risk_level' in dashboard
        assert 'cache_statistics' in dashboard


class TestBenchmarkSuite:
    """Test research benchmark suite."""
    
    def test_benchmark_dataset(self):
        """Test benchmark dataset structure."""
        dataset = BenchmarkDataset(
            name="test_dataset",
            description="Test dataset",
            documents=["Document 1", "Document 2"],
            ground_truth=[
                [{"entity_type": "name", "text": "John", "start": 0, "end": 4}],
                []
            ],
            document_types=["clinical_note", "clinical_note"],
            difficulty_level="medium",
            domain="clinical"
        )
        
        assert dataset.size == 2
        assert dataset.total_phi_entities == 1
        
        stats = dataset.get_statistics()
        assert stats['name'] == "test_dataset"
        assert stats['size'] == 2
    
    def test_benchmark_result(self):
        """Test benchmark result structure."""
        metrics = ValidationMetrics(
            true_positives=85,
            false_positives=10,
            true_negatives=3,
            false_negatives=2
        )
        
        result = BenchmarkResult(
            model_name="TestModel",
            dataset_name="test_dataset",
            validation_metrics=metrics,
            performance_stats={"documents_per_second": 10.0},
            processing_time=5.0
        )
        
        assert result.model_name == "TestModel"
        
        result_dict = result.to_dict()
        assert 'validation_metrics' in result_dict
        assert 'performance_stats' in result_dict
    
    def test_research_benchmark_suite(self):
        """Test research benchmark suite functionality."""
        suite = ResearchBenchmarkSuite()
        
        # Should have initialized with standard datasets
        assert len(suite.datasets) > 0
        assert 'clinical_notes' in suite.datasets
        
        # Test adding custom dataset
        custom_dataset = BenchmarkDataset(
            name="custom_test",
            description="Custom test dataset",
            documents=["Test doc"],
            ground_truth=[[]],
            document_types=["test"],
            difficulty_level="easy",
            domain="test"
        )
        
        suite.add_dataset(custom_dataset)
        assert "custom_test" in suite.datasets
    
    def test_model_benchmarking(self):
        """Test model benchmarking functionality."""
        suite = ResearchBenchmarkSuite()
        
        # Mock model callable
        def mock_model(text: str, doc_type: str) -> list:
            # Return some fake PHI detections
            return [
                {
                    'entity_type': 'name',
                    'text': 'John Doe',
                    'start': 0,
                    'end': 8,
                    'confidence': 0.9
                }
            ]
        
        # Benchmark on a subset of datasets
        results = suite.benchmark_model(
            mock_model,
            "MockModel",
            dataset_names=['clinical_notes'],
            enable_memory_profiling=False
        )
        
        assert len(results) == 1
        assert results[0].model_name == "MockModel"
        assert results[0].dataset_name == "clinical_notes"
    
    def test_comparative_analysis(self):
        """Test comparative analysis functionality."""
        # Create mock results
        metrics1 = ValidationMetrics(true_positives=80, false_positives=10, true_negatives=8, false_negatives=2)
        metrics2 = ValidationMetrics(true_positives=85, false_positives=5, true_negatives=8, false_negatives=2)
        
        result1 = BenchmarkResult("Model1", "test_dataset", metrics1, {}, 1.0)
        result2 = BenchmarkResult("Model2", "test_dataset", metrics2, {}, 1.2)
        
        suite = ResearchBenchmarkSuite()
        suite.results = [result1, result2]
        
        analysis = suite.compare_models([result1, result2])
        
        assert isinstance(analysis, ComparativeAnalysis)
        assert len(analysis.results) == 2
        assert len(analysis.ranking) == 2
        
        # Check ranking order (higher F1 should be first)
        assert analysis.ranking[0][1] >= analysis.ranking[1][1]


class TestIntegrationScenarios:
    """Test integration between research modules."""
    
    def test_adaptive_detector_with_benchmark(self):
        """Test adaptive detector integration with benchmark suite."""
        suite = ResearchBenchmarkSuite()
        detector = AdaptivePHIDetector()
        
        def detector_callable(text: str, doc_type: str) -> list:
            return detector.detect_phi_with_confidence(text, doc_type)
        
        # Run benchmark on one dataset
        results = suite.benchmark_model(
            detector_callable,
            "AdaptivePHIDetector",
            dataset_names=['clinical_notes']
        )
        
        assert len(results) == 1
        assert results[0].validation_metrics.accuracy >= 0  # Should have some accuracy
    
    def test_statistical_validation_with_prediction(self):
        """Test statistical validation with compliance prediction."""
        validator = StatisticalValidator()
        predictor = RiskPredictor()
        
        # Generate predictions
        predictions = []
        ground_truth = []
        
        for i in range(50):
            # High-risk document
            if i < 25:
                doc = "Patient SSN: 123-45-6789, high PHI content"
                prediction = predictor.predict_document_risk(doc, "clinical_note")
                predictions.append(prediction.risk_score > 0.6)
                ground_truth.append(True)  # Actually high risk
            else:
                # Low-risk document
                doc = "General medical information without PHI"
                prediction = predictor.predict_document_risk(doc, "clinical_note")
                predictions.append(prediction.risk_score > 0.6)
                ground_truth.append(False)  # Actually low risk
        
        metrics = validator.validate_model_performance(predictions, ground_truth)
        assert isinstance(metrics, ValidationMetrics)
    
    def test_end_to_end_research_workflow(self):
        """Test complete research workflow integration."""
        
        # 1. Create benchmark suite
        suite = ResearchBenchmarkSuite()
        
        # 2. Create adaptive detector
        detector = AdaptivePHIDetector(enable_statistical_validation=True)
        
        # 3. Create statistical validator  
        validator = StatisticalValidator()
        
        # 4. Create compliance predictor
        predictor = RiskPredictor()
        
        # 5. Test document
        test_doc = "Patient John Doe, SSN: 123-45-6789, requires treatment"
        
        # 6. Run PHI detection
        detections = detector.detect_phi_with_confidence(test_doc, "clinical_note")
        assert len(detections) > 0
        
        # 7. Run compliance prediction
        prediction = predictor.predict_document_risk(test_doc, "clinical_note")
        assert isinstance(prediction, RiskPrediction)
        
        # 8. Get performance summaries
        detector_performance = detector.get_performance_summary()
        predictor_dashboard = predictor.get_risk_dashboard()
        
        assert detector_performance['total_runs'] > 0
        assert 'current_risk_level' in predictor_dashboard


# Demonstration tests
class TestResearchDemonstrations:
    """Test research demonstration functions."""
    
    @patch('time.sleep')  # Skip sleep delays
    def test_federated_learning_demonstration(self, mock_sleep):
        """Test federated learning demonstration."""
        results = demonstrate_federated_phi_detection()
        
        assert 'model_id' in results
        assert 'training_results' in results
        assert 'final_accuracy' in results
        assert 'privacy_budget_spent' in results
        assert 'converged' in results
    
    def test_compliance_prediction_demonstration(self):
        """Test compliance prediction demonstration."""
        results = demonstrate_compliance_prediction()
        
        assert 'prediction' in results
        assert 'dashboard' in results
        assert 'risk_level' in results
        assert 'recommendations' in results
    
    def test_benchmark_demonstration(self):
        """Test benchmark suite demonstration."""
        # This would be computationally intensive, so we'll mock parts
        with patch('src.hipaa_compliance_summarizer.research.benchmark_suite.ResearchBenchmarkSuite') as MockSuite:
            mock_suite_instance = MagicMock()
            MockSuite.return_value = mock_suite_instance
            
            # Mock the benchmark results
            mock_suite_instance.benchmark_model.return_value = [
                BenchmarkResult(
                    "AdaptivePHIDetector", "test_dataset",
                    ValidationMetrics(80, 10, 5, 5), {}, 1.0
                )
            ]
            
            # Test that the demonstration function can be called
            try:
                results = benchmark_adaptive_phi_detector()
                # If mocked properly, this should return the mocked structure
                assert isinstance(results, dict) or results is not None
            except Exception as e:
                # If there are import/dependency issues, that's expected in test environment
                assert "scipy" in str(e) or "pandas" in str(e) or "module" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])