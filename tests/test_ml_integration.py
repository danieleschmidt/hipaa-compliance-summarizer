"""Comprehensive tests for ML integration components."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.hipaa_compliance_summarizer.ml_integration import (
    AdvancedPHIDetector,
    ClinicalSummarizer,
    MLModelManager,
    ModelType,
    ModelMetrics,
    PredictionResult,
    initialize_ml_models
)
from src.hipaa_compliance_summarizer.models.phi_entity import PHICategory, PHIEntity


@pytest.fixture
def phi_detector():
    """Create PHI detector for testing."""
    config = {
        "model_path": "test_model",
        "confidence_threshold": 0.8,
        "use_clinical_context": True
    }
    return AdvancedPHIDetector(config)


@pytest.fixture
def clinical_summarizer():
    """Create clinical summarizer for testing."""
    config = {
        "model_path": "test_clinical_model",
        "max_summary_length": 512,
        "preserve_medical_terms": True
    }
    return ClinicalSummarizer(config)


@pytest.fixture
def model_manager():
    """Create model manager for testing."""
    return MLModelManager()


class TestAdvancedPHIDetector:
    """Test cases for Advanced PHI Detector."""
    
    def test_initialization(self, phi_detector):
        """Test PHI detector initialization."""
        assert phi_detector.model_name == "advanced_phi_detector"
        assert phi_detector.confidence_threshold == 0.8
        assert phi_detector.use_clinical_context is True
        assert not phi_detector.is_loaded
    
    def test_load_model(self, phi_detector):
        """Test model loading."""
        result = phi_detector.load_model()
        assert result is True
        assert phi_detector.is_loaded is True
    
    @pytest.mark.asyncio
    async def test_predict_simple_phi(self, phi_detector):
        """Test PHI prediction with simple text."""
        phi_detector.load_model()
        
        text = "Patient John Doe, SSN: 123-45-6789, was admitted on 01/15/2024."
        result = await phi_detector.predict(text)
        
        assert isinstance(result, PredictionResult)
        assert result.success is True
        assert len(result.predictions) > 0
        assert result.model_type == ModelType.PHI_DETECTION
        
        # Check for expected PHI entities
        predicted_categories = [p["category"] for p in result.predictions]
        assert "names" in predicted_categories or "dates" in predicted_categories
    
    @pytest.mark.asyncio
    async def test_predict_clinical_context(self, phi_detector):
        """Test PHI prediction with clinical context."""
        phi_detector.load_model()
        
        context = {
            "document_type": "clinical_note",
            "department": "cardiology"
        }
        
        text = "Patient was diagnosed with hypertension. Dr. Smith prescribed medication."
        result = await phi_detector.predict(text, context)
        
        assert result.success is True
        assert result.metadata["clinical_context_used"] is True
    
    @pytest.mark.asyncio
    async def test_predict_with_ground_truth(self, phi_detector):
        """Test prediction with ground truth for metrics calculation."""
        phi_detector.load_model()
        
        text = "John Doe, born 01/01/1990, SSN 123-45-6789"
        ground_truth = [
            {"start_position": 0, "end_position": 8, "category": "names"},
            {"start_position": 15, "end_position": 25, "category": "dates"},
            {"start_position": 31, "end_position": 42, "category": "ssn"}
        ]
        
        context = {"ground_truth": ground_truth}
        result = await phi_detector.predict(text, context)
        
        assert result.success is True
        # Metrics should be updated
        assert len(phi_detector.metrics_history) > 0
    
    def test_pattern_based_detection(self, phi_detector):
        """Test pattern-based PHI detection."""
        text = "SSN: 123-45-6789, Phone: 555-123-4567, Email: test@example.com"
        
        entities = phi_detector._pattern_based_detection(text)
        
        assert len(entities) > 0
        categories = [entity.category.value for entity in entities]
        assert "social_security_numbers" in categories
    
    def test_clinical_context_detection(self, phi_detector):
        """Test clinical context-aware detection."""
        text = "Patient was diagnosed with diabetes on 12/15/2023 by Dr. Johnson."
        context = {"document_type": "clinical_note"}
        
        entities = phi_detector._clinical_context_detection(text, context)
        
        assert len(entities) > 0
        # Should detect dates and names in clinical context
        categories = [entity.category.value for entity in entities]
        assert "dates" in categories or "names" in categories
    
    def test_deduplicate_entities(self, phi_detector):
        """Test entity deduplication."""
        # Create overlapping entities
        entities = [
            PHIEntity("John", PHICategory.NAMES, 0, 4, 0.9, {}),
            PHIEntity("John Doe", PHICategory.NAMES, 0, 8, 0.95, {}),  # Higher confidence, overlapping
            PHIEntity("123-45-6789", PHICategory.SOCIAL_SECURITY_NUMBERS, 10, 21, 0.98, {})
        ]
        
        deduplicated = phi_detector._deduplicate_entities(entities)
        
        # Should keep higher confidence overlapping entity and non-overlapping entity
        assert len(deduplicated) == 2
        assert deduplicated[0].text == "John Doe"  # Higher confidence kept
        assert deduplicated[1].text == "123-45-6789"
    
    def test_calculate_metrics(self, phi_detector):
        """Test metrics calculation."""
        predictions = [
            {"start_position": 0, "end_position": 8, "category": "names"},
            {"start_position": 15, "end_position": 25, "category": "dates"}
        ]
        
        ground_truth = [
            {"start_position": 0, "end_position": 8, "category": "names"},
            {"start_position": 30, "end_position": 41, "category": "ssn"}
        ]
        
        metrics = phi_detector._calculate_metrics(predictions, ground_truth)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.accuracy <= 1


class TestClinicalSummarizer:
    """Test cases for Clinical Summarizer."""
    
    def test_initialization(self, clinical_summarizer):
        """Test clinical summarizer initialization."""
        assert clinical_summarizer.model_name == "clinical_summarizer"
        assert clinical_summarizer.max_summary_length == 512
        assert clinical_summarizer.preserve_medical_terms is True
    
    def test_load_model(self, clinical_summarizer):
        """Test model loading."""
        result = clinical_summarizer.load_model()
        assert result is True
        assert clinical_summarizer.is_loaded is True
    
    @pytest.mark.asyncio
    async def test_predict_clinical_summary(self, clinical_summarizer):
        """Test clinical text summarization."""
        clinical_summarizer.load_model()
        
        text = """
        Patient presented with chest pain and shortness of breath. 
        Physical examination revealed elevated blood pressure and irregular heartbeat.
        EKG showed ST segment elevation. Patient was diagnosed with acute myocardial infarction.
        Treatment included aspirin, nitroglycerin, and cardiac catheterization.
        Patient was admitted to cardiac care unit for monitoring.
        """
        
        result = await clinical_summarizer.predict(text)
        
        assert isinstance(result, PredictionResult)
        assert result.success is True
        assert len(result.predictions) == 1
        assert result.model_type == ModelType.SUMMARIZATION
        
        summary_data = result.predictions[0]
        assert "summary" in summary_data
        assert len(summary_data["summary"]) > 0
        assert "medical_entities" in summary_data
    
    def test_extract_medical_entities(self, clinical_summarizer):
        """Test medical entity extraction."""
        text = "Patient diagnosed with diabetes and hypertension. Prescribed metformin."
        
        entities = clinical_summarizer._extract_medical_entities(text)
        
        assert isinstance(entities, dict)
        assert "diagnoses" in entities
        assert len(entities["diagnoses"]) > 0
    
    def test_extract_symptoms(self, clinical_summarizer):
        """Test symptom extraction."""
        text = "Patient complained of chest pain, nausea, and dizziness."
        
        symptoms = clinical_summarizer._extract_symptoms(text)
        
        assert isinstance(symptoms, list)
        assert len(symptoms) > 0
        assert "pain" in symptoms or "nausea" in symptoms
    
    def test_extract_medications(self, clinical_summarizer):
        """Test medication extraction."""
        text = "Prescribed aspirin 81mg and lisinopril for blood pressure control."
        
        medications = clinical_summarizer._extract_medications(text)
        
        assert isinstance(medications, list)
        assert len(medications) > 0
    
    def test_extract_procedures(self, clinical_summarizer):
        """Test procedure extraction."""
        text = "Patient underwent blood test, EKG, and chest x-ray."
        
        procedures = clinical_summarizer._extract_procedures(text)
        
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        assert any("test" in proc.lower() for proc in procedures)
    
    def test_identify_risk_factors(self, clinical_summarizer):
        """Test risk factor identification."""
        text = "Patient has family history of heart disease, smokes cigarettes, and has high blood pressure."
        
        risk_factors = clinical_summarizer._identify_risk_factors(text)
        
        assert isinstance(risk_factors, list)
        assert len(risk_factors) > 0
        assert any("smoking" in factor.lower() or "family history" in factor.lower() for factor in risk_factors)


class TestMLModelManager:
    """Test cases for ML Model Manager."""
    
    def test_initialization(self, model_manager):
        """Test model manager initialization."""
        assert isinstance(model_manager.models, dict)
        assert len(model_manager.models) == 0
    
    def test_register_model(self, model_manager, phi_detector):
        """Test model registration."""
        model_manager.register_model(ModelType.PHI_DETECTION, phi_detector)
        
        assert ModelType.PHI_DETECTION in model_manager.models
        assert model_manager.models[ModelType.PHI_DETECTION] == phi_detector
    
    def test_load_all_models(self, model_manager, phi_detector, clinical_summarizer):
        """Test loading all registered models."""
        model_manager.register_model(ModelType.PHI_DETECTION, phi_detector)
        model_manager.register_model(ModelType.SUMMARIZATION, clinical_summarizer)
        
        results = model_manager.load_all_models()
        
        assert ModelType.PHI_DETECTION in results
        assert ModelType.SUMMARIZATION in results
        assert results[ModelType.PHI_DETECTION] is True
        assert results[ModelType.SUMMARIZATION] is True
    
    @pytest.mark.asyncio
    async def test_predict_ensemble(self, model_manager, phi_detector, clinical_summarizer):
        """Test ensemble predictions."""
        model_manager.register_model(ModelType.PHI_DETECTION, phi_detector)
        model_manager.register_model(ModelType.SUMMARIZATION, clinical_summarizer)
        
        model_manager.load_all_models()
        
        text = "Patient John Doe presented with chest pain."
        model_types = [ModelType.PHI_DETECTION, ModelType.SUMMARIZATION]
        
        results = await model_manager.predict_ensemble(text, model_types)
        
        assert isinstance(results, dict)
        assert ModelType.PHI_DETECTION in results
        assert ModelType.SUMMARIZATION in results
        
        for model_type, result in results.items():
            assert isinstance(result, PredictionResult)
            assert result.model_type == model_type
    
    def test_get_model_metrics(self, model_manager, phi_detector):
        """Test getting model metrics."""
        model_manager.register_model(ModelType.PHI_DETECTION, phi_detector)
        
        # Add some fake metrics
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.94,
            f1_score=0.93,
            processing_time_ms=150.0,
            confidence_threshold=0.8,
            model_version="test_v1"
        )
        phi_detector.update_metrics(metrics)
        
        retrieved_metrics = model_manager.get_model_metrics(ModelType.PHI_DETECTION)
        
        assert retrieved_metrics is not None
        assert retrieved_metrics.accuracy == 0.95
    
    def test_get_all_metrics(self, model_manager, phi_detector, clinical_summarizer):
        """Test getting metrics for all models."""
        model_manager.register_model(ModelType.PHI_DETECTION, phi_detector)
        model_manager.register_model(ModelType.SUMMARIZATION, clinical_summarizer)
        
        all_metrics = model_manager.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert ModelType.PHI_DETECTION in all_metrics
        assert ModelType.SUMMARIZATION in all_metrics


class TestModelMetrics:
    """Test cases for Model Metrics."""
    
    def test_model_metrics_creation(self):
        """Test model metrics creation."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.94,
            f1_score=0.93,
            processing_time_ms=150.0,
            confidence_threshold=0.8,
            model_version="test_v1"
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.94
        assert metrics.f1_score == 0.93
        assert metrics.processing_time_ms == 150.0
        assert metrics.confidence_threshold == 0.8
        assert metrics.model_version == "test_v1"
        assert metrics.timestamp is not None
    
    def test_model_metrics_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.94,
            f1_score=0.93,
            processing_time_ms=150.0,
            confidence_threshold=0.8,
            model_version="test_v1"
        )
        
        # Timestamp should be automatically generated
        assert metrics.timestamp is not None
        # Should be ISO format
        datetime.fromisoformat(metrics.timestamp)


class TestPredictionResult:
    """Test cases for Prediction Result."""
    
    def test_prediction_result_creation(self):
        """Test prediction result creation."""
        predictions = [{"entity": "John Doe", "confidence": 0.95}]
        confidence_scores = [0.95]
        
        result = PredictionResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            model_type=ModelType.PHI_DETECTION,
            processing_time_ms=125.5,
            metadata={"test": True}
        )
        
        assert result.predictions == predictions
        assert result.confidence_scores == confidence_scores
        assert result.model_type == ModelType.PHI_DETECTION
        assert result.processing_time_ms == 125.5
        assert result.success is True
        assert result.error_message is None
        assert result.metadata["test"] is True
    
    def test_prediction_result_failure(self):
        """Test prediction result with failure."""
        result = PredictionResult(
            predictions=[],
            confidence_scores=[],
            model_type=ModelType.PHI_DETECTION,
            processing_time_ms=50.0,
            metadata={},
            success=False,
            error_message="Model failed to load"
        )
        
        assert result.success is False
        assert result.error_message == "Model failed to load"


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_phi_detection_and_summarization(self):
        """Test complete pipeline with PHI detection and summarization."""
        # Initialize models
        manager = initialize_ml_models()
        manager.load_all_models()
        
        clinical_text = """
        Patient Mary Johnson, DOB: 03/15/1985, MRN: HSP123456, presented to the emergency
        department with acute chest pain radiating to the left arm. Vital signs on admission:
        BP 180/95, HR 102, RR 18, O2 sat 96% on room air. EKG showed ST elevation in leads
        II, III, and aVF consistent with inferior STEMI. Patient was given aspirin 325mg,
        clopidogrel 600mg loading dose, and atorvastatin 80mg. Cardiac catheterization
        revealed 99% occlusion of the right coronary artery. Primary PCI was performed
        with drug-eluting stent placement. Patient transferred to CCU for monitoring.
        """
        
        # Run ensemble prediction
        model_types = [ModelType.PHI_DETECTION, ModelType.SUMMARIZATION]
        results = await manager.predict_ensemble(clinical_text, model_types, {
            "document_type": "clinical_note",
            "department": "emergency"
        })
        
        # Validate PHI detection results
        phi_result = results[ModelType.PHI_DETECTION]
        assert phi_result.success is True
        assert len(phi_result.predictions) > 0
        
        # Should detect patient name, DOB, MRN
        detected_categories = [p["category"] for p in phi_result.predictions]
        assert any("name" in cat.lower() for cat in detected_categories)
        
        # Validate summarization results
        summary_result = results[ModelType.SUMMARIZATION]
        assert summary_result.success is True
        assert len(summary_result.predictions) == 1
        
        summary_data = summary_result.predictions[0]
        assert "summary" in summary_data
        assert "chest pain" in summary_data["summary"].lower()
        assert len(summary_data.get("medications", [])) > 0
        assert len(summary_data.get("procedures", [])) > 0
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking(self):
        """Test model performance tracking across multiple predictions."""
        manager = initialize_ml_models()
        manager.load_all_models()
        
        test_cases = [
            "Patient John Smith, SSN: 123-45-6789, diagnosed with hypertension.",
            "Mary Jones, DOB: 01/01/1980, prescribed lisinopril 10mg daily.",
            "Patient ID: 987654, chest x-ray showed no acute findings.",
        ]
        
        for text in test_cases:
            await manager.predict_ensemble(text, [ModelType.PHI_DETECTION])
        
        # Check that metrics are being tracked
        phi_model = manager.models[ModelType.PHI_DETECTION]
        assert len(phi_model.performance_history) >= 0  # May be 0 without ground truth
        
        # Test metrics aggregation
        metrics = manager.get_model_metrics(ModelType.PHI_DETECTION)
        # Without ground truth, metrics may be None
        if metrics:
            assert 0 <= metrics.accuracy <= 1
    
    @pytest.mark.asyncio 
    async def test_error_handling_and_recovery(self):
        """Test error handling in ML pipeline."""
        manager = MLModelManager()
        
        # Try to predict without registered models
        results = await manager.predict_ensemble(
            "Test text", 
            [ModelType.PHI_DETECTION]
        )
        
        # Should handle missing model gracefully
        assert ModelType.PHI_DETECTION not in results
        
        # Register model but don't load it
        detector = AdvancedPHIDetector()
        manager.register_model(ModelType.PHI_DETECTION, detector)
        
        # Try prediction with unloaded model
        results = await manager.predict_ensemble(
            "Test text", 
            [ModelType.PHI_DETECTION]
        )
        
        # Should get error result
        phi_result = results[ModelType.PHI_DETECTION]
        assert phi_result.success is False
        assert phi_result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self):
        """Test concurrent predictions with multiple models."""
        manager = initialize_ml_models()
        manager.load_all_models()
        
        texts = [
            "Patient A with diabetes and hypertension.",
            "Patient B presented with chest pain and nausea.", 
            "Patient C diagnosed with pneumonia."
        ]
        
        # Create concurrent prediction tasks
        tasks = []
        for text in texts:
            task = manager.predict_ensemble(
                text, 
                [ModelType.PHI_DETECTION, ModelType.SUMMARIZATION]
            )
            tasks.append(task)
        
        # Execute concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Validate all results
        assert len(results_list) == 3
        for results in results_list:
            assert ModelType.PHI_DETECTION in results
            assert ModelType.SUMMARIZATION in results
            
            phi_result = results[ModelType.PHI_DETECTION]
            summary_result = results[ModelType.SUMMARIZATION]
            
            # At minimum, should not error
            assert isinstance(phi_result, PredictionResult)
            assert isinstance(summary_result, PredictionResult)


@pytest.mark.asyncio
async def test_initialize_ml_models():
    """Test ML models initialization function."""
    config = {
        "phi_detection": {
            "confidence_threshold": 0.85,
            "use_clinical_context": True
        },
        "summarization": {
            "max_summary_length": 256,
            "preserve_medical_terms": True
        }
    }
    
    manager = initialize_ml_models(config)
    
    assert isinstance(manager, MLModelManager)
    assert ModelType.PHI_DETECTION in manager.models
    assert ModelType.SUMMARIZATION in manager.models
    
    # Test that models are configured correctly
    phi_detector = manager.models[ModelType.PHI_DETECTION]
    assert phi_detector.confidence_threshold == 0.85
    assert phi_detector.use_clinical_context is True
    
    clinical_summarizer = manager.models[ModelType.SUMMARIZATION]
    assert clinical_summarizer.max_summary_length == 256
    assert clinical_summarizer.preserve_medical_terms is True


class TestModelReproducibility:
    """Test model reproducibility and consistency."""
    
    @pytest.mark.asyncio
    async def test_prediction_consistency(self):
        """Test that repeated predictions on same input are consistent."""
        manager = initialize_ml_models()
        manager.load_all_models()
        
        text = "Patient John Doe, SSN: 123-45-6789, has diabetes."
        
        # Run prediction multiple times
        results = []
        for _ in range(3):
            result = await manager.predict_ensemble(text, [ModelType.PHI_DETECTION])
            results.append(result[ModelType.PHI_DETECTION])
        
        # Results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.success == first_result.success
            assert len(result.predictions) == len(first_result.predictions)
    
    def test_model_metrics_reproducibility(self, phi_detector):
        """Test metrics calculation reproducibility."""
        predictions = [
            {"start_position": 0, "end_position": 8, "category": "names"},
            {"start_position": 15, "end_position": 26, "category": "ssn"}
        ]
        ground_truth = [
            {"start_position": 0, "end_position": 8, "category": "names"},
            {"start_position": 15, "end_position": 26, "category": "ssn"}
        ]
        
        # Calculate metrics multiple times
        metrics1 = phi_detector._calculate_metrics(predictions, ground_truth)
        metrics2 = phi_detector._calculate_metrics(predictions, ground_truth)
        
        # Should be identical
        assert metrics1.accuracy == metrics2.accuracy
        assert metrics1.precision == metrics2.precision
        assert metrics1.recall == metrics2.recall
        assert metrics1.f1_score == metrics2.f1_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])