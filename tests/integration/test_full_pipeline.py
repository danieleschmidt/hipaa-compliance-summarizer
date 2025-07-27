"""Integration tests for the complete HIPAA compliance pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.processor import HIPAAProcessor


class TestFullPipeline:
    """Test the complete document processing pipeline."""

    def test_single_document_pipeline(self, temp_config_file, sample_documents):
        """Test processing a single document through the complete pipeline."""
        processor = HIPAAProcessor(config_path=temp_config_file)

        result = processor.process_document(str(sample_documents["clinical_note"]))

        assert result is not None
        assert "summary" in result
        assert "compliance_score" in result
        assert "phi_detected_count" in result
        assert "audit_trail" in result

        # Verify PHI was detected and processed
        assert result["phi_detected_count"] > 0
        assert result["compliance_score"] > 0.8

    def test_batch_processing_pipeline(
        self, temp_config_file, sample_documents, temp_dir
    ):
        """Test batch processing multiple documents."""
        processor = BatchProcessor(config_path=temp_config_file)

        # Create input and output directories
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Copy sample documents to input directory
        for name, doc_path in sample_documents.items():
            new_path = input_dir / f"{name}.txt"
            new_path.write_text(doc_path.read_text())

        # Process the batch
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            compliance_level="standard",
            generate_summaries=True,
        )

        assert len(results) == len(sample_documents)
        assert all(result["status"] == "success" for result in results)

        # Verify output files were created
        output_files = list(output_dir.glob("*.txt"))
        assert len(output_files) == len(sample_documents)

    def test_compliance_reporting_pipeline(self, temp_config_file, sample_documents):
        """Test the compliance reporting functionality."""
        from hipaa_compliance_summarizer.reporting import ComplianceReporter

        reporter = ComplianceReporter(config_path=temp_config_file)

        # Process some documents first
        processor = HIPAAProcessor(config_path=temp_config_file)
        for doc_path in sample_documents.values():
            processor.process_document(str(doc_path))

        # Generate compliance report
        report = reporter.generate_report(
            period="2024-Q1", documents_processed=len(sample_documents)
        )

        assert "overall_compliance" in report
        assert "violations_detected" in report
        assert "recommendations" in report
        assert report["overall_compliance"] >= 0.0
        assert report["overall_compliance"] <= 1.0

    def test_error_handling_pipeline(self, temp_config_file, temp_dir):
        """Test pipeline behavior with invalid or corrupted documents."""
        processor = HIPAAProcessor(config_path=temp_config_file)

        # Create an invalid document
        invalid_doc = Path(temp_dir) / "invalid.txt"
        invalid_doc.write_bytes(b"\x00\x01\x02\x03\x04")  # Binary data

        # Process should handle the error gracefully
        result = processor.process_document(str(invalid_doc))

        assert result is not None
        assert "error" in result or "status" in result

    def test_performance_pipeline(
        self, temp_config_file, performance_test_documents, temp_dir
    ):
        """Test pipeline performance with multiple documents."""
        import time

        processor = BatchProcessor(config_path=temp_config_file)

        # Create input and output directories
        input_dir = Path(temp_dir) / "perf_input"
        output_dir = Path(temp_dir) / "perf_output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Copy test documents
        for i, doc_content in enumerate(performance_test_documents[:10]):  # Test with 10
            doc_path = input_dir / f"perf_doc_{i:03d}.txt"
            doc_path.write_text(Path(doc_content).read_text())

        # Measure processing time
        start_time = time.time()
        results = processor.process_directory(
            str(input_dir), str(output_dir), show_progress=False
        )
        end_time = time.time()

        processing_time = end_time - start_time
        docs_per_second = len(results) / processing_time

        # Performance assertions
        assert len(results) == 10
        assert processing_time < 60  # Should complete within 60 seconds
        assert docs_per_second > 0.1  # At least 0.1 docs/second


class TestAPIIntegration:
    """Test API endpoints and external integrations."""

    @pytest.mark.skipif(
        not pytest.config.getoption("--integration"),
        reason="Integration tests require --integration flag",
    )
    def test_health_check_endpoint(self, integration_test_config):
        """Test health check endpoint availability."""
        import requests

        for endpoint in integration_test_config["test_endpoints"]:
            try:
                response = requests.get(
                    endpoint, timeout=integration_test_config["timeout"]
                )
                assert response.status_code == 200
                assert "status" in response.json()
            except requests.exceptions.RequestException:
                pytest.skip("Health check endpoint not available")

    def test_cache_performance_integration(self, temp_config_file, sample_documents):
        """Test caching functionality in realistic scenarios."""
        from hipaa_compliance_summarizer.phi import PHICache

        cache = PHICache(max_size=100)
        processor = HIPAAProcessor(config_path=temp_config_file, phi_cache=cache)

        # Process the same document multiple times
        doc_path = str(sample_documents["clinical_note"])

        # First processing
        result1 = processor.process_document(doc_path)

        # Second processing (should use cache)
        result2 = processor.process_document(doc_path)

        # Results should be identical
        assert result1["phi_detected_count"] == result2["phi_detected_count"]
        assert result1["compliance_score"] == result2["compliance_score"]

        # Cache should show hits
        cache_stats = cache.get_stats()
        assert cache_stats["hits"] > 0

    def test_monitoring_integration(self, temp_config_file, sample_documents):
        """Test monitoring and metrics collection."""
        from hipaa_compliance_summarizer.monitoring import MetricsCollector

        metrics = MetricsCollector()
        processor = HIPAAProcessor(
            config_path=temp_config_file, metrics_collector=metrics
        )

        # Process documents and collect metrics
        for doc_path in sample_documents.values():
            processor.process_document(str(doc_path))

        # Verify metrics were collected
        collected_metrics = metrics.get_metrics()
        assert "documents_processed" in collected_metrics
        assert "total_processing_time" in collected_metrics
        assert "phi_entities_detected" in collected_metrics
        assert collected_metrics["documents_processed"] == len(sample_documents)


class TestSecurityIntegration:
    """Test security features in integrated scenarios."""

    def test_encryption_integration(self, temp_config_file, sample_documents, temp_dir):
        """Test end-to-end encryption of processed documents."""
        from hipaa_compliance_summarizer.security import EncryptionManager

        encryption_manager = EncryptionManager(key="test-key-32-characters-long-123")
        processor = HIPAAProcessor(
            config_path=temp_config_file, encryption_manager=encryption_manager
        )

        # Process and encrypt output
        doc_path = str(sample_documents["clinical_note"])
        result = processor.process_document(doc_path, encrypt_output=True)

        assert "encrypted_data" in result
        assert result["encrypted_data"] is not None

        # Verify we can decrypt the data
        decrypted = encryption_manager.decrypt(result["encrypted_data"])
        assert "summary" in json.loads(decrypted)

    def test_audit_trail_integration(self, temp_config_file, sample_documents):
        """Test comprehensive audit trail generation."""
        processor = HIPAAProcessor(
            config_path=temp_config_file, audit_logging=True
        )

        # Process documents with audit trail
        for doc_path in sample_documents.values():
            result = processor.process_document(str(doc_path))
            assert "audit_trail" in result
            assert len(result["audit_trail"]) > 0

            # Verify audit trail contains required fields
            for entry in result["audit_trail"]:
                assert "timestamp" in entry
                assert "action" in entry
                assert "user" in entry or "system" in entry

    def test_compliance_validation_integration(
        self, temp_config_file, sample_documents
    ):
        """Test real-time compliance validation."""
        processor = HIPAAProcessor(
            config_path=temp_config_file, real_time_compliance=True
        )

        # Process document with real-time compliance checking
        doc_path = str(sample_documents["clinical_note"])
        result = processor.process_document(doc_path)

        assert "compliance_validation" in result
        validation = result["compliance_validation"]

        assert "hipaa_compliant" in validation
        assert "risk_assessment" in validation
        assert "violations" in validation
        assert isinstance(validation["hipaa_compliant"], bool)
        assert validation["risk_assessment"] in ["LOW", "MEDIUM", "HIGH"]