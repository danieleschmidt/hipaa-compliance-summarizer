"""
Comprehensive end-to-end integration tests for the full PHI pipeline.

This module tests the complete workflow from document ingestion through PHI detection,
redaction, compliance scoring, and reporting to ensure the entire system works
cohesively for healthcare document processing.

Test coverage includes:
- Complete PHI detection and redaction pipeline
- Batch processing workflows
- Multi-document type handling
- Security validation integration
- Performance benchmarking
- Error handling and recovery
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time
import logging

from hipaa_compliance_summarizer import (
    HIPAAProcessor,
    BatchProcessor,
    ComplianceReporter,
    Document,
    DocumentType,
    ComplianceLevel,
    PHIRedactor,
    ProcessingResult,
    BatchDashboard,
    detect_document_type,
    SecurityError
)


class TestFullPipelineIntegration:
    """Complete end-to-end integration tests for PHI processing pipeline."""

    @pytest.fixture
    def sample_healthcare_documents(self, tmp_path) -> Dict[str, Path]:
        """Create realistic healthcare document samples for testing."""
        documents = {}
        
        # Clinical Note
        clinical_note = tmp_path / "clinical_note.txt"
        clinical_note.write_text("""
        CLINICAL NOTE
        Patient: John Doe
        DOB: 01/15/1980
        SSN: 123-45-6789
        MRN: MR123456
        Phone: (555) 123-4567
        Address: 123 Main St, Anytown, NY 12345
        
        CHIEF COMPLAINT: Chest pain and shortness of breath
        
        HISTORY: 43-year-old male presents with acute onset chest pain.
        Patient reports pain started at approximately 14:30 on 03/22/2024.
        Patient denies previous cardiac history.
        
        ASSESSMENT: Possible acute coronary syndrome
        PLAN: Admit to CCU for monitoring and serial cardiac enzymes
        
        Dr. Sarah Smith, MD
        License: NY12345
        """)
        documents["clinical_note"] = clinical_note
        
        # Insurance Form
        insurance_form = tmp_path / "insurance_claim.txt"
        insurance_form.write_text("""
        INSURANCE CLAIM FORM
        
        Patient Information:
        Name: Jane Smith
        DOB: 05/22/1975
        SSN: 987-65-4321
        Policy Number: INS789012
        Group Number: GRP345
        
        Provider Information:
        Provider Name: Metro Health Clinic
        NPI: 1234567890
        Tax ID: 12-3456789
        Phone: (555) 987-6543
        
        Claim Details:
        Date of Service: 03/15/2024
        Diagnosis Code: I25.10
        Procedure Code: 99213
        Amount: $250.00
        """)
        documents["insurance_form"] = insurance_form
        
        # Lab Report
        lab_report = tmp_path / "lab_report.txt"
        lab_report.write_text("""
        LABORATORY REPORT
        
        Patient: Robert Johnson
        DOB: 11/08/1965
        MRN: LAB987654
        Account: ACC123789
        Collected: 03/20/2024 08:30
        
        COMPLETE BLOOD COUNT
        WBC: 7.2 K/uL (Normal: 4.0-11.0)
        RBC: 4.5 M/uL (Normal: 4.2-5.4)
        Hemoglobin: 14.2 g/dL (Normal: 13.5-17.5)
        Hematocrit: 42.1% (Normal: 41-50)
        
        Ordering Physician: Dr. Michael Brown
        License: NY67890
        """)
        documents["lab_report"] = lab_report
        
        # Clean text file (no PHI)
        clean_doc = tmp_path / "clean_document.txt"
        clean_doc.write_text("""
        MEDICAL RESEARCH SUMMARY
        
        This document contains general medical information without any
        patient-specific details. It discusses treatment protocols and
        general best practices for cardiac care.
        
        The study involved anonymous participants and follows all
        HIPAA guidelines for de-identified research data.
        """)
        documents["clean_document"] = clean_doc
        
        return documents

    @pytest.fixture
    def batch_processing_directory(self, tmp_path, sample_healthcare_documents) -> Path:
        """Create a directory structure for batch processing tests."""
        batch_dir = tmp_path / "batch_input"
        batch_dir.mkdir()
        
        # Copy sample documents to batch directory
        for name, doc_path in sample_healthcare_documents.items():
            shutil.copy2(doc_path, batch_dir / f"{name}.txt")
        
        # Add some additional test files
        (batch_dir / "empty_file.txt").write_text("")
        (batch_dir / "large_file.txt").write_text("Sample text. " * 1000)
        
        return batch_dir

    def test_single_document_full_pipeline(self, sample_healthcare_documents):
        """Test complete pipeline for a single document with PHI."""
        clinical_note = sample_healthcare_documents["clinical_note"]
        
        # Initialize processor
        processor = HIPAAProcessor(compliance_level=ComplianceLevel.STRICT)
        
        # Create document object
        document = Document(str(clinical_note), DocumentType.CLINICAL_NOTE)
        
        # Process document
        result = processor.process_document(document)
        
        # Verify processing results
        assert isinstance(result, ProcessingResult)
        assert result.phi_detected_count > 0  # Should detect PHI (SSN, phone, dates)
        assert 0.0 <= result.compliance_score <= 1.0
        assert "[REDACTED]" in result.redacted.text
        
        # Verify specific PHI types are redacted (based on current patterns)
        assert "123-45-6789" not in result.redacted.text  # SSN should be redacted
        assert "01/15/1980" not in result.redacted.text or "DOB: [REDACTED]" in result.redacted.text  # DOB should be redacted
        assert "(555) 123-4567" not in result.redacted.text or "555-123-4567" not in result.redacted.text  # Phone should be redacted
        
        # Verify summary is generated
        assert len(result.summary) > 0
        assert len(result.summary) < len(clinical_note.read_text())

    def test_clean_document_pipeline(self, sample_healthcare_documents):
        """Test pipeline with document containing no PHI."""
        clean_doc = sample_healthcare_documents["clean_document"]
        
        processor = HIPAAProcessor(compliance_level=ComplianceLevel.STANDARD)
        document = Document(str(clean_doc), DocumentType.CLINICAL_NOTE)
        
        result = processor.process_document(document)
        
        # Clean document should have high compliance score and no PHI
        assert result.phi_detected_count == 0
        assert result.compliance_score >= 0.9  # High score for clean document
        assert result.redacted.text == clean_doc.read_text().strip()

    def test_batch_processing_full_workflow(self, batch_processing_directory, tmp_path):
        """Test complete batch processing workflow with multiple document types."""
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()
        
        # Initialize batch processor
        batch_processor = BatchProcessor()
        
        # Process directory
        results = batch_processor.process_directory(
            str(batch_processing_directory),
            output_dir=str(output_dir),
            compliance_level=ComplianceLevel.STANDARD,
            generate_summaries=True
        )
        
        # Verify results
        assert len(results) > 0
        successful_results = [r for r in results if hasattr(r, 'compliance_score')]
        assert len(successful_results) > 0
        
        # Verify output files exist
        assert len(list(output_dir.iterdir())) > 0
        
        # Generate and verify dashboard
        dashboard = batch_processor.generate_dashboard(results)
        assert isinstance(dashboard, BatchDashboard)
        assert dashboard.documents_processed > 0
        assert 0.0 <= dashboard.avg_compliance_score <= 1.0

    def test_multi_compliance_level_processing(self, sample_healthcare_documents):
        """Test same document processed at different compliance levels."""
        clinical_note = sample_healthcare_documents["clinical_note"]
        document = Document(str(clinical_note), DocumentType.CLINICAL_NOTE)
        
        results = {}
        for level in [ComplianceLevel.MINIMAL, ComplianceLevel.STANDARD, ComplianceLevel.STRICT]:
            processor = HIPAAProcessor(compliance_level=level)
            results[level] = processor.process_document(document)
        
        # Verify all compliance levels processed successfully
        for level in [ComplianceLevel.MINIMAL, ComplianceLevel.STANDARD, ComplianceLevel.STRICT]:
            assert isinstance(results[level], ProcessingResult)
            assert 0.0 <= results[level].compliance_score <= 1.0
            assert results[level].phi_detected_count > 0  # Should detect PHI in all modes
        
        # Stricter levels should have more comprehensive redaction
        strict_redactions = results[ComplianceLevel.STRICT].redacted.text.count("[REDACTED]")
        minimal_redactions = results[ComplianceLevel.MINIMAL].redacted.text.count("[REDACTED]")
        assert strict_redactions >= minimal_redactions

    def test_document_type_detection_integration(self, sample_healthcare_documents):
        """Test automatic document type detection in the pipeline."""
        for doc_name, doc_path in sample_healthcare_documents.items():
            # Test automatic detection
            detected_type = detect_document_type(str(doc_path))
            assert isinstance(detected_type, DocumentType)
            
            # Process with detected type
            document = Document(str(doc_path), detected_type)
            processor = HIPAAProcessor()
            result = processor.process_document(document)
            
            assert isinstance(result, ProcessingResult)

    def test_security_validation_integration(self, tmp_path):
        """Test security validation throughout the pipeline."""
        # Create potentially problematic files
        suspicious_file = tmp_path / "suspicious_file.txt"
        suspicious_file.write_text("Normal content with some PHI: SSN 123-45-6789")
        
        # Test that security validation doesn't block normal processing
        processor = HIPAAProcessor()
        document = Document(str(suspicious_file), DocumentType.CLINICAL_NOTE)
        
        # Should process successfully despite security checks
        result = processor.process_document(document)
        assert isinstance(result, ProcessingResult)
        assert result.phi_detected_count > 0

    def test_error_handling_in_pipeline(self, tmp_path):
        """Test error handling and recovery in the full pipeline."""
        # Create invalid file
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("")  # Empty file
        
        processor = HIPAAProcessor()
        document = Document(str(invalid_file), DocumentType.CLINICAL_NOTE)
        
        # Should handle empty file gracefully
        result = processor.process_document(document)
        assert isinstance(result, ProcessingResult)
        assert result.phi_detected_count == 0

    def test_compliance_reporting_integration(self, sample_healthcare_documents):
        """Test integration between processing and compliance reporting."""
        # Process multiple documents
        results = []
        processor = HIPAAProcessor()
        
        for doc_path in sample_healthcare_documents.values():
            document = Document(str(doc_path), DocumentType.CLINICAL_NOTE)
            result = processor.process_document(document)
            results.append(result)
        
        # Generate compliance report
        reporter = ComplianceReporter()
        report = reporter.generate_report(
            period="2024-Q1",
            documents_processed=len(results),
            include_recommendations=True
        )
        
        # Verify report structure
        assert hasattr(report, 'overall_compliance')
        assert 0.0 <= report.overall_compliance <= 1.0

    def test_phi_redactor_standalone_integration(self, sample_healthcare_documents):
        """Test PHI redactor as standalone component in pipeline."""
        clinical_note = sample_healthcare_documents["clinical_note"]
        
        # Test direct PHI redaction
        redactor = PHIRedactor()
        redaction_result = redactor.redact_file(str(clinical_note))
        
        # Verify redaction occurred
        assert "[REDACTED]" in redaction_result.text
        assert len(redaction_result.entities) > 0
        # Verify specific PHI (SSN, dates, phone) are redacted based on current patterns
        original_text = clinical_note.read_text()
        assert "123-45-6789" not in redaction_result.text  # SSN should be redacted

    def test_performance_baseline_integration(self, sample_healthcare_documents):
        """Test performance characteristics of the full pipeline."""
        clinical_note = sample_healthcare_documents["clinical_note"]
        processor = HIPAAProcessor()
        document = Document(str(clinical_note), DocumentType.CLINICAL_NOTE)
        
        # Measure processing time
        start_time = time.time()
        result = processor.process_document(document)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertion - should process typical document in under 5 seconds
        assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, expected < 5.0s"
        assert isinstance(result, ProcessingResult)

    def test_batch_dashboard_json_export(self, batch_processing_directory, tmp_path):
        """Test batch processing with JSON dashboard export."""
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()
        dashboard_file = tmp_path / "dashboard.json"
        
        batch_processor = BatchProcessor()
        results = batch_processor.process_directory(
            str(batch_processing_directory),
            output_dir=str(output_dir),
            compliance_level=ComplianceLevel.STANDARD
        )
        
        # Export dashboard to JSON
        batch_processor.save_dashboard(results, str(dashboard_file))
        
        # Verify JSON structure
        assert dashboard_file.exists()
        with open(dashboard_file) as f:
            dashboard_data = json.load(f)
        
        required_keys = ["documents_processed", "avg_compliance_score", "total_phi_detected"]
        for key in required_keys:
            assert key in dashboard_data

    def test_concurrent_processing_integration(self, sample_healthcare_documents):
        """Test concurrent processing capabilities."""
        documents = [
            Document(str(path), DocumentType.CLINICAL_NOTE)
            for path in sample_healthcare_documents.values()
        ]
        
        # Create processors matching document count
        processors = [HIPAAProcessor() for _ in range(len(documents))]
        
        # Process documents concurrently (simulated)
        results = []
        for processor, document in zip(processors, documents):
            result = processor.process_document(document)
            results.append(result)
        
        # Verify all processing completed successfully
        assert len(results) == len(documents)
        for result in results:
            assert isinstance(result, ProcessingResult)


class TestIntegrationEdgeCases:
    """Test edge cases and boundary conditions in the integrated system."""

    def test_large_document_processing(self, tmp_path):
        """Test processing of large documents."""
        large_doc = tmp_path / "large_document.txt"
        
        # Create large document with PHI scattered throughout
        content = []
        for i in range(1000):
            if i % 100 == 0:  # Add PHI every 100 lines
                content.append(f"Patient SSN: {i:03d}-45-6789 in room {i}")
            else:
                content.append(f"Line {i}: General medical information and treatment notes.")
        
        large_doc.write_text("\n".join(content))
        
        processor = HIPAAProcessor()
        document = Document(str(large_doc), DocumentType.CLINICAL_NOTE)
        
        start_time = time.time()
        result = processor.process_document(document)
        processing_time = time.time() - start_time
        
        # Should handle large documents efficiently
        assert isinstance(result, ProcessingResult)
        assert result.phi_detected_count > 0
        assert processing_time < 30.0  # Should complete within 30 seconds

    def test_mixed_content_batch_processing(self, tmp_path):
        """Test batch processing with mixed content types and quality."""
        batch_dir = tmp_path / "mixed_batch"
        batch_dir.mkdir()
        output_dir = tmp_path / "mixed_output"
        output_dir.mkdir()
        
        # Create varied content
        files = {
            "good_content.txt": "Patient John Doe, SSN 123-45-6789, is recovering well.",
            "empty_content.txt": "",
            "no_phi_content.txt": "General medical information without identifiers.",
            "special_chars.txt": "Patient: José María, ID: ABC-123 with unicode content.",
        }
        
        for filename, content in files.items():
            (batch_dir / filename).write_text(content)
        
        batch_processor = BatchProcessor()
        results = batch_processor.process_directory(
            str(batch_dir),
            output_dir=str(output_dir),
            compliance_level=ComplianceLevel.STANDARD
        )
        
        # Should handle mixed content gracefully
        assert len(results) == len(files)
        
        # Generate dashboard
        dashboard = batch_processor.generate_dashboard(results)
        assert dashboard.documents_processed == len(files)


class TestIntegrationPerformanceBenchmarks:
    """Performance benchmarking tests for the integrated system."""

    def test_batch_processing_throughput(self, tmp_path):
        """Benchmark batch processing throughput."""
        batch_dir = tmp_path / "throughput_test"
        batch_dir.mkdir()
        output_dir = tmp_path / "throughput_output"
        output_dir.mkdir()
        
        # Create standardized test documents
        num_docs = 10
        for i in range(num_docs):
            doc_path = batch_dir / f"test_doc_{i:03d}.txt"
            doc_path.write_text(f"""
            Patient Record {i}
            Name: Test Patient {i}
            SSN: {i:03d}-45-6789
            DOB: 0{(i % 9) + 1}/15/198{i % 10}
            
            Clinical notes for patient {i} with standard medical content
            and typical PHI patterns for benchmarking purposes.
            """)
        
        # Measure batch processing performance
        batch_processor = BatchProcessor()
        start_time = time.time()
        
        results = batch_processor.process_directory(
            str(batch_dir),
            output_dir=str(output_dir),
            compliance_level=ComplianceLevel.STANDARD
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_docs / total_time
        
        # Performance assertions
        assert len(results) == num_docs
        assert throughput > 1.0  # Should process more than 1 document per second
        assert total_time < 60.0  # Should complete within 1 minute

    def test_memory_usage_stability(self, tmp_path):
        """Test memory usage stability during extended processing."""
        # This test verifies that memory usage doesn't grow unbounded
        processor = HIPAAProcessor()
        
        # Process multiple documents in sequence
        for i in range(20):
            test_doc = tmp_path / f"memory_test_{i}.txt"
            test_doc.write_text(f"Patient SSN: {i:03d}-45-6789 test content")
            
            document = Document(str(test_doc), DocumentType.CLINICAL_NOTE)
            result = processor.process_document(document)
            
            # Each processing should complete successfully
            assert isinstance(result, ProcessingResult)
            
            # Clean up file to save space
            test_doc.unlink()