import tempfile
import os
import sys
from unittest.mock import patch
from hipaa_compliance_summarizer.cli.batch_process import main


def test_cli_cache_performance_flag(tmp_path, capsys, monkeypatch):
    """Test that CLI can display cache performance metrics."""
    
    # Create test input directory
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create test files
    test_content = "Patient John Doe, SSN: 123-45-6789, Phone: 555-123-4567"
    for i in range(3):
        test_file = input_dir / f"patient_{i}.txt"
        test_file.write_text(test_content)
    
    # Mock command line arguments
    test_args = [
        "hipaa-batch-process",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--show-cache-performance"
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the CLI command
    main()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should display cache performance metrics
    assert "Cache Performance:" in captured.out
    assert "Pattern Compilation - Hits:" in captured.out
    assert "PHI Detection - Hits:" in captured.out
    assert "Cache Memory Usage" in captured.out
    assert "Hit Ratio:" in captured.out


def test_cli_combined_dashboard_and_cache(tmp_path, capsys, monkeypatch):
    """Test CLI with both dashboard and cache performance flags."""
    
    # Create test input directory
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create test files with duplicate content for cache hits
    test_content = "Patient SSN: 123-45-6789"
    for i in range(5):
        test_file = input_dir / f"duplicate_{i}.txt"
        test_file.write_text(test_content)
    
    # Mock command line arguments
    test_args = [
        "hipaa-batch-process",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--show-dashboard",
        "--show-cache-performance"
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the CLI command
    main()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should display both dashboard and cache performance
    assert "Documents processed: 5" in captured.out
    assert "Cache Performance:" in captured.out
    assert "PHI Detection - Hits:" in captured.out
    
    # With duplicate content, should have some cache hits for PHI detection
    # (Pattern hits are harder to predict due to initialization)


def test_cli_cache_performance_with_unique_content(tmp_path, capsys, monkeypatch):
    """Test cache performance with unique content (should show mostly misses)."""
    
    # Create test input directory
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create test files with unique SSNs
    for i in range(5):
        ssn = f"{123+i:03d}-{45+i:02d}-{6789+i:04d}"
        test_content = f"Patient SSN: {ssn}"
        test_file = input_dir / f"patient_{i}.txt"
        test_file.write_text(test_content)
    
    # Mock command line arguments
    test_args = [
        "hipaa-batch-process",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--show-cache-performance"
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the CLI command
    main()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should display cache performance metrics
    assert "Cache Performance:" in captured.out
    assert "Pattern Compilation" in captured.out
    assert "PHI Detection" in captured.out
    
    # With unique content, PHI detection hit ratio should be low
    # Pattern compilation should have some hits due to reuse


def test_cli_without_cache_performance_flag(tmp_path, capsys, monkeypatch):
    """Test that cache performance is not shown without the flag."""
    
    # Create test input directory
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create test file
    test_file = input_dir / "patient.txt"
    test_file.write_text("Patient SSN: 123-45-6789")
    
    # Mock command line arguments (without cache performance flag)
    test_args = [
        "hipaa-batch-process",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir)
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the CLI command
    main()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Should NOT display cache performance metrics
    assert "Cache Performance:" not in captured.out
    assert "Pattern Compilation - Hits:" not in captured.out
    assert "PHI Detection - Hits:" not in captured.out