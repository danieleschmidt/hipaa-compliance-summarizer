# Quick Start Guide

Get up and running with HIPAA Compliance Summarizer in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Healthcare documents for testing (optional)

## Installation

### Option 1: Standard Installation
```bash
# Clone the repository
git clone <repository-url>
cd hipaa-compliance-summarizer

# Install dependencies
pip install -r requirements.txt

# Install the package locally
pip install -e .
```

### Option 2: Docker Installation
```bash
# Pull and run the Docker container
docker pull hipaa-summarizer:latest
docker run -it --rm hipaa-summarizer:latest
```

## First Steps

### 1. Verify Installation
```bash
# Check if CLI tools are available
hipaa-summarize --help
hipaa-batch-process --help
hipaa-compliance-report --help
```

### 2. Configure Environment
```bash
# Copy example configuration
cp config/hipaa_config.yml.example config/hipaa_config.yml

# Edit configuration as needed
vim config/hipaa_config.yml
```

### 3. Process Your First Document
```bash
# Create a test document
echo "Patient John Doe, DOB: 01/15/1985, MRN: 123456789" > test_document.txt

# Process the document
hipaa-summarize --file test_document.txt --compliance-level standard
```

Expected output:
```
Processing document: test_document.txt
PHI entities detected: 3
Compliance score: 0.95
Redacted content saved to: test_document_redacted.txt
```

### 4. View Results
```bash
# Check the redacted document
cat test_document_redacted.txt
# Output: "Patient [NAME_REDACTED], DOB: [DATE_REDACTED], MRN: [MRN_REDACTED]"
```

## Next Steps

1. **Explore Batch Processing**: Process multiple documents at once
   ```bash
   hipaa-batch-process --input-dir ./documents --output-dir ./processed
   ```

2. **Generate Compliance Reports**: Create audit-ready reports
   ```bash
   hipaa-compliance-report --audit-period "2024-Q1"
   ```

3. **Customize Configuration**: Modify `config/hipaa_config.yml` for your needs

4. **Read Full Documentation**: Explore detailed guides in `docs/guides/`

## Common Issues

### "Command not found" Error
If CLI commands aren't recognized:
```bash
pip install -e .  # Reinstall the package
source ~/.bashrc  # Reload your shell
```

### Configuration Errors
If you see configuration warnings:
```bash
export HIPAA_CONFIG_PATH=/path/to/your/config.yml
```

### Permission Issues
If you encounter file permission errors:
```bash
chmod +x scripts/*.py
sudo chown -R $USER:$USER ./output_directory
```

## Getting Help

- **Documentation**: Check `docs/guides/` for detailed information
- **CLI Help**: Use `--help` flag with any command
- **Issues**: Report bugs at [GitHub Issues](../../CONTRIBUTING.md#reporting-issues)
- **Support**: Contact support@hipaa-summarizer.com

## Sample Workflow

Here's a complete workflow example:

```bash
# 1. Setup
mkdir my_medical_docs
echo "Patient Jane Smith, SSN: 123-45-6789, Phone: (555) 123-4567" > my_medical_docs/patient1.txt
echo "Patient Bob Johnson, DOB: 03/22/1978, Address: 123 Main St" > my_medical_docs/patient2.txt

# 2. Batch process
hipaa-batch-process \
  --input-dir my_medical_docs \
  --output-dir processed_docs \
  --compliance-level strict \
  --generate-summaries \
  --show-dashboard

# 3. Generate report
hipaa-compliance-report \
  --audit-period "$(date +%Y-%m)" \
  --documents-processed 2 \
  --include-recommendations

# 4. Review results
ls processed_docs/
cat processed_docs/dashboard.json
```

This completes your quick start! For more advanced usage, see the [User Guides](../README.md).