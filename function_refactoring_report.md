# Python Function Refactoring Report
## Functions Exceeding 50 Lines

Found 10 functions that exceed 50 lines and may need refactoring:

### 1. **main** (98 lines)
- **File**: `src/hipaa_compliance_summarizer/cli/batch_process.py`
- **Lines**: 13-124
- **Type**: Function
- **Priority**: ⚡ Medium

### 2. **main** (88 lines)
- **File**: `src/cli_autonomous_backlog.py`
- **Lines**: 225-334
- **Type**: Function
- **Priority**: ⚡ Medium

### 3. **get_cache_performance** (71 lines)
- **File**: `src/hipaa_compliance_summarizer/batch.py`
- **Lines**: 581-661
- **Type**: Method in class `BatchProcessor`
- **Priority**: 📈 Low

### 4. **_execute_file_processing** (60 lines)
- **File**: `src/hipaa_compliance_summarizer/batch.py`
- **Lines**: 286-362
- **Type**: Method in class `BatchProcessor`
- **Priority**: 📈 Low

### 5. **load_default_patterns** (60 lines)
- **File**: `src/hipaa_compliance_summarizer/phi_patterns.py`
- **Lines**: 93-162
- **Type**: Method in class `PHIPatternManager`
- **Priority**: 📈 Low

### 6. **_load_text** (57 lines)
- **File**: `src/hipaa_compliance_summarizer/parsers.py`
- **Lines**: 83-163
- **Type**: Function
- **Priority**: 📈 Low

### 7. **validate_environment** (57 lines)
- **File**: `src/hipaa_compliance_summarizer/startup.py`
- **Lines**: 13-86
- **Type**: Function
- **Priority**: 📈 Low

### 8. **_process_file_with_monitoring** (55 lines)
- **File**: `src/hipaa_compliance_summarizer/batch.py`
- **Lines**: 364-434
- **Type**: Method in class `BatchProcessor`
- **Priority**: 📈 Low

### 9. **process_document** (53 lines)
- **File**: `src/hipaa_compliance_summarizer/processor.py`
- **Lines**: 202-265
- **Type**: Method in class `HIPAAProcessor`
- **Priority**: 📈 Low

### 10. **_parse_comprehensive_backlog** (52 lines)
- **File**: `src/autonomous_backlog_assistant.py`
- **Lines**: 122-182
- **Type**: Method in class `AutonomousBacklogAssistant`
- **Priority**: 📈 Low

## Summary Statistics

- **Total functions over 50 lines**: 10
- **Total lines in large functions**: 651
- **Average lines per large function**: 65.1
- **Largest function**: 98 lines

## Priority Breakdown

- **🔥 High Priority (>100 lines)**: 0 functions
- **⚡ Medium Priority (75-100 lines)**: 2 functions
- **📈 Low Priority (50-75 lines)**: 8 functions

## Files with Large Functions

- `src/hipaa_compliance_summarizer/batch.py`: 3 functions
- `src/hipaa_compliance_summarizer/cli/batch_process.py`: 1 functions
- `src/cli_autonomous_backlog.py`: 1 functions
- `src/hipaa_compliance_summarizer/phi_patterns.py`: 1 functions
- `src/hipaa_compliance_summarizer/parsers.py`: 1 functions
- `src/hipaa_compliance_summarizer/startup.py`: 1 functions
- `src/hipaa_compliance_summarizer/processor.py`: 1 functions
- `src/autonomous_backlog_assistant.py`: 1 functions

## Refactoring Recommendations

### High Priority Functions (>100 lines)