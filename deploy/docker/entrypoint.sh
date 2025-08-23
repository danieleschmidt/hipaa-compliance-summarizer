#!/bin/bash
set -euo pipefail

# HIPAA Compliance Summarizer Generation 4 - Production Entrypoint
# Terragon Labs - Secure Healthcare AI System

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    case $level in
        "ERROR")
            echo -e "${timestamp} ${RED}[ERROR]${NC} $message" >&2
            ;;
        "WARN")
            echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message"
            ;;
        "INFO")
            echo -e "${timestamp} ${GREEN}[INFO]${NC} $message"
            ;;
        "DEBUG")
            echo -e "${timestamp} ${BLUE}[DEBUG]${NC} $message"
            ;;
        *)
            echo -e "${timestamp} [LOG] $message"
            ;;
    esac
}

# Environment validation
validate_environment() {
    log "INFO" "Validating production environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "ENVIRONMENT"
        "COMPLIANCE_LEVEL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "ERROR" "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate compliance level
    if [[ "$COMPLIANCE_LEVEL" != "strict" ]] && [[ "$COMPLIANCE_LEVEL" != "standard" ]] && [[ "$COMPLIANCE_LEVEL" != "minimal" ]]; then
        log "ERROR" "Invalid COMPLIANCE_LEVEL: $COMPLIANCE_LEVEL. Must be 'strict', 'standard', or 'minimal'"
        exit 1
    fi
    
    log "INFO" "Environment validation passed"
    log "INFO" "Running in $ENVIRONMENT mode with $COMPLIANCE_LEVEL compliance"
}

# Security checks
perform_security_checks() {
    log "INFO" "Performing security validation checks..."
    
    # Check if running as non-root
    if [[ $(id -u) -eq 0 ]]; then
        log "ERROR" "Container is running as root user - security violation"
        exit 1
    fi
    
    # Verify file permissions
    local secure_dirs=("/app/logs" "/app/cache" "/app/tmp")
    for dir in "${secure_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local perms=$(stat -c "%a" "$dir")
            if [[ "$perms" != "755" ]] && [[ "$perms" != "700" ]]; then
                log "WARN" "Directory $dir has permissions $perms, expected 755 or 700"
            fi
        fi
    done
    
    # Check for sensitive files in PATH
    if [[ -f "/app/.env" ]] || [[ -f "/app/secrets.txt" ]]; then
        log "ERROR" "Sensitive files detected in application directory"
        exit 1
    fi
    
    log "INFO" "Security checks completed"
}

# Initialize application directories
initialize_directories() {
    log "INFO" "Initializing application directories..."
    
    # Create directories if they don't exist
    local dirs=("/app/logs" "/app/cache" "/app/tmp" "/app/data")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "DEBUG" "Created directory: $dir"
        fi
    done
    
    # Initialize cache structure for ML optimization
    if [[ "$ENABLE_ML_OPTIMIZATION" == "true" ]]; then
        mkdir -p /app/cache/models /app/cache/patterns /app/cache/performance
        log "INFO" "ML optimization cache directories initialized"
    fi
    
    # Initialize monitoring directories
    mkdir -p /app/logs/access /app/logs/error /app/logs/security /app/logs/performance
    
    log "INFO" "Directory initialization completed"
}

# Configuration validation
validate_configuration() {
    log "INFO" "Validating HIPAA configuration..."
    
    # Check if configuration file exists
    if [[ -f "/app/config/hipaa_config.yml" ]]; then
        log "INFO" "HIPAA configuration file found"
        
        # Basic YAML syntax check
        if ! python3 -c "import yaml; yaml.safe_load(open('/app/config/hipaa_config.yml'))" 2>/dev/null; then
            log "ERROR" "Invalid YAML syntax in HIPAA configuration file"
            exit 1
        fi
    else
        log "WARN" "HIPAA configuration file not found, using defaults"
    fi
    
    # Validate ML models availability if enabled
    if [[ "$ENABLE_ML_OPTIMIZATION" == "true" ]]; then
        log "INFO" "Validating ML optimization components..."
        
        # Check if required Python packages are available
        if ! python3 -c "import sklearn, numpy" 2>/dev/null; then
            log "ERROR" "Required ML packages not available"
            exit 1
        fi
        
        log "INFO" "ML optimization components validated"
    fi
    
    log "INFO" "Configuration validation completed"
}

# Performance optimization setup
setup_performance_optimization() {
    log "INFO" "Setting up performance optimization..."
    
    # Set optimal Python settings for production
    export PYTHONOPTIMIZE=1
    export PYTHONUTF8=1
    
    # Configure memory settings based on container limits
    if [[ -n "${MEMORY_LIMIT:-}" ]]; then
        # Calculate optimal worker processes based on memory
        local memory_gb=$(echo "$MEMORY_LIMIT" | sed 's/[^0-9]*//g')
        if [[ -n "$memory_gb" ]] && [[ "$memory_gb" -gt 0 ]]; then
            export WORKERS=$((memory_gb / 1 + 1))  # 1 worker per GB + 1
            log "INFO" "Set worker processes to $WORKERS based on memory limit ${MEMORY_LIMIT}"
        fi
    fi
    
    # Enable ML optimization if configured
    if [[ "$ENABLE_ML_OPTIMIZATION" == "true" ]]; then
        log "INFO" "ML-driven performance optimization enabled"
        
        # Pre-warm ML models and caches
        log "INFO" "Pre-warming ML optimization components..."
        timeout 30s python3 -c "
from src.hipaa_compliance_summarizer.performance_gen4 import ml_optimizer, schedule_ml_training
from src.hipaa_compliance_summarizer.intelligent_scaling import setup_hipaa_scaling_policies
log_msg = lambda msg: print(f'[PREWARM] {msg}')
log_msg('ML optimizer initialized')
setup_hipaa_scaling_policies()
log_msg('HIPAA scaling policies configured')
" || log "WARN" "ML pre-warming completed with warnings"
    fi
    
    log "INFO" "Performance optimization setup completed"
}

# Health check initialization
initialize_health_checks() {
    log "INFO" "Initializing health check endpoints..."
    
    # Create health check status files
    touch /app/tmp/startup_complete
    touch /app/tmp/ready
    touch /app/tmp/healthy
    
    # Set up performance monitoring if enabled
    if [[ "${PROMETHEUS_METRICS_ENABLED:-false}" == "true" ]]; then
        log "INFO" "Prometheus metrics enabled on port 8080"
    fi
    
    log "INFO" "Health check initialization completed"
}

# Signal handling for graceful shutdown
setup_signal_handlers() {
    log "INFO" "Setting up signal handlers for graceful shutdown..."
    
    # Function for graceful shutdown
    graceful_shutdown() {
        log "INFO" "Received shutdown signal, initiating graceful shutdown..."
        
        # Mark as not ready
        rm -f /app/tmp/ready 2>/dev/null || true
        
        # Stop accepting new connections
        if [[ -n "${MAIN_PID:-}" ]]; then
            log "INFO" "Sending TERM signal to main process..."
            kill -TERM "$MAIN_PID" 2>/dev/null || true
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$MAIN_PID" 2>/dev/null && [[ $count -lt 30 ]]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if kill -0 "$MAIN_PID" 2>/dev/null; then
                log "WARN" "Forcing shutdown of main process..."
                kill -KILL "$MAIN_PID" 2>/dev/null || true
            fi
        fi
        
        # Cleanup
        rm -f /app/tmp/healthy /app/tmp/startup_complete 2>/dev/null || true
        log "INFO" "Graceful shutdown completed"
        exit 0
    }
    
    # Set up signal traps
    trap graceful_shutdown SIGTERM SIGINT SIGQUIT
}

# Main execution
main() {
    log "INFO" "Starting HIPAA Compliance Summarizer Generation 4..."
    log "INFO" "Terragon Labs - Secure Healthcare AI System"
    
    # Perform all initialization steps
    validate_environment
    perform_security_checks
    initialize_directories
    validate_configuration
    setup_performance_optimization
    initialize_health_checks
    setup_signal_handlers
    
    # Mark startup as complete
    echo "$(date -u +%s)" > /app/tmp/startup_complete
    echo "ready" > /app/tmp/ready
    echo "healthy" > /app/tmp/healthy
    
    log "INFO" "Initialization completed successfully"
    log "INFO" "Starting application with command: $*"
    
    # Execute the main command
    exec "$@" &
    MAIN_PID=$!
    
    # Wait for the main process
    wait $MAIN_PID
    local exit_code=$?
    
    log "INFO" "Application exited with code: $exit_code"
    exit $exit_code
}

# Run main function with all arguments
main "$@"