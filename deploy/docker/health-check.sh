#!/bin/bash
set -euo pipefail

# HIPAA Compliance Summarizer - Health Check Script
# Comprehensive health validation for production deployment

# Configuration
HEALTH_CHECK_TIMEOUT=10
APPLICATION_PORT=${APPLICATION_PORT:-8000}
METRICS_PORT=${METRICS_PORT:-8080}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Health check endpoints
HEALTH_ENDPOINT="http://localhost:${APPLICATION_PORT}/health"
READY_ENDPOINT="http://localhost:${APPLICATION_PORT}/ready"
STARTUP_ENDPOINT="http://localhost:${APPLICATION_PORT}/startup"
METRICS_ENDPOINT="http://localhost:${METRICS_PORT}/metrics"

# Status files
STARTUP_FILE="/app/tmp/startup_complete"
READY_FILE="/app/tmp/ready"
HEALTHY_FILE="/app/tmp/healthy"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    if [[ "$LOG_LEVEL" == "DEBUG" ]] || [[ "$level" != "DEBUG" ]]; then
        echo "${timestamp} [${level}] HEALTHCHECK: $message" >&2
    fi
}

# Check if application is responding
check_application_response() {
    log "DEBUG" "Checking application response on port $APPLICATION_PORT"
    
    # Use curl with timeout and specific options
    if curl --fail \
           --silent \
           --max-time "$HEALTH_CHECK_TIMEOUT" \
           --connect-timeout 5 \
           --retry 0 \
           "$HEALTH_ENDPOINT" >/dev/null 2>&1; then
        log "DEBUG" "Application health endpoint responded successfully"
        return 0
    else
        log "ERROR" "Application health endpoint not responding"
        return 1
    fi
}

# Check application readiness
check_application_readiness() {
    log "DEBUG" "Checking application readiness"
    
    # Check readiness file
    if [[ ! -f "$READY_FILE" ]]; then
        log "ERROR" "Application readiness file not found"
        return 1
    fi
    
    # Check readiness endpoint if available
    if curl --fail \
           --silent \
           --max-time "$HEALTH_CHECK_TIMEOUT" \
           --connect-timeout 3 \
           --retry 0 \
           "$READY_ENDPOINT" >/dev/null 2>&1; then
        log "DEBUG" "Application readiness endpoint responded successfully"
        return 0
    else
        log "WARN" "Application readiness endpoint not responding (may not be implemented)"
        # Return success if file exists even if endpoint doesn't respond
        return 0
    fi
}

# Check startup completion
check_startup_complete() {
    log "DEBUG" "Checking startup completion"
    
    if [[ ! -f "$STARTUP_FILE" ]]; then
        log "ERROR" "Application startup not complete"
        return 1
    fi
    
    # Check if startup was recent (within last hour)
    local startup_time
    startup_time=$(cat "$STARTUP_FILE" 2>/dev/null || echo "0")
    local current_time
    current_time=$(date +%s)
    local time_diff=$((current_time - startup_time))
    
    if [[ $time_diff -gt 3600 ]]; then
        log "WARN" "Application startup was more than 1 hour ago ($time_diff seconds)"
    else
        log "DEBUG" "Application startup completed $time_diff seconds ago"
    fi
    
    return 0
}

# Check resource usage
check_resource_usage() {
    log "DEBUG" "Checking resource usage"
    
    # Check memory usage (if available)
    if command -v ps >/dev/null 2>&1; then
        local memory_usage
        memory_usage=$(ps -o pid,ppid,vsz,rss,comm -p $$ 2>/dev/null | tail -n +2 | head -1 | awk '{print $4}')
        
        if [[ -n "$memory_usage" ]] && [[ "$memory_usage" -gt 0 ]]; then
            local memory_mb=$((memory_usage / 1024))
            log "DEBUG" "Current memory usage: ${memory_mb}MB"
            
            # Alert if memory usage is very high (> 3GB)
            if [[ $memory_mb -gt 3072 ]]; then
                log "WARN" "High memory usage detected: ${memory_mb}MB"
            fi
        fi
    fi
    
    # Check disk space for logs and cache
    if command -v df >/dev/null 2>&1; then
        local disk_usage
        disk_usage=$(df /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
        
        if [[ -n "$disk_usage" ]] && [[ "$disk_usage" -gt 0 ]]; then
            log "DEBUG" "Disk usage: ${disk_usage}%"
            
            if [[ $disk_usage -gt 85 ]]; then
                log "WARN" "High disk usage: ${disk_usage}%"
            fi
        fi
    fi
    
    return 0
}

# Check metrics endpoint (if enabled)
check_metrics_endpoint() {
    if [[ "${PROMETHEUS_METRICS_ENABLED:-false}" == "true" ]]; then
        log "DEBUG" "Checking metrics endpoint on port $METRICS_PORT"
        
        if curl --fail \
               --silent \
               --max-time 5 \
               --connect-timeout 2 \
               --retry 0 \
               "$METRICS_ENDPOINT" >/dev/null 2>&1; then
            log "DEBUG" "Metrics endpoint responding successfully"
        else
            log "WARN" "Metrics endpoint not responding"
            # Don't fail health check for metrics endpoint
        fi
    fi
    
    return 0
}

# Check security indicators
check_security_status() {
    log "DEBUG" "Checking security status"
    
    # Verify running as non-root
    if [[ $(id -u) -eq 0 ]]; then
        log "ERROR" "Running as root user - security violation"
        return 1
    fi
    
    # Check for security status file if it exists
    local security_file="/app/tmp/security_status"
    if [[ -f "$security_file" ]]; then
        local security_status
        security_status=$(cat "$security_file" 2>/dev/null || echo "unknown")
        
        if [[ "$security_status" == "compromised" ]]; then
            log "ERROR" "Security compromise detected"
            return 1
        fi
        
        log "DEBUG" "Security status: $security_status"
    fi
    
    return 0
}

# Check HIPAA compliance indicators
check_compliance_status() {
    log "DEBUG" "Checking HIPAA compliance status"
    
    # Verify compliance level is set
    if [[ -z "${COMPLIANCE_LEVEL:-}" ]]; then
        log "ERROR" "COMPLIANCE_LEVEL not set"
        return 1
    fi
    
    # Check for compliance audit log
    local audit_log="/app/logs/security/audit.log"
    if [[ -f "$audit_log" ]] && [[ -w "$audit_log" ]]; then
        log "DEBUG" "HIPAA audit logging enabled"
    else
        log "WARN" "HIPAA audit log not accessible"
    fi
    
    return 0
}

# Performance health check
check_performance_health() {
    log "DEBUG" "Checking performance health"
    
    # Check if ML optimization is enabled and functioning
    if [[ "${ENABLE_ML_OPTIMIZATION:-false}" == "true" ]]; then
        # Look for ML performance indicators
        local perf_file="/app/cache/performance/status"
        if [[ -f "$perf_file" ]]; then
            log "DEBUG" "ML optimization status file found"
        else
            log "DEBUG" "ML optimization status file not found (may be initializing)"
        fi
    fi
    
    # Check response time (simple test)
    local start_time
    local end_time
    local response_time
    
    start_time=$(date +%s%N)
    if curl --fail --silent --max-time 5 "$HEALTH_ENDPOINT" >/dev/null 2>&1; then
        end_time=$(date +%s%N)
        response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        log "DEBUG" "Health endpoint response time: ${response_time}ms"
        
        # Alert if response time is very slow (> 5 seconds)
        if [[ $response_time -gt 5000 ]]; then
            log "WARN" "Slow health endpoint response: ${response_time}ms"
        fi
    fi
    
    return 0
}

# Main health check function
perform_health_check() {
    local exit_code=0
    local checks_passed=0
    local checks_total=8
    
    log "INFO" "Starting comprehensive health check"
    
    # Core health checks (failures cause health check to fail)
    if check_startup_complete; then
        ((checks_passed++))
    else
        exit_code=1
    fi
    
    if check_application_response; then
        ((checks_passed++))
    else
        exit_code=1
    fi
    
    if check_application_readiness; then
        ((checks_passed++))
    else
        exit_code=1
    fi
    
    if check_security_status; then
        ((checks_passed++))
    else
        exit_code=1
    fi
    
    if check_compliance_status; then
        ((checks_passed++))
    else
        exit_code=1
    fi
    
    # Informational checks (warnings only)
    if check_resource_usage; then
        ((checks_passed++))
    fi
    
    if check_metrics_endpoint; then
        ((checks_passed++))
    fi
    
    if check_performance_health; then
        ((checks_passed++))
    fi
    
    # Summary
    log "INFO" "Health check completed: $checks_passed/$checks_total checks passed"
    
    if [[ $exit_code -eq 0 ]]; then
        log "INFO" "Overall health status: HEALTHY"
        echo "healthy" > "$HEALTHY_FILE" 2>/dev/null || true
    else
        log "ERROR" "Overall health status: UNHEALTHY"
        rm -f "$HEALTHY_FILE" 2>/dev/null || true
    fi
    
    return $exit_code
}

# Execute health check
main() {
    # Set timeout for entire health check
    if ! timeout "$HEALTH_CHECK_TIMEOUT" perform_health_check; then
        log "ERROR" "Health check timed out after $HEALTH_CHECK_TIMEOUT seconds"
        exit 1
    fi
}

# Run main function
main "$@"