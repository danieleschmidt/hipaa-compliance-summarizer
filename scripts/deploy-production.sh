#!/bin/bash

# HIPAA Compliance Summarizer - Production Deployment Script
# This script handles secure deployment of the HIPAA compliance system
# to production environments with proper security and validation checks.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
SKIP_HEALTH_CHECK="${SKIP_HEALTH_CHECK:-false}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Cleaning up..."
        # Add cleanup logic here
    fi
}
trap cleanup EXIT

# Validation functions
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD" 
        "ENCRYPTION_KEY"
        "JWT_SECRET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            return 1
        fi
    done
    
    # Validate encryption key strength
    if [[ ${#ENCRYPTION_KEY} -lt 32 ]]; then
        log_error "ENCRYPTION_KEY must be at least 32 characters long"
        return 1
    fi
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        return 1
    fi
    
    log_success "Environment validation passed"
}

validate_configuration() {
    log_info "Validating configuration files..."
    
    local config_files=(
        "$PROJECT_ROOT/config/production.yml"
        "$PROJECT_ROOT/docker-compose.prod.yml"
        "$PROJECT_ROOT/nginx/nginx.conf"
    )
    
    for file in "${config_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required configuration file not found: $file"
            return 1
        fi
    done
    
    # Validate SSL certificates
    if [[ ! -f "$PROJECT_ROOT/nginx/ssl/cert.pem" ]] || [[ ! -f "$PROJECT_ROOT/nginx/ssl/key.pem" ]]; then
        log_warning "SSL certificates not found. HTTPS will not work properly."
        log_warning "Generate certificates with: openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem"
    fi
    
    log_success "Configuration validation passed"
}

backup_database() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Skipping database backup (BACKUP_BEFORE_DEPLOY=false)"
        return 0
    fi
    
    log_info "Creating database backup before deployment..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Check if database container is running
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps postgres | grep -q "Up"; then
        docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres \
            pg_dump -U hipaa_user hipaa_db > "$backup_dir/database.sql" || {
            log_error "Database backup failed"
            return 1
        }
        log_success "Database backup created: $backup_dir/database.sql"
    else
        log_warning "Database container not running, skipping backup"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image_name="$DOCKER_REGISTRY/hipaa-compliance-summarizer:$IMAGE_TAG"
        docker build -f Dockerfile.production -t "$image_name" .
        
        log_info "Pushing image to registry..."
        docker push "$image_name"
        
        # Update docker-compose to use registry image
        export HIPAA_IMAGE="$image_name"
    else
        docker build -f Dockerfile.production -t "hipaa-compliance-summarizer:$IMAGE_TAG" .
    fi
    
    log_success "Docker images built successfully"
}

deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables for docker-compose
    export COMPOSE_PROJECT_NAME="hipaa-compliance"
    export COMPOSE_FILE="docker-compose.prod.yml"
    
    # Deploy with docker-compose
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.prod.yml up -d --remove-orphans
    else
        docker compose -f docker-compose.prod.yml up -d --remove-orphans
    fi
    
    log_success "Services deployed successfully"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check if main application is responding
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Application is responding"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to start within expected time"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

run_health_checks() {
    if [[ "$SKIP_HEALTH_CHECK" == "true" ]]; then
        log_info "Skipping health checks (SKIP_HEALTH_CHECK=true)"
        return 0
    fi
    
    log_info "Running comprehensive health checks..."
    
    # Application health check
    local health_response
    health_response=$(curl -s http://localhost:8000/health || echo "FAILED")
    
    if [[ "$health_response" == *"healthy"* ]]; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed"
        return 1
    fi
    
    # Database connectivity check
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres \
       pg_isready -U hipaa_user -d hipaa_db > /dev/null 2>&1; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Redis connectivity check
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T redis \
       redis-cli ping > /dev/null 2>&1; then
        log_success "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Start Prometheus and Grafana
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps prometheus | grep -q "Up"; then
        log_success "Prometheus is running"
    else
        log_warning "Prometheus is not running"
    fi
    
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps grafana | grep -q "Up"; then
        log_success "Grafana is running"
        log_info "Grafana dashboard: http://localhost:3000"
    else
        log_warning "Grafana is not running"
    fi
}

setup_ssl_certificates() {
    log_info "Setting up SSL certificates..."
    
    local ssl_dir="$PROJECT_ROOT/nginx/ssl"
    mkdir -p "$ssl_dir"
    
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        log_warning "SSL certificates not found. Generating self-signed certificates..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$ssl_dir/key.pem" \
            -out "$ssl_dir/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=IT/CN=localhost" 2>/dev/null || {
            log_error "Failed to generate SSL certificates"
            return 1
        }
        
        # Set proper permissions
        chmod 600 "$ssl_dir/key.pem"
        chmod 644 "$ssl_dir/cert.pem"
        
        log_success "Self-signed SSL certificates generated"
        log_warning "For production, replace with proper SSL certificates from a CA"
    else
        log_success "SSL certificates found"
    fi
}

display_deployment_info() {
    log_success "Deployment completed successfully!"
    echo
    echo "=== Deployment Information ==="
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Image Tag: $IMAGE_TAG"
    echo "Project Root: $PROJECT_ROOT"
    echo
    echo "=== Service URLs ==="
    echo "Application: https://localhost (HTTP redirects to HTTPS)"
    echo "Health Check: http://localhost:8000/health"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
    echo
    echo "=== Service Status ==="
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps
    else
        docker compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps
    fi
    echo
    echo "=== Next Steps ==="
    echo "1. Configure DNS to point to this server"
    echo "2. Replace self-signed SSL certificates with CA-signed certificates"
    echo "3. Configure backup destinations (S3, etc.)"
    echo "4. Set up log aggregation and monitoring"
    echo "5. Run security hardening checklist"
    echo
    echo "=== Security Reminders ==="
    echo "- Change default passwords for Grafana and any other services"
    echo "- Review and update security configurations"
    echo "- Enable firewall and restrict access to necessary ports"
    echo "- Set up regular security updates"
}

# Main deployment flow
main() {
    log_info "Starting HIPAA Compliance Summarizer deployment..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Image Tag: $IMAGE_TAG"
    
    validate_environment
    validate_configuration
    setup_ssl_certificates
    backup_database
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    setup_monitoring
    display_deployment_info
    
    log_success "Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "backup")
        backup_database
        ;;
    "health-check")
        run_health_checks
        ;;
    "ssl")
        setup_ssl_certificates
        ;;
    "build")
        build_images
        ;;
    "help")
        echo "Usage: $0 [deploy|backup|health-check|ssl|build|help]"
        echo
        echo "Commands:"
        echo "  deploy       Full deployment (default)"
        echo "  backup       Create database backup only"
        echo "  health-check Run health checks only"
        echo "  ssl          Generate SSL certificates only"
        echo "  build        Build Docker images only"
        echo "  help         Show this help message"
        echo
        echo "Environment Variables:"
        echo "  DEPLOYMENT_ENV     Deployment environment (default: production)"
        echo "  DOCKER_REGISTRY    Docker registry URL (optional)"
        echo "  IMAGE_TAG         Docker image tag (default: latest)"
        echo "  BACKUP_BEFORE_DEPLOY  Create backup before deploy (default: true)"
        echo "  SKIP_HEALTH_CHECK  Skip health checks (default: false)"
        echo
        echo "Required Environment Variables:"
        echo "  POSTGRES_PASSWORD  Database password"
        echo "  REDIS_PASSWORD     Redis password"
        echo "  ENCRYPTION_KEY     Encryption key (min 32 chars)"
        echo "  JWT_SECRET_KEY     JWT secret key"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac