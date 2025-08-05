#!/bin/bash

# HIPAA Compliance Summarizer - Production Deployment Script
# Automated deployment with security, monitoring, and scaling

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Default values
ENVIRONMENT="production"
DEPLOYMENT_TYPE="docker"
SKIP_TESTS=false
SKIP_SECURITY_SCAN=false
ENABLE_MONITORING=true
DOMAIN=""
EMAIL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-security-scan)
            SKIP_SECURITY_SCAN=false
            shift
            ;;
        --no-monitoring)
            ENABLE_MONITORING=false
            shift
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV     Deployment environment (production, staging, development)"
            echo "  -t, --type TYPE          Deployment type (docker, kubernetes, local)"
            echo "  --skip-tests             Skip running tests before deployment"
            echo "  --skip-security-scan     Skip security vulnerability scanning"
            echo "  --no-monitoring          Disable monitoring and observability"
            echo "  -d, --domain DOMAIN      Domain name for the deployment"
            echo "  --email EMAIL            Email for SSL certificate registration"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(production|staging|development)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be production, staging, or development."
fi

# Validate deployment type
if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|kubernetes|local)$ ]]; then
    error "Invalid deployment type: $DEPLOYMENT_TYPE. Must be docker, kubernetes, or local."
fi

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed. Please install Docker first."
        fi
        
        if ! docker info &> /dev/null; then
            error "Docker is not running. Please start Docker first."
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose is not installed. Please install Docker Compose first."
        fi
    fi
    
    # Check if kubectl is installed for Kubernetes deployment
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            error "kubectl is not installed. Please install kubectl first."
        fi
        
        # Check if we can connect to Kubernetes cluster
        if ! kubectl cluster-info &> /dev/null; then
            error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        fi
    fi
    
    # Check if Python and required tools are available
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3 first."
    fi
    
    log "Prerequisites check completed successfully"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests as requested"
        return
    fi
    
    log "Running tests..."
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
    fi
    
    # Activate virtual environment and install dependencies
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    
    # Run core functionality tests
    python -m pytest tests/test_foundational.py tests/test_config.py tests/test_startup.py -v
    
    # Run advanced features tests
    python -m pytest tests/test_advanced_features.py -v
    
    # Run batch processing tests
    python -m pytest tests/test_batch_processor.py -v
    
    log "All tests passed successfully"
}

# Function to run security scan
run_security_scan() {
    if [[ "$SKIP_SECURITY_SCAN" == "true" ]]; then
        warn "Skipping security scan as requested"
        return
    fi
    
    log "Running security vulnerability scan..."
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Run bandit for Python security issues
    if command -v bandit &> /dev/null; then
        bandit -r src/ -f json -o security_scan_results.json || warn "Security scan found issues, check security_scan_results.json"
    fi
    
    # Run safety for dependency vulnerabilities
    if command -v safety &> /dev/null; then
        safety check --json --output dependency_audit_results.json || warn "Dependency audit found issues, check dependency_audit_results.json"
    fi
    
    log "Security scan completed"
}

# Function to build application
build_application() {
    log "Building application..."
    cd "$PROJECT_ROOT"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]] || [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Build Docker image
        docker build -t hipaa-compliance:latest .
        docker build -t hipaa-compliance:$ENVIRONMENT .
        
        # Tag image with timestamp
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        docker tag hipaa-compliance:latest hipaa-compliance:$TIMESTAMP
        
        log "Docker image built successfully: hipaa-compliance:$TIMESTAMP"
    else
        # Build wheel for local installation
        pip install build
        python -m build
        
        log "Python package built successfully"
    fi
}

# Function to generate environment file
generate_env_file() {
    log "Generating environment configuration..."
    
    ENV_FILE="${DEPLOY_DIR}/.env.${ENVIRONMENT}"
    
    # Generate random passwords if not set
    DB_PASSWORD=${DB_PASSWORD:-$(openssl rand -base64 32)}
    REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
    JWT_SECRET=${JWT_SECRET:-$(openssl rand -base64 64)}
    GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-$(openssl rand -base64 16)}
    
    cat > "$ENV_FILE" << EOF
# HIPAA Compliance Summarizer - ${ENVIRONMENT} Environment
ENVIRONMENT=${ENVIRONMENT}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE}
DEPLOYMENT_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Database Configuration
DB_PASSWORD=${DB_PASSWORD}
DATABASE_URL=postgresql://hipaa_user:${DB_PASSWORD}@postgres:5432/hipaa_db

# Redis Configuration
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# Security Configuration
JWT_SECRET_KEY=${JWT_SECRET}
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Monitoring Configuration
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
ENABLE_MONITORING=${ENABLE_MONITORING}

# Domain Configuration
DOMAIN=${DOMAIN}
EMAIL=${EMAIL}

# Application Configuration
WORKERS=4
MAX_WORKERS=10
ENABLE_AUTO_SCALING=true
ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO
EOF

    chmod 600 "$ENV_FILE"
    log "Environment file created: $ENV_FILE"
}

# Function to deploy with Docker
deploy_docker() {
    log "Deploying with Docker Compose..."
    cd "$DEPLOY_DIR"
    
    # Source environment variables
    set -a
    source ".env.${ENVIRONMENT}"
    set +a
    
    # Stop existing deployment
    docker-compose -f docker-compose.production.yml down || true
    
    # Deploy with Docker Compose
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    if docker-compose -f docker-compose.production.yml ps | grep -q "unhealthy"; then
        error "Some services are unhealthy. Check docker-compose logs."
    fi
    
    log "Docker deployment completed successfully"
    
    # Display access information
    info "=== Deployment Information ==="
    info "API Endpoint: http://localhost:8000"
    info "Health Check: http://localhost:8000/health"
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        info "Grafana Dashboard: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
        info "Prometheus: http://localhost:9090"
    fi
    info "==========================="
}

# Function to deploy with Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    cd "$DEPLOY_DIR/kubernetes"
    
    # Create namespace if it doesn't exist
    kubectl create namespace hipaa-system --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic hipaa-secrets \
        --from-literal=database-url="postgresql://hipaa_user:${DB_PASSWORD}@postgres:5432/hipaa_db" \
        --from-literal=redis-url="redis://:${REDIS_PASSWORD}@redis:6379/0" \
        --from-literal=jwt-secret="${JWT_SECRET}" \
        --namespace=hipaa-system \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/hipaa-compliance-api -n hipaa-system
    kubectl wait --for=condition=available --timeout=300s deployment/hipaa-compliance-worker -n hipaa-system
    
    log "Kubernetes deployment completed successfully"
    
    # Display access information
    info "=== Kubernetes Deployment Information ==="
    info "Namespace: hipaa-system"
    info "API Service: hipaa-compliance-api-service"
    info "Check status: kubectl get pods -n hipaa-system"
    if [[ -n "$DOMAIN" ]]; then
        info "External Access: https://${DOMAIN}"
    fi
    info "======================================="
}

# Function to deploy locally
deploy_local() {
    log "Deploying locally..."
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install package in development mode
    pip install -e .
    
    # Start the application
    info "Starting HIPAA Compliance Summarizer locally..."
    info "Use the following commands to interact with the system:"
    info "  hipaa-summarize --help"
    info "  hipaa-batch-process --help"
    info "  hipaa-compliance-report --help"
    
    log "Local deployment completed successfully"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    log "Running post-deployment tests..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        # Test API endpoint
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log "API health check passed"
        else
            error "API health check failed"
        fi
        
        # Test CLI functionality
        if docker-compose -f "${DEPLOY_DIR}/docker-compose.production.yml" exec -T hipaa-api hipaa-summarize --help > /dev/null 2>&1; then
            log "CLI functionality test passed"
        else
            warn "CLI functionality test failed (this might be expected in containerized environment)"
        fi
    fi
    
    log "Post-deployment tests completed"
}

# Function to setup monitoring
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        warn "Monitoring is disabled"
        return
    fi
    
    log "Setting up monitoring and observability..."
    
    # Create monitoring configuration files if they don't exist
    mkdir -p "${DEPLOY_DIR}/monitoring"
    
    # Prometheus configuration
    if [[ ! -f "${DEPLOY_DIR}/prometheus.yml" ]]; then
        cat > "${DEPLOY_DIR}/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hipaa-api'
    static_configs:
      - targets: ['hipaa-api:8000']
    metrics_path: '/metrics'
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    fi
    
    log "Monitoring setup completed"
}

# Function to create backup strategy
setup_backup_strategy() {
    log "Setting up backup strategy..."
    
    mkdir -p "${DEPLOY_DIR}/backup"
    
    # Create backup script
    cat > "${DEPLOY_DIR}/backup/backup.sh" << 'EOF'
#!/bin/bash

# HIPAA Compliance Summarizer - Backup Script
BACKUP_DIR="/var/backups/hipaa-compliance"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup database
if command -v pg_dump &> /dev/null; then
    pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database-$TIMESTAMP.sql"
fi

# Backup configuration
cp -r /app/config "$BACKUP_DIR/config-$TIMESTAMP"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "config-*" -mtime +7 -exec rm -rf {} +

echo "Backup completed: $TIMESTAMP"
EOF
    
    chmod +x "${DEPLOY_DIR}/backup/backup.sh"
    
    log "Backup strategy setup completed"
}

# Main deployment function
main() {
    log "Starting HIPAA Compliance Summarizer deployment..."
    log "Environment: $ENVIRONMENT"
    log "Deployment Type: $DEPLOYMENT_TYPE"
    
    # Run deployment steps
    check_prerequisites
    run_tests
    run_security_scan
    build_application
    generate_env_file
    setup_monitoring
    setup_backup_strategy
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        local)
            deploy_local
            ;;
    esac
    
    # Run post-deployment tests
    if [[ "$DEPLOYMENT_TYPE" != "local" ]]; then
        run_post_deployment_tests
    fi
    
    log "Deployment completed successfully!"
    log "Environment: $ENVIRONMENT"
    log "Type: $DEPLOYMENT_TYPE"
    log "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    # Show next steps
    info "=== Next Steps ==="
    info "1. Review logs: docker-compose logs -f (for Docker) or kubectl logs -f deployment/hipaa-compliance-api -n hipaa-system (for Kubernetes)"
    info "2. Monitor health: Check /health endpoint"
    info "3. Configure SSL certificates if using custom domain"
    info "4. Set up log aggregation and alerting"
    info "5. Schedule regular backups"
    info "=================="
}

# Run main function
main "$@"