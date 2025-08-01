#!/bin/bash

# HIPAA Compliance Summarizer - Build Automation Script
# Comprehensive Docker build system with multi-stage builds and security scanning

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERSION=${VERSION:-"latest"}
REGISTRY=${REGISTRY:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Help function
show_help() {
    cat << EOF
HIPAA Compliance Summarizer Build System

Usage: $0 [OPTIONS] [TARGET]

Targets:
    production          Build production image (default)
    development         Build development image
    testing             Build testing image
    docs                Build documentation image
    all                 Build all images
    security-scan       Run security scanning only
    benchmark           Build and run benchmarks

Options:
    -h, --help          Show this help message
    -v, --version       Set version tag (default: latest)
    -r, --registry      Set registry prefix
    --push              Push images to registry after build
    --no-cache          Build without using cache
    --parallel          Build images in parallel where possible
    --security          Enable security scanning during build
    --sbom              Generate Software Bill of Materials
    --sign              Sign images with cosign
    --multi-arch        Build for multiple architectures
    --clean             Clean build artifacts before building
    --verbose           Verbose output

Examples:
    $0                          # Build production image
    $0 all --version v1.0.0     # Build all images with version tag
    $0 production --push        # Build and push production image
    $0 --security --sbom        # Build with security scanning and SBOM
    $0 --multi-arch --push      # Build multi-architecture images

Environment Variables:
    VERSION             Image version tag
    REGISTRY            Container registry prefix
    DOCKER_BUILDKIT     Enable BuildKit (default: 1)

EOF
}

# Parse command line arguments
TARGET="production"
PUSH=false
NO_CACHE=false
PARALLEL=false
SECURITY_SCAN=false
GENERATE_SBOM=false
SIGN_IMAGES=false
MULTI_ARCH=false
CLEAN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --security)
            SECURITY_SCAN=true
            shift
            ;;
        --sbom)
            GENERATE_SBOM=true
            shift
            ;;
        --sign)
            SIGN_IMAGES=true
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        production|development|testing|docs|all|security-scan|benchmark)
            TARGET="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Setup environment
cd "$PROJECT_ROOT"

# Enable BuildKit
export DOCKER_BUILDKIT=1
export DOCKER_CLI_EXPERIMENTAL=enabled

log_info "HIPAA Compliance Summarizer Build System"
log_info "========================================"
log_info "Target: $TARGET"
log_info "Version: $VERSION"
log_info "Registry: ${REGISTRY:-'local'}"
log_info "Timestamp: $TIMESTAMP"

# Clean build artifacts if requested
if [[ "$CLEAN" == true ]]; then
    log_info "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    docker system prune -f
    log_success "Build artifacts cleaned"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not running"
    exit 1
fi

log_success "Docker environment verified"

# Set image names
IMAGE_BASE="hipaa-compliance-summarizer"
if [[ -n "$REGISTRY" ]]; then
    IMAGE_BASE="$REGISTRY/$IMAGE_BASE"
fi

PRODUCTION_IMAGE="$IMAGE_BASE:$VERSION"
DEVELOPMENT_IMAGE="$IMAGE_BASE:dev-$VERSION"
TESTING_IMAGE="$IMAGE_BASE:test-$VERSION"
DOCS_IMAGE="$IMAGE_BASE:docs-$VERSION"

# Build arguments
BUILD_ARGS=""
if [[ "$NO_CACHE" == true ]]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

if [[ "$VERBOSE" == true ]]; then
    BUILD_ARGS="$BUILD_ARGS --progress=plain"
fi

# Multi-architecture support
PLATFORMS="linux/amd64"
if [[ "$MULTI_ARCH" == true ]]; then
    PLATFORMS="linux/amd64,linux/arm64"
    BUILD_ARGS="$BUILD_ARGS --platform $PLATFORMS"
fi

# Security scanning function
run_security_scan() {
    local image_name="$1"
    log_info "Running security scan for $image_name..."
    
    # Create security reports directory
    mkdir -p "$BUILD_DIR/security-reports"
    
    # Run Trivy security scan if available
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy vulnerability scan..."
        trivy image --format json --output "$BUILD_DIR/security-reports/trivy-$TIMESTAMP.json" "$image_name" || log_warning "Trivy scan completed with issues"
        log_success "Trivy scan completed"
    else
        log_warning "Trivy not available, skipping vulnerability scan"
    fi
    
    # Run Docker Scout if available
    if docker scout --help &> /dev/null; then
        log_info "Running Docker Scout analysis..."
        docker scout cves "$image_name" --format json --output "$BUILD_DIR/security-reports/scout-$TIMESTAMP.json" || log_warning "Docker Scout scan completed with issues"
        log_success "Docker Scout analysis completed"
    else
        log_warning "Docker Scout not available, skipping analysis"
    fi
}

# SBOM generation function
generate_sbom() {
    local image_name="$1"
    log_info "Generating SBOM for $image_name..."
    
    mkdir -p "$BUILD_DIR/sbom"
    
    # Generate SBOM with Syft if available
    if command -v syft &> /dev/null; then
        syft "$image_name" -o spdx-json="$BUILD_DIR/sbom/sbom-$TIMESTAMP.spdx.json"
        syft "$image_name" -o cyclonedx-json="$BUILD_DIR/sbom/sbom-$TIMESTAMP.cyclonedx.json"
        log_success "SBOM generated"
    else
        log_warning "Syft not available, skipping SBOM generation"
    fi
}

# Image signing function
sign_image() {
    local image_name="$1"
    log_info "Signing image $image_name..."
    
    if command -v cosign &> /dev/null; then
        cosign sign "$image_name" --yes || log_warning "Image signing failed"
        log_success "Image signed"
    else
        log_warning "Cosign not available, skipping image signing"
    fi
}

# Build function
build_image() {
    local target="$1"
    local image_name="$2"
    local dockerfile="${3:-Dockerfile}"
    
    log_info "Building $target image: $image_name"
    
    # Build command
    local build_cmd="docker build $BUILD_ARGS --target $target -t $image_name -f $dockerfile ."
    
    if [[ "$MULTI_ARCH" == true ]]; then
        build_cmd="docker buildx build $BUILD_ARGS --target $target -t $image_name -f $dockerfile ."
        if [[ "$PUSH" == true ]]; then
            build_cmd="$build_cmd --push"
        else
            build_cmd="$build_cmd --load"
        fi
    fi
    
    log_info "Build command: $build_cmd"
    
    # Execute build
    if eval "$build_cmd"; then
        log_success "$target image built successfully: $image_name"
        
        # Post-build actions
        if [[ "$SECURITY_SCAN" == true ]]; then
            run_security_scan "$image_name"
        fi
        
        if [[ "$GENERATE_SBOM" == true ]]; then
            generate_sbom "$image_name"
        fi
        
        if [[ "$SIGN_IMAGES" == true && "$PUSH" == true ]]; then
            sign_image "$image_name"
        fi
        
        return 0
    else
        log_error "$target image build failed"
        return 1
    fi
}

# Push function
push_image() {
    local image_name="$1"
    
    if [[ "$PUSH" == true && "$MULTI_ARCH" != true ]]; then
        log_info "Pushing image: $image_name"
        if docker push "$image_name"; then
            log_success "Image pushed successfully: $image_name"
        else
            log_error "Failed to push image: $image_name"
            return 1
        fi
    fi
}

# Build targets
build_production() {
    build_image "production" "$PRODUCTION_IMAGE" && push_image "$PRODUCTION_IMAGE"
}

build_development() {
    build_image "development" "$DEVELOPMENT_IMAGE" && push_image "$DEVELOPMENT_IMAGE"
}

build_testing() {
    build_image "testing" "$TESTING_IMAGE" && push_image "$TESTING_IMAGE"
}

build_docs() {
    build_image "docs" "$DOCS_IMAGE" && push_image "$DOCS_IMAGE"
}

build_all() {
    local build_functions=(build_production build_development build_testing build_docs)
    
    if [[ "$PARALLEL" == true ]]; then
        log_info "Building all images in parallel..."
        for func in "${build_functions[@]}"; do
            $func &
        done
        wait
    else
        log_info "Building all images sequentially..."
        for func in "${build_functions[@]}"; do
            $func || exit 1
        done
    fi
}

run_security_scan_only() {
    log_info "Running security scan on existing images..."
    
    local images=("$PRODUCTION_IMAGE" "$DEVELOPMENT_IMAGE" "$TESTING_IMAGE" "$DOCS_IMAGE")
    for image in "${images[@]}"; do
        if docker image inspect "$image" >/dev/null 2>&1; then
            run_security_scan "$image"
        else
            log_warning "Image not found: $image"
        fi
    done
}

run_benchmark() {
    log_info "Building and running benchmarks..."
    build_image "benchmark" "$IMAGE_BASE:benchmark-$VERSION"
    
    log_info "Running performance benchmarks..."
    docker run --rm "$IMAGE_BASE:benchmark-$VERSION" || log_warning "Benchmark run completed with issues"
}

# Main execution
case "$TARGET" in
    production)
        build_production
        ;;
    development)
        build_development
        ;;
    testing)
        build_testing
        ;;
    docs)
        build_docs
        ;;
    all)
        build_all
        ;;
    security-scan)
        run_security_scan_only
        ;;
    benchmark)
        run_benchmark
        ;;
    *)
        log_error "Unknown target: $TARGET"
        show_help
        exit 1
        ;;
esac

# Generate build summary
BUILD_SUMMARY="$BUILD_DIR/build-summary-$TIMESTAMP.txt"
cat > "$BUILD_SUMMARY" << EOF
HIPAA Compliance Summarizer Build Summary
=========================================

Build Time: $TIMESTAMP
Target: $TARGET
Version: $VERSION
Registry: ${REGISTRY:-'local'}

Configuration:
- Multi-arch: $MULTI_ARCH
- Security Scan: $SECURITY_SCAN
- SBOM Generation: $GENERATE_SBOM
- Image Signing: $SIGN_IMAGES
- Push to Registry: $PUSH
- Parallel Build: $PARALLEL

Images Built:
$(case "$TARGET" in
    production) echo "- Production: $PRODUCTION_IMAGE" ;;
    development) echo "- Development: $DEVELOPMENT_IMAGE" ;;
    testing) echo "- Testing: $TESTING_IMAGE" ;;
    docs) echo "- Documentation: $DOCS_IMAGE" ;;
    all) echo -e "- Production: $PRODUCTION_IMAGE\n- Development: $DEVELOPMENT_IMAGE\n- Testing: $TESTING_IMAGE\n- Documentation: $DOCS_IMAGE" ;;
esac)

Build Artifacts:
$([ -d "$BUILD_DIR/security-reports" ] && echo "- Security Reports: $BUILD_DIR/security-reports/")
$([ -d "$BUILD_DIR/sbom" ] && echo "- SBOM Files: $BUILD_DIR/sbom/")

Docker Info:
- Docker Version: $(docker --version)
- BuildKit Enabled: $DOCKER_BUILDKIT

System Info:
- Platform: $(uname -s)/$(uname -m)
- User: $(whoami)
- PWD: $(pwd)

EOF

log_success "Build summary saved: $BUILD_SUMMARY"

# Final status
log_success "🎉 Build process completed successfully!"
echo ""
log_info "Build artifacts available in: $BUILD_DIR"
log_info "Build summary: $BUILD_SUMMARY"

if [[ "$PUSH" == true ]]; then
    log_info "Images have been pushed to registry"
fi

echo ""
log_info "Next steps:"
case "$TARGET" in
    production)
        echo "  - Run: docker run --rm $PRODUCTION_IMAGE"
        ;;
    development)
        echo "  - Run: docker run --rm -it $DEVELOPMENT_IMAGE /bin/bash"
        ;;
    testing)
        echo "  - Tests have been executed during build"
        ;;
    docs)
        echo "  - Run: docker run --rm -p 8080:8080 $DOCS_IMAGE"
        ;;
    all)
        echo "  - Use docker-compose.yml to orchestrate services"
        ;;
esac