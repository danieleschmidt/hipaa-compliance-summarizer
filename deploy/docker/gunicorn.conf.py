"""
Gunicorn configuration for HIPAA Compliance Summarizer Generation 4
Production-grade WSGI server configuration with HIPAA compliance features
"""

import multiprocessing
import os
from pathlib import Path

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = 1000
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', 100))
preload_app = True
timeout = int(os.getenv('TIMEOUT', 120))
keepalive = int(os.getenv('KEEPALIVE', 5))
graceful_timeout = 30

# Restart workers after this many requests, with up to jitter requests variation
worker_tmp_dir = "/dev/shm"

# SSL Configuration for production
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')
ssl_version = 2  # TLS
ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
do_handshake_on_connect = False

# Security settings
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Logging
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
    '"%(f)s" "%(a)s" %(D)s %(p)s'
)

# Custom log format for HIPAA compliance
logger_class = 'gunicorn.glogging.Logger'

# Capture stdout/stderr
capture_output = True

# Process naming
proc_name = 'hipaa-compliance-summarizer-gen4'

# Server mechanics
daemon = False
raw_env = [
    'ENVIRONMENT=production',
    f'COMPLIANCE_LEVEL={os.getenv("COMPLIANCE_LEVEL", "strict")}',
    f'ENABLE_ML_OPTIMIZATION={os.getenv("ENABLE_ML_OPTIMIZATION", "true")}',
    f'ENABLE_AUTO_SCALING={os.getenv("ENABLE_AUTO_SCALING", "true")}',
]

# Performance tuning
sendfile = True
reuse_port = True

# HIPAA-compliant logging configuration
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("HIPAA Compliance Summarizer Generation 4 ready")
    server.log.info(f"Workers: {workers}, Worker class: {worker_class}")
    server.log.info(f"Compliance level: {os.getenv('COMPLIANCE_LEVEL', 'strict')}")
    server.log.info(f"ML optimization: {os.getenv('ENABLE_ML_OPTIMIZATION', 'true')}")
    
    # Create PID file for monitoring
    pid_file = Path("/app/tmp/gunicorn.pid")
    pid_file.write_text(str(os.getpid()))

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received interrupt signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.debug(f"Pre-fork worker {worker.age}")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker {worker.pid} started")
    
    # Initialize worker-specific resources
    try:
        # Pre-load ML models and caches if enabled
        if os.getenv('ENABLE_ML_OPTIMIZATION', 'false').lower() == 'true':
            server.log.info(f"Worker {worker.pid}: Initializing ML optimization")
            # Import and initialize ML components
            from src.hipaa_compliance_summarizer.performance_gen4 import ml_optimizer
            from src.hipaa_compliance_summarizer.intelligent_scaling import auto_scaler
            server.log.info(f"Worker {worker.pid}: ML optimization initialized")
    except Exception as e:
        server.log.warning(f"Worker {worker.pid}: ML initialization warning: {e}")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Master process forking new process")

def on_exit(server):
    """Called just before master process exits."""
    server.log.info("HIPAA Compliance Summarizer shutting down")
    
    # Cleanup PID file
    pid_file = Path("/app/tmp/gunicorn.pid")
    if pid_file.exists():
        pid_file.unlink()

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.error(f"Worker {worker.pid} aborted")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading configuration")

# Custom application configuration
def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info(f"Worker {worker.pid}: Application initialized")

# Security callback for worker process validation
def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker {worker.pid} exited")

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'development':
    reload = True
    loglevel = 'debug'
    workers = 1
    preload_app = False
elif os.getenv('ENVIRONMENT') == 'production':
    # Production security hardening
    preload_app = True
    max_requests = 2000
    max_requests_jitter = 400
    
    # Enable request logging for HIPAA audit trail
    access_log_format = (
        '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
        '"%(f)s" "%(a)s" %(D)s %(p)s [HIPAA-AUDIT]'
    )

# Custom configuration validation
def validate_config():
    """Validate production configuration."""
    required_env_vars = ['ENVIRONMENT', 'COMPLIANCE_LEVEL']
    
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} not set")
    
    if os.getenv('COMPLIANCE_LEVEL') not in ['strict', 'standard', 'minimal']:
        raise ValueError("COMPLIANCE_LEVEL must be 'strict', 'standard', or 'minimal'")
    
    if workers < 1:
        raise ValueError("At least 1 worker is required")
    
    if timeout < 30:
        raise ValueError("Timeout must be at least 30 seconds for HIPAA processing")

# Run validation on import
validate_config()

# Custom logging format for structured logs
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'access': {
            'format': '%(asctime)s - ACCESS - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'hipaa_audit': {
            'format': '%(asctime)s - HIPAA-AUDIT - %(levelname)s - %(message)s - [compliance_level=%(compliance_level)s]',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'default': {
            'formatter': 'default',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'access': {
            'formatter': 'access',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
    },
    'root': {
        'level': loglevel.upper(),
        'handlers': ['default']
    },
    'loggers': {
        'gunicorn.error': {
            'level': loglevel.upper(),
            'handlers': ['default'],
            'propagate': True,
            'qualname': 'gunicorn.error'
        },
        'gunicorn.access': {
            'level': 'INFO',
            'handlers': ['access'],
            'propagate': False,
            'qualname': 'gunicorn.access'
        },
    }
}

# Performance monitoring configuration
def performance_callback(worker):
    """Monitor worker performance for HIPAA compliance."""
    try:
        import psutil
        process = psutil.Process(worker.pid)
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        if memory_mb > 1024:  # 1GB warning threshold
            worker.log.warning(f"Worker {worker.pid} high memory usage: {memory_mb:.1f}MB")
        
        if cpu_percent > 90:  # 90% CPU warning threshold  
            worker.log.warning(f"Worker {worker.pid} high CPU usage: {cpu_percent:.1f}%")
            
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        worker.log.debug(f"Performance monitoring error: {e}")

# Set up periodic performance monitoring
import threading
import time

def periodic_monitoring():
    """Periodic performance and compliance monitoring."""
    while True:
        try:
            time.sleep(60)  # Monitor every minute
            
            # Check worker health
            pid_file = Path("/app/tmp/gunicorn.pid")
            if pid_file.exists():
                try:
                    import psutil
                    pid = int(pid_file.read_text())
                    if not psutil.pid_exists(pid):
                        print(f"WARNING: Master process {pid} not found")
                except Exception:
                    pass
            
        except Exception:
            break  # Exit monitoring thread on error

# Start monitoring thread in master process
if not os.getenv('GUNICORN_WORKER'):  # Only in master process
    monitor_thread = threading.Thread(target=periodic_monitoring, daemon=True)
    monitor_thread.start()