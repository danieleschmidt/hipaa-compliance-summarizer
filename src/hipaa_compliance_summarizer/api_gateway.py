"""Advanced API Gateway for HIPAA Compliance System.

This module provides a comprehensive API gateway including:
- RESTful endpoints for all system operations
- Authentication and authorization
- Rate limiting and throttling
- Request/response logging and monitoring
- Health checks and system status endpoints
- Real-time metrics and dashboard APIs
- WebSocket support for real-time updates
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from functools import wraps
import hashlib
import secrets
import uuid

# FastAPI and related imports (would need to be added to requirements)
try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    # Fallback for when FastAPI is not available
    FastAPI = None
    HTTPException = None
    logger = logging.getLogger(__name__)
    logger.warning("FastAPI not available - API gateway will run in mock mode")

# Import our system components
from .system_initialization import get_system_initializer, get_system_status
from .advanced_security import get_security_monitor
from .advanced_monitoring import get_advanced_monitor
from .advanced_error_handling import get_error_handler
from .distributed_processing import get_cluster_coordinator, submit_distributed_task
from .intelligent_autoscaling import get_intelligent_autoscaler


logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
if FastAPI:
    class TaskRequest(BaseModel):
        task_type: str = Field(..., description="Type of task to execute")
        payload: Dict[str, Any] = Field(..., description="Task payload data")
        priority: str = Field("NORMAL", description="Task priority level")
        
    class TaskResponse(BaseModel):
        task_id: str
        status: str
        message: str
        
    class HealthCheckResponse(BaseModel):
        status: str
        timestamp: str
        components: Dict[str, str]
        uptime_seconds: float
        
    class MetricsRequest(BaseModel):
        metrics: Dict[str, float] = Field(..., description="System metrics")
        
    class ScalingRequest(BaseModel):
        target_instances: int = Field(..., description="Target number of instances")
        reason: str = Field("manual", description="Reason for scaling")


class APIRateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self._request_counts: Dict[str, List[datetime]] = {}
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old requests
        if client_id in self._request_counts:
            self._request_counts[client_id] = [
                req_time for req_time in self._request_counts[client_id]
                if req_time > window_start
            ]
        else:
            self._request_counts[client_id] = []
        
        # Check limit
        if len(self._request_counts[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self._request_counts[client_id].append(now)
        return True


class APIAuthentication:
    """Simple API authentication system."""
    
    def __init__(self):
        """Initialize authentication."""
        # In production, these would be stored securely
        self._api_keys = {
            "hipaa_client_1": {
                "key_hash": hashlib.sha256("demo_api_key_123".encode()).hexdigest(),
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000
            }
        }
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return client info."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for client_id, client_info in self._api_keys.items():
            if client_info["key_hash"] == key_hash:
                return {
                    "client_id": client_id,
                    "permissions": client_info["permissions"],
                    "rate_limit": client_info["rate_limit"]
                }
        
        return None
    
    def has_permission(self, client_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if client has required permission."""
        return required_permission in client_info.get("permissions", [])


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class HIPAAAPIGateway:
    """Advanced API Gateway for HIPAA Compliance System."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize API Gateway."""
        self.config = config or {}
        self.auth = APIAuthentication()
        self.rate_limiter = APIRateLimiter(
            max_requests=self.config.get('rate_limit_requests', 100),
            window_minutes=self.config.get('rate_limit_window_minutes', 1)
        )
        self.websocket_manager = WebSocketManager()
        
        if not FastAPI:
            logger.warning("FastAPI not available - running in mock mode")
            self.app = None
            return
        
        self.app = self._create_fastapi_app()
        self._setup_routes()
        self._setup_middleware()
        
        # Start background services
        self._setup_background_tasks()
        
        logger.info("HIPAA API Gateway initialized")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("API Gateway starting up...")
            
            # Initialize system components
            system_initializer = get_system_initializer()
            if not system_initializer.status.ready_for_requests:
                logger.info("System not ready, initializing...")
                success = system_initializer.initialize_system(
                    production_mode=self.config.get('production_mode', False)
                )
                if not success:
                    logger.error("System initialization failed")
            
            yield
            
            # Shutdown
            logger.info("API Gateway shutting down...")
        
        app = FastAPI(
            title="HIPAA Compliance System API",
            description="Advanced API Gateway for HIPAA compliance processing",
            version="1.0.0",
            lifespan=lifespan,
            docs_url="/api/docs" if self.config.get('enable_docs', True) else None,
            redoc_url="/api/redoc" if self.config.get('enable_docs', True) else None
        )
        
        return app
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        if not self.app:
            return
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for logging and security
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            client_ip = request.client.host if request.client else "unknown"
            logger.info(f"Request: {request.method} {request.url} from {client_ip}")
            
            # Security monitoring
            security_monitor = get_security_monitor()
            security_monitor.log_security_event(
                event_type="api_request",
                severity="INFO",
                description=f"API request: {request.method} {request.url.path}",
                source_ip=client_ip,
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "user_agent": request.headers.get("user-agent", "")
                }
            )
            
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _get_current_client(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Dependency to get current authenticated client."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        client_info = self.auth.verify_api_key(credentials.credentials)
        if not client_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return client_info
    
    def _check_rate_limit(self, request: Request, client_info: Dict[str, Any]):
        """Check rate limiting for client."""
        client_id = client_info["client_id"]
        
        if not self.rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
    
    def _setup_routes(self):
        """Setup API routes."""
        if not self.app:
            return
        
        # Health check endpoints
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Basic health check endpoint."""
            try:
                system_status = get_system_status()
                
                return HealthCheckResponse(
                    status="healthy" if system_status["ready_for_requests"] else "unhealthy",
                    timestamp=datetime.now().isoformat(),
                    components={
                        "configuration": "healthy" if system_status["configuration_valid"] else "unhealthy",
                        "security": "healthy" if system_status["security_monitoring_active"] else "unhealthy",
                        "monitoring": "healthy" if system_status["advanced_monitoring_active"] else "unhealthy",
                        "error_handling": "healthy" if system_status["error_handling_active"] else "unhealthy"
                    },
                    uptime_seconds=system_status["uptime_seconds"]
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @self.app.get("/health/detailed")
        async def detailed_health_check(client_info: Dict = Depends(self._get_current_client)):
            """Detailed health check with component status."""
            if not self.auth.has_permission(client_info, "read"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                # Get detailed health information
                monitor = get_advanced_monitor()
                health_results = monitor.run_health_checks()
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "health_checks": {
                        name: result.to_dict() for name, result in health_results.items()
                    },
                    "system_metrics": monitor.collect_system_metrics().to_dict()
                }
            except Exception as e:
                logger.error(f"Detailed health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        # System status endpoints
        @self.app.get("/api/v1/status")
        async def get_system_status(client_info: Dict = Depends(self._get_current_client)):
            """Get comprehensive system status."""
            if not self.auth.has_permission(client_info, "read"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return get_system_status()
        
        @self.app.get("/api/v1/metrics")
        async def get_system_metrics(client_info: Dict = Depends(self._get_current_client)):
            """Get system metrics and monitoring data."""
            if not self.auth.has_permission(client_info, "read"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                monitor = get_advanced_monitor()
                return monitor.get_monitoring_dashboard()
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get metrics")
        
        @self.app.post("/api/v1/metrics")
        async def update_metrics(
            request: MetricsRequest,
            client_info: Dict = Depends(self._get_current_client)
        ):
            """Update system metrics (for external monitoring)."""
            if not self.auth.has_permission(client_info, "write"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                # Update auto-scaling metrics
                autoscaler = get_intelligent_autoscaler()
                autoscaler.update_metrics(request.metrics)
                
                # Record custom metrics in monitoring system
                monitor = get_advanced_monitor()
                for metric_name, value in request.metrics.items():
                    monitor.record_custom_metric(metric_name, value)
                
                return {"status": "success", "message": "Metrics updated"}
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to update metrics")
        
        # Task processing endpoints
        @self.app.post("/api/v1/tasks", response_model=TaskResponse)
        async def submit_task(
            request: TaskRequest,
            client_info: Dict = Depends(self._get_current_client)
        ):
            """Submit a task for distributed processing."""
            if not self.auth.has_permission(client_info, "write"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                from .distributed_processing import TaskPriority
                
                priority_map = {
                    "LOW": TaskPriority.LOW,
                    "NORMAL": TaskPriority.NORMAL,
                    "HIGH": TaskPriority.HIGH,
                    "URGENT": TaskPriority.URGENT
                }
                
                priority = priority_map.get(request.priority.upper(), TaskPriority.NORMAL)
                
                task_id = submit_distributed_task(
                    task_type=request.task_type,
                    payload=request.payload,
                    priority=priority
                )
                
                return TaskResponse(
                    task_id=task_id,
                    status="submitted",
                    message="Task submitted successfully"
                )
            except Exception as e:
                logger.error(f"Failed to submit task: {e}")
                raise HTTPException(status_code=500, detail="Failed to submit task")
        
        @self.app.get("/api/v1/tasks/{task_id}")
        async def get_task_status(
            task_id: str,
            client_info: Dict = Depends(self._get_current_client)
        ):
            """Get status of a specific task."""
            if not self.auth.has_permission(client_info, "read"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                coordinator = get_cluster_coordinator()
                task_status = coordinator.get_task_status(task_id)
                
                if not task_status:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return task_status
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get task status")
        
        # Auto-scaling endpoints
        @self.app.get("/api/v1/autoscaling")
        async def get_autoscaling_status(client_info: Dict = Depends(self._get_current_client)):
            """Get auto-scaling status and metrics."""
            if not self.auth.has_permission(client_info, "read"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                autoscaler = get_intelligent_autoscaler()
                return autoscaler.get_scaling_status()
            except Exception as e:
                logger.error(f"Failed to get autoscaling status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get autoscaling status")
        
        @self.app.post("/api/v1/autoscaling/scale")
        async def manual_scaling(
            request: ScalingRequest,
            client_info: Dict = Depends(self._get_current_client)
        ):
            """Manually trigger scaling action."""
            if not self.auth.has_permission(client_info, "admin"):
                raise HTTPException(status_code=403, detail="Admin permissions required")
            
            try:
                autoscaler = get_intelligent_autoscaler()
                success = autoscaler.force_scaling_decision(request.target_instances, request.reason)
                
                return {
                    "status": "success" if success else "failed",
                    "message": f"Scaling to {request.target_instances} instances {'succeeded' if success else 'failed'}"
                }
            except Exception as e:
                logger.error(f"Manual scaling failed: {e}")
                raise HTTPException(status_code=500, detail="Manual scaling failed")
        
        # Security endpoints
        @self.app.get("/api/v1/security/dashboard")
        async def get_security_dashboard(client_info: Dict = Depends(self._get_current_client)):
            """Get security monitoring dashboard."""
            if not self.auth.has_permission(client_info, "admin"):
                raise HTTPException(status_code=403, detail="Admin permissions required")
            
            try:
                security_monitor = get_security_monitor()
                return security_monitor.get_security_dashboard()
            except Exception as e:
                logger.error(f"Failed to get security dashboard: {e}")
                raise HTTPException(status_code=500, detail="Failed to get security dashboard")
        
        @self.app.post("/api/v1/security/block-ip")
        async def block_ip_address(
            request: dict,
            client_info: Dict = Depends(self._get_current_client)
        ):
            """Block an IP address."""
            if not self.auth.has_permission(client_info, "admin"):
                raise HTTPException(status_code=403, detail="Admin permissions required")
            
            try:
                ip_address = request.get("ip_address")
                reason = request.get("reason", "Manual block via API")
                
                if not ip_address:
                    raise HTTPException(status_code=400, detail="IP address required")
                
                security_monitor = get_security_monitor()
                success = security_monitor.manually_block_ip(ip_address, reason)
                
                return {
                    "status": "success" if success else "failed",
                    "message": f"IP {ip_address} {'blocked' if success else 'block failed'}"
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to block IP: {e}")
                raise HTTPException(status_code=500, detail="Failed to block IP")
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time system updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)
                    
                    # Get latest system status
                    status_update = {
                        "type": "status_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": get_system_status()
                    }
                    
                    await websocket.send_text(json.dumps(status_update))
            
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)
        
        # Dashboard endpoint
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """Simple dashboard HTML page."""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>HIPAA Compliance System Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .status-healthy { color: green; }
                    .status-unhealthy { color: red; }
                    #metrics { margin-top: 20px; }
                </style>
            </head>
            <body>
                <h1>HIPAA Compliance System Dashboard</h1>
                <div id="status">Loading...</div>
                <div id="metrics"></div>
                
                <script>
                    async function updateDashboard() {
                        try {
                            const response = await fetch('/health');
                            const data = await response.json();
                            
                            document.getElementById('status').innerHTML = `
                                <h2>System Status: <span class="status-${data.status}">${data.status.toUpperCase()}</span></h2>
                                <p>Uptime: ${Math.round(data.uptime_seconds)} seconds</p>
                            `;
                            
                            const metricsHtml = Object.entries(data.components).map(([component, status]) => 
                                `<div class="metric">${component}: <span class="status-${status}">${status}</span></div>`
                            ).join('');
                            
                            document.getElementById('metrics').innerHTML = `<h3>Components</h3>${metricsHtml}`;
                        } catch (error) {
                            document.getElementById('status').innerHTML = '<h2>Error loading dashboard</h2>';
                        }
                    }
                    
                    // Update dashboard every 5 seconds
                    updateDashboard();
                    setInterval(updateDashboard, 5000);
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
    
    def _setup_background_tasks(self):
        """Setup background tasks for the API gateway."""
        if not self.app:
            return
        
        # Background task to send WebSocket updates would go here
        # For now, updates are sent in the WebSocket endpoint loop
        pass
    
    async def broadcast_system_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast system event to WebSocket clients."""
        message = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        await self.websocket_manager.broadcast_message(message)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API gateway server."""
        if not self.app:
            logger.error("Cannot start server - FastAPI not available")
            return
        
        logger.info(f"Starting HIPAA API Gateway on {host}:{port}")
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level=self.config.get('log_level', 'info'),
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to start API Gateway: {e}")


# Global API gateway instance
_global_gateway: Optional[HIPAAAPIGateway] = None


def get_api_gateway(config: Optional[Dict[str, Any]] = None) -> HIPAAAPIGateway:
    """Get or create global API gateway."""
    global _global_gateway
    
    if _global_gateway is None:
        _global_gateway = HIPAAAPIGateway(config)
    
    return _global_gateway


def initialize_api_gateway(config: Optional[Dict[str, Any]] = None) -> HIPAAAPIGateway:
    """Initialize API gateway."""
    return get_api_gateway(config)


def run_api_gateway(host: str = "0.0.0.0", port: int = 8000, 
                   config: Optional[Dict[str, Any]] = None, **kwargs):
    """Run API gateway server."""
    gateway = get_api_gateway(config)
    gateway.run(host=host, port=port, **kwargs)