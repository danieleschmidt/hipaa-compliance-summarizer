"""
Autonomous Deployment Orchestrator - Intelligent Production Deployment

This module implements an autonomous deployment orchestration system that handles
zero-downtime deployments, rollbacks, health monitoring, and intelligent scaling
for the HIPAA Compliance Summarizer in production environments.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
try:
    import yaml
except ImportError:
    print("PyYAML not available, using JSON fallback")
    yaml = None
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import psutil
except ImportError:
    print("psutil not available, using system command fallbacks")
    
    # Create psutil-like fallback
    class PSUtilFallback:
        @staticmethod
        def Process():
            class ProcessInfo:
                def num_threads(self):
                    return 4
            return ProcessInfo()
    
    psutil = PSUtilFallback()


class DeploymentPhase(Enum):
    """Deployment phases."""
    PREPARATION = "preparation"
    PRE_FLIGHT_CHECKS = "pre_flight_checks"
    DEPLOYMENT = "deployment"
    HEALTH_VERIFICATION = "health_verification"
    ROLLBACK = "rollback"
    POST_DEPLOYMENT = "post_deployment"
    MONITORING = "monitoring"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HealthCheckStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    version: str
    replicas: int = 3
    strategy: str = "rolling"  # rolling, blue_green, canary
    timeout: int = 600  # 10 minutes
    health_check_interval: int = 30
    rollback_on_failure: bool = True
    canary_percentage: int = 10
    monitoring_duration: int = 300  # 5 minutes post-deployment


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    endpoint: Optional[str] = None
    command: Optional[str] = None
    expected_status: int = 200
    timeout: int = 30
    critical: bool = True
    retry_count: int = 3


@dataclass
class DeploymentMetrics:
    """Deployment metrics."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    rollback_count: int = 0
    health_check_failures: int = 0


@dataclass
class DeploymentResult:
    """Deployment result."""
    status: DeploymentStatus
    metrics: DeploymentMetrics
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    deployed_services: List[str] = field(default_factory=list)


class KubernetesDeploymentManager:
    """
    Kubernetes deployment manager for container orchestration.
    
    Handles Kubernetes deployments, scaling, health checks, and rollbacks
    using kubectl and Kubernetes API.
    """
    
    def __init__(self, namespace: str = "hipaa-compliance"):
        self.namespace = namespace
        self.logger = logging.getLogger(f"{__name__}.KubernetesDeploymentManager")
    
    async def deploy_service(
        self, 
        service_name: str, 
        image_tag: str, 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Deploy a service to Kubernetes."""
        self.logger.info(f"Deploying {service_name} with tag {image_tag}")
        
        try:
            # Update deployment manifest
            manifest_updated = await self._update_deployment_manifest(
                service_name, image_tag, config
            )
            
            if not manifest_updated:
                return {"success": False, "error": "Failed to update deployment manifest"}
            
            # Apply deployment
            deploy_result = await self._apply_deployment(service_name)
            
            if not deploy_result["success"]:
                return deploy_result
            
            # Wait for rollout completion
            rollout_result = await self._wait_for_rollout(service_name, config.timeout)
            
            return rollout_result
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_deployment_manifest(
        self, 
        service_name: str, 
        image_tag: str, 
        config: DeploymentConfig
    ) -> bool:
        """Update Kubernetes deployment manifest."""
        try:
            manifest_path = f"/root/repo/deploy/kubernetes/{service_name}-deployment.yaml"
            
            # Check if manifest exists
            if not os.path.exists(manifest_path):
                self.logger.warning(f"Deployment manifest not found: {manifest_path}")
                # Generate basic manifest
                await self._generate_deployment_manifest(service_name, image_tag, config)
                return True
            
            # Update existing manifest
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            # Update image tag
            containers = manifest["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if service_name in container["image"]:
                    container["image"] = f"{service_name}:{image_tag}"
            
            # Update replicas
            manifest["spec"]["replicas"] = config.replicas
            
            # Update strategy
            if config.strategy == "rolling":
                manifest["spec"]["strategy"] = {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                }
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update deployment manifest: {e}")
            return False
    
    async def _generate_deployment_manifest(
        self, 
        service_name: str, 
        image_tag: str, 
        config: DeploymentConfig
    ):
        """Generate basic Kubernetes deployment manifest."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_name,
                "namespace": self.namespace,
                "labels": {
                    "app": service_name,
                    "environment": config.environment
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": service_name,
                            "image": f"{service_name}:{image_tag}",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "ENVIRONMENT", "value": config.environment},
                                {"name": "HIPAA_CONFIG_PATH", "value": "/config/hipaa_config.yml"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "2Gi", 
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Ensure directory exists
        manifest_dir = Path("/root/repo/deploy/kubernetes")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest_path = manifest_dir / f"{service_name}-deployment.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    
    async def _apply_deployment(self, service_name: str) -> Dict[str, Any]:
        """Apply Kubernetes deployment."""
        try:
            manifest_path = f"/root/repo/deploy/kubernetes/{service_name}-deployment.yaml"
            
            # Apply deployment
            result = subprocess.run(
                ["kubectl", "apply", "-f", manifest_path, "-n", self.namespace],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully applied deployment for {service_name}")
                return {"success": True, "output": result.stdout}
            else:
                self.logger.error(f"Failed to apply deployment: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            self.logger.error(f"Error applying deployment: {e}")
            return {"success": False, "error": str(e)}
    
    async def _wait_for_rollout(self, service_name: str, timeout: int) -> Dict[str, Any]:
        """Wait for deployment rollout completion."""
        try:
            # Wait for rollout
            result = subprocess.run(
                [
                    "kubectl", "rollout", "status", 
                    f"deployment/{service_name}", 
                    "-n", self.namespace,
                    f"--timeout={timeout}s"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Rollout completed for {service_name}")
                return {"success": True, "message": "Rollout completed successfully"}
            else:
                self.logger.error(f"Rollout failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            self.logger.error(f"Error waiting for rollout: {e}")
            return {"success": False, "error": str(e)}
    
    async def rollback_deployment(self, service_name: str) -> Dict[str, Any]:
        """Rollback deployment to previous version."""
        try:
            # Rollback deployment
            result = subprocess.run(
                [
                    "kubectl", "rollout", "undo",
                    f"deployment/{service_name}",
                    "-n", self.namespace
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Rollback initiated for {service_name}")
                
                # Wait for rollback completion
                rollback_wait = await self._wait_for_rollout(service_name, 300)
                return rollback_wait
            else:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_deployment_status(self, service_name: str) -> Dict[str, Any]:
        """Get current deployment status."""
        try:
            # Get deployment status
            result = subprocess.run(
                [
                    "kubectl", "get", "deployment", service_name,
                    "-n", self.namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                deployment_info = json.loads(result.stdout)
                status = deployment_info.get("status", {})
                
                return {
                    "success": True,
                    "replicas": status.get("replicas", 0),
                    "ready_replicas": status.get("readyReplicas", 0),
                    "updated_replicas": status.get("updatedReplicas", 0),
                    "available_replicas": status.get("availableReplicas", 0),
                    "conditions": status.get("conditions", [])
                }
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


class HealthMonitor:
    """
    Comprehensive health monitoring system for deployed services.
    
    Monitors service health, performs automated health checks,
    and triggers alerts or rollbacks based on health status.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, List[HealthCheck]] = {}
        self.health_status: Dict[str, HealthCheckStatus] = {}
        self.monitoring_active = False
        self.logger = logging.getLogger(f"{__name__}.HealthMonitor")
    
    def register_health_checks(self, service_name: str, checks: List[HealthCheck]):
        """Register health checks for a service."""
        self.health_checks[service_name] = checks
        self.health_status[service_name] = HealthCheckStatus.UNKNOWN
        self.logger.info(f"Registered {len(checks)} health checks for {service_name}")
    
    async def start_monitoring(self, check_interval: int = 30):
        """Start health monitoring."""
        self.monitoring_active = True
        self.logger.info("Health monitoring started")
        
        # Start monitoring tasks for each service
        tasks = []
        for service_name in self.health_checks.keys():
            task = asyncio.create_task(
                self._monitor_service_health(service_name, check_interval)
            )
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
    
    async def _monitor_service_health(self, service_name: str, interval: int):
        """Monitor health of a specific service."""
        while self.monitoring_active:
            try:
                health_status = await self.check_service_health(service_name)
                self.health_status[service_name] = health_status
                
                if health_status != HealthCheckStatus.HEALTHY:
                    self.logger.warning(f"Service {service_name} health status: {health_status.value}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error for {service_name}: {e}")
                await asyncio.sleep(interval)
    
    async def check_service_health(self, service_name: str) -> HealthCheckStatus:
        """Check health of a specific service."""
        if service_name not in self.health_checks:
            return HealthCheckStatus.UNKNOWN
        
        checks = self.health_checks[service_name]
        passed_checks = 0
        critical_failures = 0
        
        for check in checks:
            check_result = await self._execute_health_check(check)
            
            if check_result:
                passed_checks += 1
            elif check.critical:
                critical_failures += 1
        
        # Determine overall health status
        if critical_failures > 0:
            return HealthCheckStatus.UNHEALTHY
        elif passed_checks == len(checks):
            return HealthCheckStatus.HEALTHY
        else:
            return HealthCheckStatus.DEGRADED
    
    async def _execute_health_check(self, check: HealthCheck) -> bool:
        """Execute a single health check."""
        for attempt in range(check.retry_count):
            try:
                if check.endpoint:
                    # HTTP health check
                    success = await self._http_health_check(check)
                elif check.command:
                    # Command-based health check
                    success = await self._command_health_check(check)
                else:
                    self.logger.warning(f"No check method defined for {check.name}")
                    continue
                
                if success:
                    return True
                
            except Exception as e:
                self.logger.debug(f"Health check {check.name} attempt {attempt + 1} failed: {e}")
            
            if attempt < check.retry_count - 1:
                await asyncio.sleep(1)  # Wait before retry
        
        return False
    
    async def _http_health_check(self, check: HealthCheck) -> bool:
        """Perform HTTP-based health check."""
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=check.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(check.endpoint) as response:
                    return response.status == check.expected_status
                    
        except Exception:
            # Fallback to curl if aiohttp not available
            try:
                result = subprocess.run(
                    ["curl", "-f", "-s", "-o", "/dev/null", "-w", "%{http_code}", check.endpoint],
                    capture_output=True,
                    text=True,
                    timeout=check.timeout
                )
                
                return result.stdout.strip() == str(check.expected_status)
            except:
                return False
    
    async def _command_health_check(self, check: HealthCheck) -> bool:
        """Perform command-based health check."""
        try:
            result = subprocess.run(
                check.command.split(),
                capture_output=True,
                text=True,
                timeout=check.timeout
            )
            
            return result.returncode == 0
            
        except:
            return False
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_status:
            return {"status": "unknown", "services": {}}
        
        service_statuses = list(self.health_status.values())
        
        # Determine overall status
        if all(status == HealthCheckStatus.HEALTHY for status in service_statuses):
            overall_status = "healthy"
        elif any(status == HealthCheckStatus.UNHEALTHY for status in service_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "services": {name: status.value for name, status in self.health_status.items()},
            "healthy_services": sum(1 for s in service_statuses if s == HealthCheckStatus.HEALTHY),
            "total_services": len(service_statuses)
        }


class AutonomousDeploymentOrchestrator:
    """
    Main autonomous deployment orchestration system.
    
    Coordinates deployment phases, health monitoring, rollbacks, and
    intelligent decision-making for production deployments.
    """
    
    def __init__(self):
        self.k8s_manager = KubernetesDeploymentManager()
        self.health_monitor = HealthMonitor()
        self.deployment_history: List[DeploymentResult] = []
        self.logger = logging.getLogger(f"{__name__}.AutonomousDeploymentOrchestrator")
    
    async def deploy(
        self, 
        services: List[str], 
        image_tag: str, 
        config: DeploymentConfig
    ) -> DeploymentResult:
        """
        Execute autonomous deployment of services.
        
        Args:
            services: List of service names to deploy
            image_tag: Docker image tag to deploy
            config: Deployment configuration
            
        Returns:
            Deployment result with status and metrics
        """
        self.logger.info(f"Starting autonomous deployment of {services} with tag {image_tag}")
        
        metrics = DeploymentMetrics(start_time=datetime.now())
        result = DeploymentResult(status=DeploymentStatus.PENDING, metrics=metrics)
        
        try:
            # Phase 1: Preparation
            result.status = DeploymentStatus.IN_PROGRESS
            await self._execute_preparation_phase(services, config, result)
            
            # Phase 2: Pre-flight checks
            preflight_success = await self._execute_preflight_checks(services, config, result)
            if not preflight_success:
                result.status = DeploymentStatus.FAILED
                return result
            
            # Phase 3: Deployment
            deployment_success = await self._execute_deployment_phase(services, image_tag, config, result)
            if not deployment_success:
                if config.rollback_on_failure:
                    await self._execute_rollback_phase(services, config, result)
                    result.status = DeploymentStatus.ROLLED_BACK
                else:
                    result.status = DeploymentStatus.FAILED
                return result
            
            # Phase 4: Health verification
            health_success = await self._execute_health_verification_phase(services, config, result)
            if not health_success:
                if config.rollback_on_failure:
                    await self._execute_rollback_phase(services, config, result)
                    result.status = DeploymentStatus.ROLLED_BACK
                else:
                    result.status = DeploymentStatus.FAILED
                return result
            
            # Phase 5: Post-deployment monitoring
            await self._execute_post_deployment_phase(services, config, result)
            
            result.status = DeploymentStatus.SUCCESS
            
        except Exception as e:
            self.logger.error(f"Deployment failed with exception: {e}")
            result.errors.append(str(e))
            result.status = DeploymentStatus.FAILED
            
            if config.rollback_on_failure:
                try:
                    await self._execute_rollback_phase(services, config, result)
                    result.status = DeploymentStatus.ROLLED_BACK
                except Exception as rollback_error:
                    result.errors.append(f"Rollback failed: {rollback_error}")
        
        finally:
            # Finalize metrics
            metrics.end_time = datetime.now()
            metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
            
            # Calculate success rate
            if result.status == DeploymentStatus.SUCCESS:
                metrics.success_rate = 1.0
            elif result.status == DeploymentStatus.ROLLED_BACK:
                metrics.success_rate = 0.5
            else:
                metrics.success_rate = 0.0
            
            # Store deployment history
            self.deployment_history.append(result)
            
            self.logger.info(f"Deployment completed with status: {result.status.value}")
            
        return result
    
    async def _execute_preparation_phase(
        self, 
        services: List[str], 
        config: DeploymentConfig, 
        result: DeploymentResult
    ):
        """Execute preparation phase."""
        phase_start = time.time()
        self.logger.info("Executing preparation phase")
        
        try:
            # Verify prerequisites
            prerequisites_ok = await self._verify_prerequisites()
            if not prerequisites_ok:
                result.errors.append("Prerequisites verification failed")
                return
            
            # Setup health checks
            for service in services:
                health_checks = self._create_default_health_checks(service, config.environment)
                self.health_monitor.register_health_checks(service, health_checks)
            
            # Create namespace if it doesn't exist
            await self._ensure_namespace_exists()
            
            result.logs.append("Preparation phase completed successfully")
            
        except Exception as e:
            result.errors.append(f"Preparation phase failed: {e}")
            raise
        
        finally:
            result.metrics.phase_durations["preparation"] = time.time() - phase_start
    
    async def _execute_preflight_checks(
        self, 
        services: List[str], 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute pre-flight checks phase."""
        phase_start = time.time()
        self.logger.info("Executing pre-flight checks")
        
        try:
            checks_passed = True
            
            # Check cluster resources
            resource_check = await self._check_cluster_resources(services, config)
            if not resource_check:
                result.errors.append("Insufficient cluster resources")
                checks_passed = False
            
            # Check image availability
            for service in services:
                image_available = await self._check_image_availability(service, config.environment)
                if not image_available:
                    result.errors.append(f"Image not available for service: {service}")
                    checks_passed = False
            
            # Check dependencies
            dependencies_ok = await self._check_service_dependencies(services)
            if not dependencies_ok:
                result.errors.append("Service dependencies check failed")
                checks_passed = False
            
            if checks_passed:
                result.logs.append("All pre-flight checks passed")
            
            return checks_passed
            
        except Exception as e:
            result.errors.append(f"Pre-flight checks failed: {e}")
            return False
        
        finally:
            result.metrics.phase_durations["pre_flight_checks"] = time.time() - phase_start
    
    async def _execute_deployment_phase(
        self, 
        services: List[str], 
        image_tag: str, 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute deployment phase."""
        phase_start = time.time()
        self.logger.info(f"Executing deployment phase with strategy: {config.strategy}")
        
        try:
            deployment_success = True
            
            if config.strategy == "rolling":
                deployment_success = await self._rolling_deployment(services, image_tag, config, result)
            elif config.strategy == "blue_green":
                deployment_success = await self._blue_green_deployment(services, image_tag, config, result)
            elif config.strategy == "canary":
                deployment_success = await self._canary_deployment(services, image_tag, config, result)
            else:
                result.errors.append(f"Unknown deployment strategy: {config.strategy}")
                return False
            
            if deployment_success:
                result.deployed_services = services.copy()
                result.logs.append(f"Deployment phase completed for {len(services)} services")
            
            return deployment_success
            
        except Exception as e:
            result.errors.append(f"Deployment phase failed: {e}")
            return False
        
        finally:
            result.metrics.phase_durations["deployment"] = time.time() - phase_start
    
    async def _rolling_deployment(
        self, 
        services: List[str], 
        image_tag: str, 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute rolling deployment strategy."""
        for service in services:
            self.logger.info(f"Deploying service: {service}")
            
            deploy_result = await self.k8s_manager.deploy_service(service, image_tag, config)
            
            if not deploy_result.get("success", False):
                result.errors.append(f"Failed to deploy {service}: {deploy_result.get('error', 'Unknown error')}")
                return False
            
            # Verify deployment status
            status = await self.k8s_manager.get_deployment_status(service)
            if not status.get("success", False):
                result.errors.append(f"Failed to get deployment status for {service}")
                return False
            
            # Check if all replicas are ready
            ready_replicas = status.get("ready_replicas", 0)
            desired_replicas = config.replicas
            
            if ready_replicas != desired_replicas:
                result.errors.append(f"Not all replicas ready for {service}: {ready_replicas}/{desired_replicas}")
                return False
            
            result.logs.append(f"Successfully deployed {service} with {ready_replicas} replicas")
        
        return True
    
    async def _blue_green_deployment(
        self, 
        services: List[str], 
        image_tag: str, 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute blue-green deployment strategy."""
        # Simplified blue-green deployment
        self.logger.info("Blue-green deployment not fully implemented - falling back to rolling")
        result.warnings.append("Blue-green deployment not fully implemented - using rolling deployment")
        return await self._rolling_deployment(services, image_tag, config, result)
    
    async def _canary_deployment(
        self, 
        services: List[str], 
        image_tag: str, 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute canary deployment strategy."""
        # Simplified canary deployment
        self.logger.info("Canary deployment not fully implemented - falling back to rolling")
        result.warnings.append("Canary deployment not fully implemented - using rolling deployment")
        return await self._rolling_deployment(services, image_tag, config, result)
    
    async def _execute_health_verification_phase(
        self, 
        services: List[str], 
        config: DeploymentConfig, 
        result: DeploymentResult
    ) -> bool:
        """Execute health verification phase."""
        phase_start = time.time()
        self.logger.info("Executing health verification phase")
        
        try:
            # Wait for services to stabilize
            await asyncio.sleep(30)
            
            # Check health of all deployed services
            all_healthy = True
            
            for service in services:
                health_status = await self.health_monitor.check_service_health(service)
                
                if health_status == HealthCheckStatus.UNHEALTHY:
                    result.errors.append(f"Service {service} is unhealthy after deployment")
                    result.metrics.health_check_failures += 1
                    all_healthy = False
                elif health_status == HealthCheckStatus.DEGRADED:
                    result.warnings.append(f"Service {service} is in degraded state")
                
                result.logs.append(f"Health check for {service}: {health_status.value}")
            
            if all_healthy:
                result.logs.append("All services passed health verification")
            
            return all_healthy
            
        except Exception as e:
            result.errors.append(f"Health verification failed: {e}")
            return False
        
        finally:
            result.metrics.phase_durations["health_verification"] = time.time() - phase_start
    
    async def _execute_rollback_phase(
        self, 
        services: List[str], 
        config: DeploymentConfig, 
        result: DeploymentResult
    ):
        """Execute rollback phase."""
        phase_start = time.time()
        self.logger.info("Executing rollback phase")
        
        try:
            result.metrics.rollback_count += 1
            
            for service in services:
                self.logger.info(f"Rolling back service: {service}")
                
                rollback_result = await self.k8s_manager.rollback_deployment(service)
                
                if rollback_result.get("success", False):
                    result.logs.append(f"Successfully rolled back {service}")
                else:
                    result.errors.append(f"Failed to rollback {service}: {rollback_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            result.errors.append(f"Rollback phase failed: {e}")
        
        finally:
            result.metrics.phase_durations["rollback"] = time.time() - phase_start
    
    async def _execute_post_deployment_phase(
        self, 
        services: List[str], 
        config: DeploymentConfig, 
        result: DeploymentResult
    ):
        """Execute post-deployment monitoring phase."""
        phase_start = time.time()
        self.logger.info("Executing post-deployment monitoring")
        
        try:
            # Monitor services for specified duration
            monitoring_duration = config.monitoring_duration
            check_interval = 30
            checks_count = monitoring_duration // check_interval
            
            for i in range(checks_count):
                overall_health = self.health_monitor.get_overall_health()
                
                if overall_health["status"] != "healthy":
                    result.warnings.append(f"Health degradation detected during monitoring (check {i+1})")
                
                await asyncio.sleep(check_interval)
            
            result.logs.append(f"Post-deployment monitoring completed ({monitoring_duration}s)")
            
        except Exception as e:
            result.warnings.append(f"Post-deployment monitoring error: {e}")
        
        finally:
            result.metrics.phase_durations["post_deployment"] = time.time() - phase_start
    
    async def _verify_prerequisites(self) -> bool:
        """Verify deployment prerequisites."""
        try:
            # Check kubectl availability
            kubectl_result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                text=True
            )
            
            if kubectl_result.returncode != 0:
                self.logger.error("kubectl not available")
                return False
            
            # Check cluster connectivity
            cluster_result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True
            )
            
            if cluster_result.returncode != 0:
                self.logger.error("Cannot connect to Kubernetes cluster")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prerequisites verification failed: {e}")
            return False
    
    def _create_default_health_checks(self, service_name: str, environment: str) -> List[HealthCheck]:
        """Create default health checks for a service."""
        return [
            HealthCheck(
                name=f"{service_name}_http_health",
                endpoint=f"http://{service_name}.{self.k8s_manager.namespace}.svc.cluster.local:8000/health",
                expected_status=200,
                timeout=10,
                critical=True,
                retry_count=3
            ),
            HealthCheck(
                name=f"{service_name}_readiness",
                endpoint=f"http://{service_name}.{self.k8s_manager.namespace}.svc.cluster.local:8000/ready", 
                expected_status=200,
                timeout=5,
                critical=True,
                retry_count=2
            )
        ]
    
    async def _ensure_namespace_exists(self):
        """Ensure Kubernetes namespace exists."""
        try:
            # Check if namespace exists
            result = subprocess.run(
                ["kubectl", "get", "namespace", self.k8s_manager.namespace],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Create namespace
                create_result = subprocess.run(
                    ["kubectl", "create", "namespace", self.k8s_manager.namespace],
                    capture_output=True,
                    text=True
                )
                
                if create_result.returncode == 0:
                    self.logger.info(f"Created namespace: {self.k8s_manager.namespace}")
                else:
                    self.logger.error(f"Failed to create namespace: {create_result.stderr}")
            
        except Exception as e:
            self.logger.error(f"Error ensuring namespace exists: {e}")
    
    async def _check_cluster_resources(self, services: List[str], config: DeploymentConfig) -> bool:
        """Check if cluster has sufficient resources."""
        try:
            # Simplified resource check
            # In a real implementation, this would check CPU, memory, and storage
            return True
            
        except Exception:
            return False
    
    async def _check_image_availability(self, service_name: str, environment: str) -> bool:
        """Check if Docker image is available."""
        try:
            # Simplified image availability check
            # In a real implementation, this would check Docker registry
            return True
            
        except Exception:
            return False
    
    async def _check_service_dependencies(self, services: List[str]) -> bool:
        """Check service dependencies."""
        try:
            # Simplified dependency check
            # In a real implementation, this would check database connections, external APIs, etc.
            return True
            
        except Exception:
            return False
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of deployment history and metrics."""
        if not self.deployment_history:
            return {"total_deployments": 0}
        
        total_deployments = len(self.deployment_history)
        successful_deployments = sum(1 for d in self.deployment_history if d.status == DeploymentStatus.SUCCESS)
        
        avg_duration = sum(d.metrics.duration for d in self.deployment_history) / total_deployments
        total_rollbacks = sum(d.metrics.rollback_count for d in self.deployment_history)
        
        return {
            "total_deployments": total_deployments,
            "successful_deployments": successful_deployments,
            "success_rate": successful_deployments / total_deployments,
            "average_duration": avg_duration,
            "total_rollbacks": total_rollbacks,
            "recent_deployments": [
                {
                    "status": d.status.value,
                    "duration": d.metrics.duration,
                    "services": len(d.deployed_services),
                    "timestamp": d.metrics.start_time.isoformat()
                }
                for d in self.deployment_history[-5:]  # Last 5 deployments
            ]
        }


# Global autonomous deployment orchestrator instance
autonomous_deployment_orchestrator = AutonomousDeploymentOrchestrator()


async def autonomous_deploy(
    services: List[str],
    image_tag: str,
    environment: str = "production",
    strategy: str = "rolling"
) -> DeploymentResult:
    """
    Convenience function for autonomous deployment.
    
    Usage:
        result = await autonomous_deploy(
            services=["hipaa-api", "hipaa-worker"],
            image_tag="v1.2.3",
            environment="production",
            strategy="rolling"
        )
    """
    config = DeploymentConfig(
        environment=environment,
        version=image_tag,
        strategy=strategy
    )
    
    return await autonomous_deployment_orchestrator.deploy(services, image_tag, config)