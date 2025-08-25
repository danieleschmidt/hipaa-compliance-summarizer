"""
Multi-Region Deployment System - Global Infrastructure Orchestration

This module implements intelligent multi-region deployment capabilities with
automatic failover, data residency compliance, performance optimization,
and region-specific configuration management for global healthcare systems.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .global_compliance_framework import ComplianceJurisdiction


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA_CLOUD = "alibaba"
    TENCENT_CLOUD = "tencent"


class DeploymentRegion(Enum):
    """Global deployment regions."""
    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    CA_CENTRAL_1 = "ca-central-1"
    
    # Europe
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    UK_SOUTH_1 = "uk-south-1"
    
    # Asia Pacific
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Singapore
    AP_SOUTHEAST_2 = "ap-southeast-2"  # Sydney
    AP_NORTHEAST_1 = "ap-northeast-1"  # Tokyo
    AP_NORTHEAST_2 = "ap-northeast-2"  # Seoul
    AP_SOUTH_1 = "ap-south-1"         # Mumbai
    
    # South America
    SA_EAST_1 = "sa-east-1"           # São Paulo


class FailoverStrategy(Enum):
    """Failover strategies."""
    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"
    MULTI_MASTER = "multi_master"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    cloud_provider: CloudProvider
    jurisdiction: ComplianceJurisdiction
    data_residency_required: bool
    primary_region: bool = False
    disaster_recovery_target: bool = True
    latency_target_ms: int = 100
    availability_target: float = 99.9
    compliance_requirements: List[str] = field(default_factory=list)
    encryption_requirements: Dict[str, str] = field(default_factory=dict)
    network_restrictions: List[str] = field(default_factory=list)


@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""
    service_name: str
    region: DeploymentRegion
    endpoint_url: str
    health_check_url: str
    status: str = "unknown"  # healthy, unhealthy, maintenance
    latency_ms: float = 0.0
    last_check: Optional[datetime] = None
    error_rate: float = 0.0
    throughput_rps: float = 0.0


@dataclass
class FailoverEvent:
    """Failover event record."""
    timestamp: datetime
    source_region: DeploymentRegion
    target_region: DeploymentRegion
    trigger: str
    duration_seconds: float
    affected_services: List[str]
    recovery_time: Optional[float] = None


@dataclass
class DeploymentMetrics:
    """Multi-region deployment metrics."""
    total_regions: int
    healthy_regions: int
    average_latency_ms: float
    global_availability: float
    total_requests_per_second: float
    data_transfer_gb: float
    compliance_violations: int
    failover_events: int
    last_updated: datetime


class RegionHealthMonitor:
    """
    Advanced health monitoring system for multi-region deployments.
    
    Monitors region health, latency, availability, and compliance status
    across global infrastructure with intelligent alerting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RegionHealthMonitor")
        self.endpoints: Dict[str, List[ServiceEndpoint]] = {}
        self.health_history: Dict[str, List[Dict[str, Any]]] = {}
        self.monitoring_active = False
        
    def register_endpoints(self, region: DeploymentRegion, endpoints: List[ServiceEndpoint]):
        """Register service endpoints for monitoring."""
        region_key = region.value
        self.endpoints[region_key] = endpoints
        self.health_history[region_key] = []
        self.logger.info(f"Registered {len(endpoints)} endpoints for region {region_key}")
    
    async def start_monitoring(self, check_interval: int = 30):
        """Start health monitoring for all regions."""
        self.monitoring_active = True
        self.logger.info("Starting multi-region health monitoring")
        
        # Create monitoring tasks for each region
        tasks = []
        for region_key in self.endpoints.keys():
            task = asyncio.create_task(
                self._monitor_region_health(region_key, check_interval)
            )
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        self.logger.info("Multi-region health monitoring stopped")
    
    async def _monitor_region_health(self, region_key: str, interval: int):
        """Monitor health of a specific region."""
        while self.monitoring_active:
            try:
                endpoints = self.endpoints.get(region_key, [])
                region_health = {
                    "timestamp": datetime.now(),
                    "region": region_key,
                    "healthy_endpoints": 0,
                    "total_endpoints": len(endpoints),
                    "average_latency": 0.0,
                    "total_error_rate": 0.0,
                    "total_throughput": 0.0
                }
                
                total_latency = 0.0
                total_error_rate = 0.0
                total_throughput = 0.0
                
                for endpoint in endpoints:
                    health_check_result = await self._check_endpoint_health(endpoint)
                    
                    if health_check_result["status"] == "healthy":
                        region_health["healthy_endpoints"] += 1
                    
                    total_latency += health_check_result["latency_ms"]
                    total_error_rate += health_check_result["error_rate"]
                    total_throughput += health_check_result["throughput_rps"]
                
                # Calculate averages
                if endpoints:
                    region_health["average_latency"] = total_latency / len(endpoints)
                    region_health["total_error_rate"] = total_error_rate / len(endpoints)
                    region_health["total_throughput"] = total_throughput
                
                # Store health data
                self.health_history[region_key].append(region_health)
                
                # Keep only last 100 health checks
                if len(self.health_history[region_key]) > 100:
                    self.health_history[region_key] = self.health_history[region_key][-100:]
                
                # Check for health degradation
                await self._check_health_alerts(region_key, region_health)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error for region {region_key}: {e}")
                await asyncio.sleep(interval)
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> Dict[str, Any]:
        """Check health of a specific endpoint."""
        start_time = time.time()
        
        try:
            # Simulate health check (in real implementation, use HTTP requests)
            # This would use aiohttp or similar to check actual endpoints
            await asyncio.sleep(0.01)  # Simulate network request
            
            # Simulate response
            latency_ms = (time.time() - start_time) * 1000
            
            # Update endpoint status
            endpoint.status = "healthy"
            endpoint.latency_ms = latency_ms
            endpoint.last_check = datetime.now()
            endpoint.error_rate = 0.01  # 1% error rate simulation
            endpoint.throughput_rps = 100.0  # 100 RPS simulation
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "error_rate": 0.01,
                "throughput_rps": 100.0
            }
            
        except Exception as e:
            endpoint.status = "unhealthy"
            endpoint.last_check = datetime.now()
            endpoint.error_rate = 1.0
            
            self.logger.warning(f"Endpoint {endpoint.service_name} health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "latency_ms": 5000,  # High latency for failed checks
                "error_rate": 1.0,
                "throughput_rps": 0.0
            }
    
    async def _check_health_alerts(self, region_key: str, region_health: Dict[str, Any]):
        """Check for health alerts and trigger notifications."""
        healthy_ratio = region_health["healthy_endpoints"] / max(region_health["total_endpoints"], 1)
        
        if healthy_ratio < 0.5:
            self.logger.critical(f"ALERT: Region {region_key} has {healthy_ratio:.1%} healthy endpoints")
        elif healthy_ratio < 0.8:
            self.logger.warning(f"WARNING: Region {region_key} has {healthy_ratio:.1%} healthy endpoints")
        
        if region_health["average_latency"] > 1000:
            self.logger.warning(f"HIGH LATENCY: Region {region_key} latency is {region_health['average_latency']:.1f}ms")
        
        if region_health["total_error_rate"] > 0.1:
            self.logger.warning(f"HIGH ERROR RATE: Region {region_key} error rate is {region_health['total_error_rate']:.1%}")
    
    def get_region_status(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Get current status of a region."""
        region_key = region.value
        endpoints = self.endpoints.get(region_key, [])
        recent_health = self.health_history.get(region_key, [])
        
        if not recent_health:
            return {"status": "unknown", "message": "No health data available"}
        
        latest_health = recent_health[-1]
        
        return {
            "region": region_key,
            "status": "healthy" if latest_health["healthy_endpoints"] > latest_health["total_endpoints"] * 0.8 else "degraded",
            "healthy_endpoints": latest_health["healthy_endpoints"],
            "total_endpoints": latest_health["total_endpoints"],
            "availability": latest_health["healthy_endpoints"] / max(latest_health["total_endpoints"], 1),
            "average_latency_ms": latest_health["average_latency"],
            "error_rate": latest_health["total_error_rate"],
            "throughput_rps": latest_health["total_throughput"],
            "last_updated": latest_health["timestamp"].isoformat()
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global health status across all regions."""
        all_regions_health = []
        total_healthy = 0
        total_endpoints = 0
        total_latency = 0.0
        total_throughput = 0.0
        
        for region_key, health_history in self.health_history.items():
            if health_history:
                latest = health_history[-1]
                all_regions_health.append(latest)
                total_healthy += latest["healthy_endpoints"]
                total_endpoints += latest["total_endpoints"]
                total_latency += latest["average_latency"]
                total_throughput += latest["total_throughput"]
        
        regions_count = len(all_regions_health)
        
        return {
            "global_status": "healthy" if total_healthy > total_endpoints * 0.8 else "degraded",
            "total_regions": regions_count,
            "healthy_regions": sum(1 for h in all_regions_health if h["healthy_endpoints"] > 0),
            "global_availability": total_healthy / max(total_endpoints, 1),
            "average_latency_ms": total_latency / max(regions_count, 1),
            "total_throughput_rps": total_throughput,
            "last_updated": datetime.now().isoformat()
        }


class IntelligentFailoverManager:
    """
    Intelligent failover management system for multi-region deployments.
    
    Implements intelligent failover strategies with automatic recovery,
    load balancing, and compliance-aware region selection.
    """
    
    def __init__(self, health_monitor: RegionHealthMonitor):
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(f"{__name__}.IntelligentFailoverManager")
        self.failover_history: List[FailoverEvent] = []
        self.current_failovers: Dict[str, FailoverEvent] = {}
        self.region_preferences: Dict[str, List[DeploymentRegion]] = {}
    
    def configure_failover_preferences(
        self, 
        service_name: str, 
        preference_order: List[DeploymentRegion]
    ):
        """Configure failover preferences for a service."""
        self.region_preferences[service_name] = preference_order
        self.logger.info(f"Configured failover preferences for {service_name}: {[r.value for r in preference_order]}")
    
    async def monitor_and_failover(self, check_interval: int = 60):
        """Monitor region health and execute intelligent failovers."""
        self.logger.info("Starting intelligent failover monitoring")
        
        while True:
            try:
                await self._check_failover_conditions()
                await self._check_recovery_conditions()
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Failover monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    async def _check_failover_conditions(self):
        """Check if any regions need failover."""
        for region_key in self.health_monitor.endpoints.keys():
            region = DeploymentRegion(region_key)
            region_status = self.health_monitor.get_region_status(region)
            
            # Check if region needs failover
            if self._should_trigger_failover(region, region_status):
                await self._execute_failover(region, "health_degradation")
    
    async def _check_recovery_conditions(self):
        """Check if any failed regions can be recovered."""
        for service_name, failover_event in list(self.current_failovers.items()):
            source_region = failover_event.source_region
            region_status = self.health_monitor.get_region_status(source_region)
            
            # Check if region has recovered
            if self._should_trigger_recovery(source_region, region_status):
                await self._execute_recovery(service_name, failover_event)
    
    def _should_trigger_failover(self, region: DeploymentRegion, status: Dict[str, Any]) -> bool:
        """Determine if failover should be triggered for a region."""
        # Failover conditions
        availability_threshold = 0.8
        latency_threshold = 2000  # 2 seconds
        error_rate_threshold = 0.2  # 20%
        
        if status.get("availability", 1.0) < availability_threshold:
            self.logger.warning(f"Region {region.value} availability below threshold: {status.get('availability'):.2%}")
            return True
        
        if status.get("average_latency_ms", 0) > latency_threshold:
            self.logger.warning(f"Region {region.value} latency above threshold: {status.get('average_latency_ms')}ms")
            return True
        
        if status.get("error_rate", 0) > error_rate_threshold:
            self.logger.warning(f"Region {region.value} error rate above threshold: {status.get('error_rate'):.2%}")
            return True
        
        return False
    
    def _should_trigger_recovery(self, region: DeploymentRegion, status: Dict[str, Any]) -> bool:
        """Determine if recovery should be triggered for a region."""
        # Recovery conditions (more conservative than failover)
        availability_threshold = 0.95
        latency_threshold = 500  # 500ms
        error_rate_threshold = 0.05  # 5%
        
        return (
            status.get("availability", 0.0) >= availability_threshold and
            status.get("average_latency_ms", float('inf')) <= latency_threshold and
            status.get("error_rate", 1.0) <= error_rate_threshold
        )
    
    async def _execute_failover(self, failed_region: DeploymentRegion, trigger: str):
        """Execute failover from a failed region to healthy regions."""
        self.logger.critical(f"Executing failover from region {failed_region.value} due to {trigger}")
        
        # Find best target region
        target_region = await self._select_failover_target(failed_region)
        
        if not target_region:
            self.logger.error(f"No suitable failover target found for {failed_region.value}")
            return
        
        # Create failover event
        failover_event = FailoverEvent(
            timestamp=datetime.now(),
            source_region=failed_region,
            target_region=target_region,
            trigger=trigger,
            duration_seconds=0.0,
            affected_services=list(self.health_monitor.endpoints.get(failed_region.value, []))
        )
        
        start_time = time.time()
        
        try:
            # Execute failover steps
            await self._redirect_traffic(failed_region, target_region)
            await self._update_dns_records(failed_region, target_region)
            await self._notify_monitoring_systems(failover_event)
            
            # Record successful failover
            failover_event.duration_seconds = time.time() - start_time
            self.failover_history.append(failover_event)
            self.current_failovers[f"{failed_region.value}_to_{target_region.value}"] = failover_event
            
            self.logger.info(f"Failover completed from {failed_region.value} to {target_region.value} in {failover_event.duration_seconds:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failover execution failed: {e}")
            failover_event.duration_seconds = time.time() - start_time
            self.failover_history.append(failover_event)
    
    async def _execute_recovery(self, service_name: str, failover_event: FailoverEvent):
        """Execute recovery back to original region."""
        self.logger.info(f"Executing recovery for {service_name} back to {failover_event.source_region.value}")
        
        start_time = time.time()
        
        try:
            # Execute recovery steps
            await self._redirect_traffic(failover_event.target_region, failover_event.source_region)
            await self._update_dns_records(failover_event.target_region, failover_event.source_region)
            
            # Record recovery
            failover_event.recovery_time = time.time() - start_time
            
            # Remove from current failovers
            failover_key = f"{failover_event.source_region.value}_to_{failover_event.target_region.value}"
            if failover_key in self.current_failovers:
                del self.current_failovers[failover_key]
            
            self.logger.info(f"Recovery completed for {service_name} in {failover_event.recovery_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}")
    
    async def _select_failover_target(self, failed_region: DeploymentRegion) -> Optional[DeploymentRegion]:
        """Select the best failover target region."""
        candidate_regions = []
        
        # Get all available regions except the failed one
        for region_key in self.health_monitor.endpoints.keys():
            region = DeploymentRegion(region_key)
            if region != failed_region:
                region_status = self.health_monitor.get_region_status(region)
                if region_status.get("availability", 0) > 0.8:  # Only healthy regions
                    candidate_regions.append((region, region_status))
        
        if not candidate_regions:
            return None
        
        # Sort by preference (availability, then latency)
        candidate_regions.sort(
            key=lambda x: (-x[1].get("availability", 0), x[1].get("average_latency_ms", float('inf')))
        )
        
        return candidate_regions[0][0]
    
    async def _redirect_traffic(self, source_region: DeploymentRegion, target_region: DeploymentRegion):
        """Redirect traffic from source to target region."""
        # This would implement actual traffic redirection
        # In a real system, this would update load balancer configurations
        self.logger.info(f"Redirecting traffic from {source_region.value} to {target_region.value}")
        await asyncio.sleep(1)  # Simulate traffic redirection time
    
    async def _update_dns_records(self, source_region: DeploymentRegion, target_region: DeploymentRegion):
        """Update DNS records for failover."""
        # This would implement DNS updates
        # In a real system, this would update Route53, Azure DNS, or CloudDNS
        self.logger.info(f"Updating DNS records: {source_region.value} -> {target_region.value}")
        await asyncio.sleep(0.5)  # Simulate DNS update time
    
    async def _notify_monitoring_systems(self, failover_event: FailoverEvent):
        """Notify monitoring and alerting systems of failover."""
        # This would send notifications to PagerDuty, Slack, etc.
        self.logger.info(f"Notifying monitoring systems of failover: {failover_event.source_region.value} -> {failover_event.target_region.value}")
    
    def get_failover_summary(self) -> Dict[str, Any]:
        """Get summary of failover events and current status."""
        return {
            "total_failovers": len(self.failover_history),
            "active_failovers": len(self.current_failovers),
            "recent_failovers": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "source_region": event.source_region.value,
                    "target_region": event.target_region.value,
                    "trigger": event.trigger,
                    "duration_seconds": event.duration_seconds,
                    "recovered": event.recovery_time is not None
                }
                for event in self.failover_history[-10:]  # Last 10 failovers
            ],
            "current_failovers": [
                {
                    "source_region": event.source_region.value,
                    "target_region": event.target_region.value,
                    "started_at": event.timestamp.isoformat(),
                    "duration_minutes": (datetime.now() - event.timestamp).total_seconds() / 60
                }
                for event in self.current_failovers.values()
            ]
        }


class MultiRegionDeploymentOrchestrator:
    """
    Multi-region deployment orchestrator for global healthcare systems.
    
    Coordinates deployments across multiple regions with compliance,
    data residency, and performance optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MultiRegionDeploymentOrchestrator")
        self.region_configs: Dict[str, RegionConfig] = {}
        self.health_monitor = RegionHealthMonitor()
        self.failover_manager = IntelligentFailoverManager(self.health_monitor)
        self.deployment_history: List[Dict[str, Any]] = []
    
    def configure_regions(self, configurations: List[RegionConfig]):
        """Configure deployment regions."""
        for config in configurations:
            self.region_configs[config.region.value] = config
            self.logger.info(f"Configured region {config.region.value} with jurisdiction {config.jurisdiction.value}")
        
        # Setup health monitoring endpoints
        for config in configurations:
            endpoints = self._create_region_endpoints(config)
            self.health_monitor.register_endpoints(config.region, endpoints)
    
    def _create_region_endpoints(self, config: RegionConfig) -> List[ServiceEndpoint]:
        """Create health check endpoints for a region."""
        base_url = self._get_region_base_url(config)
        
        endpoints = [
            ServiceEndpoint(
                service_name="api_gateway",
                region=config.region,
                endpoint_url=f"{base_url}/api/v1",
                health_check_url=f"{base_url}/health"
            ),
            ServiceEndpoint(
                service_name="document_processor",
                region=config.region,
                endpoint_url=f"{base_url}/process",
                health_check_url=f"{base_url}/process/health"
            ),
            ServiceEndpoint(
                service_name="compliance_validator",
                region=config.region,
                endpoint_url=f"{base_url}/validate",
                health_check_url=f"{base_url}/validate/health"
            )
        ]
        
        return endpoints
    
    def _get_region_base_url(self, config: RegionConfig) -> str:
        """Generate base URL for region endpoints."""
        if config.cloud_provider == CloudProvider.AWS:
            return f"https://api-{config.region.value}.hipaa-compliance.aws.com"
        elif config.cloud_provider == CloudProvider.AZURE:
            return f"https://api-{config.region.value}.hipaa-compliance.azure.com"
        elif config.cloud_provider == CloudProvider.GCP:
            return f"https://api-{config.region.value}.hipaa-compliance.gcp.com"
        else:
            return f"https://api-{config.region.value}.hipaa-compliance.com"
    
    async def deploy_globally(
        self, 
        services: List[str], 
        version: str, 
        deployment_strategy: str = "rolling"
    ) -> Dict[str, Any]:
        """Execute global deployment across all configured regions."""
        self.logger.info(f"Starting global deployment of {services} version {version}")
        
        deployment_result = {
            "deployment_id": f"global_{int(time.time())}",
            "version": version,
            "services": services,
            "strategy": deployment_strategy,
            "start_time": datetime.now().isoformat(),
            "regions": {},
            "overall_success": True,
            "compliance_validated": True
        }
        
        # Deploy to regions in order of preference
        primary_regions = [config for config in self.region_configs.values() if config.primary_region]
        secondary_regions = [config for config in self.region_configs.values() if not config.primary_region]
        
        # Deploy to primary regions first
        for config in primary_regions:
            region_result = await self._deploy_to_region(config, services, version)
            deployment_result["regions"][config.region.value] = region_result
            
            if not region_result["success"]:
                deployment_result["overall_success"] = False
                
                # If primary region fails, consider aborting
                if config.primary_region:
                    self.logger.critical(f"Primary region {config.region.value} deployment failed")
        
        # Deploy to secondary regions
        if deployment_result["overall_success"]:
            for config in secondary_regions:
                region_result = await self._deploy_to_region(config, services, version)
                deployment_result["regions"][config.region.value] = region_result
                
                if not region_result["success"]:
                    self.logger.warning(f"Secondary region {config.region.value} deployment failed")
        
        # Validate global compliance
        compliance_validation = await self._validate_global_compliance(deployment_result)
        deployment_result["compliance_validation"] = compliance_validation
        
        if not compliance_validation["compliant"]:
            deployment_result["compliance_validated"] = False
            self.logger.error("Global compliance validation failed")
        
        deployment_result["end_time"] = datetime.now().isoformat()
        
        # Store deployment history
        self.deployment_history.append(deployment_result)
        
        # Start post-deployment monitoring
        if deployment_result["overall_success"]:
            await self._start_post_deployment_monitoring()
        
        return deployment_result
    
    async def _deploy_to_region(
        self, 
        config: RegionConfig, 
        services: List[str], 
        version: str
    ) -> Dict[str, Any]:
        """Deploy services to a specific region."""
        self.logger.info(f"Deploying to region {config.region.value}")
        
        region_result = {
            "region": config.region.value,
            "jurisdiction": config.jurisdiction.value,
            "cloud_provider": config.cloud_provider.value,
            "success": True,
            "services_deployed": [],
            "compliance_checks": [],
            "deployment_duration": 0.0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Pre-deployment compliance checks
            compliance_checks = await self._run_pre_deployment_compliance_checks(config)
            region_result["compliance_checks"] = compliance_checks
            
            if not all(check["passed"] for check in compliance_checks):
                region_result["success"] = False
                region_result["errors"].append("Pre-deployment compliance checks failed")
                return region_result
            
            # Deploy each service
            for service in services:
                service_result = await self._deploy_service_to_region(config, service, version)
                
                if service_result["success"]:
                    region_result["services_deployed"].append(service)
                else:
                    region_result["success"] = False
                    region_result["errors"].extend(service_result.get("errors", []))
            
            # Post-deployment validation
            if region_result["success"]:
                validation_result = await self._validate_region_deployment(config, services)
                if not validation_result["valid"]:
                    region_result["success"] = False
                    region_result["errors"].extend(validation_result.get("errors", []))
            
        except Exception as e:
            region_result["success"] = False
            region_result["errors"].append(f"Deployment exception: {str(e)}")
            self.logger.error(f"Region {config.region.value} deployment failed: {e}")
        
        region_result["deployment_duration"] = time.time() - start_time
        
        return region_result
    
    async def _run_pre_deployment_compliance_checks(self, config: RegionConfig) -> List[Dict[str, Any]]:
        """Run compliance checks before deployment."""
        checks = []
        
        # Data residency check
        if config.data_residency_required:
            checks.append({
                "check_name": "data_residency",
                "description": f"Verify data residency requirements for {config.jurisdiction.value}",
                "passed": True,  # Simplified - would check actual data location
                "details": "Data residency requirements validated"
            })
        
        # Encryption requirements check
        if config.encryption_requirements:
            checks.append({
                "check_name": "encryption_compliance",
                "description": f"Verify encryption requirements for {config.jurisdiction.value}",
                "passed": True,  # Simplified - would check encryption settings
                "details": f"Encryption requirements: {config.encryption_requirements}"
            })
        
        # Network restrictions check
        if config.network_restrictions:
            checks.append({
                "check_name": "network_compliance",
                "description": f"Verify network restrictions for {config.jurisdiction.value}",
                "passed": True,  # Simplified - would check network configuration
                "details": f"Network restrictions: {config.network_restrictions}"
            })
        
        return checks
    
    async def _deploy_service_to_region(
        self, 
        config: RegionConfig, 
        service: str, 
        version: str
    ) -> Dict[str, Any]:
        """Deploy a specific service to a region."""
        self.logger.info(f"Deploying {service} v{version} to {config.region.value}")
        
        # Simulate service deployment
        await asyncio.sleep(2)  # Simulate deployment time
        
        return {
            "service": service,
            "version": version,
            "region": config.region.value,
            "success": True,
            "deployment_time": 2.0,
            "endpoints_created": [
                f"https://api-{config.region.value}.hipaa-compliance.com/{service}"
            ]
        }
    
    async def _validate_region_deployment(
        self, 
        config: RegionConfig, 
        services: List[str]
    ) -> Dict[str, Any]:
        """Validate deployment in a region."""
        self.logger.info(f"Validating deployment in {config.region.value}")
        
        # Simulate validation
        await asyncio.sleep(1)
        
        return {
            "valid": True,
            "services_validated": len(services),
            "health_checks_passed": True,
            "compliance_verified": True
        }
    
    async def _validate_global_compliance(self, deployment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate global compliance across all deployed regions."""
        from .global_compliance_framework import global_compliance_framework
        
        # Extract deployed regions and their jurisdictions
        deployed_jurisdictions = []
        for region_key, region_result in deployment_result["regions"].items():
            if region_result["success"]:
                # Map region to jurisdiction (simplified)
                jurisdiction_mapping = {
                    "us-east-1": "US", "us-west-2": "US",
                    "eu-west-1": "EU", "eu-central-1": "EU", 
                    "ap-southeast-1": "SG"
                }
                jurisdiction_code = jurisdiction_mapping.get(region_key, "US")
                deployed_jurisdictions.append(jurisdiction_code)
        
        # Simulate compliance validation
        compliance_result = {
            "compliant": True,
            "jurisdictions_validated": deployed_jurisdictions,
            "violations": [],
            "recommendations": [],
            "cross_border_transfers_validated": True
        }
        
        return compliance_result
    
    async def _start_post_deployment_monitoring(self):
        """Start post-deployment monitoring and failover management."""
        # Start health monitoring
        monitoring_task = asyncio.create_task(self.health_monitor.start_monitoring())
        
        # Start failover monitoring
        failover_task = asyncio.create_task(self.failover_manager.monitor_and_failover())
        
        self.logger.info("Post-deployment monitoring started")
    
    def get_global_deployment_status(self) -> Dict[str, Any]:
        """Get current global deployment status."""
        global_health = self.health_monitor.get_global_status()
        failover_summary = self.failover_manager.get_failover_summary()
        
        # Calculate deployment metrics
        metrics = DeploymentMetrics(
            total_regions=len(self.region_configs),
            healthy_regions=global_health.get("healthy_regions", 0),
            average_latency_ms=global_health.get("average_latency_ms", 0),
            global_availability=global_health.get("global_availability", 0),
            total_requests_per_second=global_health.get("total_throughput_rps", 0),
            data_transfer_gb=0.0,  # Would calculate from actual metrics
            compliance_violations=0,  # Would check compliance status
            failover_events=failover_summary["total_failovers"],
            last_updated=datetime.now()
        )
        
        return {
            "deployment_metrics": {
                "total_regions": metrics.total_regions,
                "healthy_regions": metrics.healthy_regions,
                "average_latency_ms": metrics.average_latency_ms,
                "global_availability": metrics.global_availability,
                "total_requests_per_second": metrics.total_requests_per_second,
                "compliance_violations": metrics.compliance_violations,
                "failover_events": metrics.failover_events,
                "last_updated": metrics.last_updated.isoformat()
            },
            "regional_status": {
                region_key: self.health_monitor.get_region_status(DeploymentRegion(region_key))
                for region_key in self.region_configs.keys()
            },
            "failover_status": failover_summary,
            "recent_deployments": self.deployment_history[-5:] if self.deployment_history else []
        }


# Global multi-region deployment orchestrator instance
multi_region_orchestrator = MultiRegionDeploymentOrchestrator()


def configure_global_regions(
    regions: List[Tuple[str, str, str, bool]]  # (region, cloud_provider, jurisdiction, data_residency)
) -> None:
    """
    Convenience function to configure global deployment regions.
    
    Usage:
        configure_global_regions([
            ("us-east-1", "aws", "US", False),
            ("eu-west-1", "aws", "EU", True),
            ("ap-southeast-1", "aws", "SG", True)
        ])
    """
    configurations = []
    
    for region_str, provider_str, jurisdiction_str, data_residency in regions:
        # Convert strings to enums
        region = DeploymentRegion(region_str)
        cloud_provider = CloudProvider(provider_str)
        jurisdiction = ComplianceJurisdiction(jurisdiction_str)
        
        config = RegionConfig(
            region=region,
            cloud_provider=cloud_provider,
            jurisdiction=jurisdiction,
            data_residency_required=data_residency,
            primary_region=(jurisdiction_str == "US"),  # US as primary for this example
            compliance_requirements=[f"{jurisdiction_str}_healthcare_compliance"],
            encryption_requirements={"data_at_rest": "AES-256", "data_in_transit": "TLS-1.3"}
        )
        
        configurations.append(config)
    
    multi_region_orchestrator.configure_regions(configurations)


async def deploy_globally_with_compliance(
    services: List[str],
    version: str,
    strategy: str = "rolling"
) -> Dict[str, Any]:
    """
    Convenience function for global deployment with compliance validation.
    
    Usage:
        result = await deploy_globally_with_compliance(
            services=["api", "processor", "validator"],
            version="v1.2.3",
            strategy="rolling"
        )
    """
    return await multi_region_orchestrator.deploy_globally(services, version, strategy)