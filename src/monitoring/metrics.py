"""
Monitoring and Metrics
Author: Balaji Koneti

Comprehensive monitoring system with Prometheus metrics, Grafana dashboards,
and distributed tracing using OpenTelemetry.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricData:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime


class MetricsCollector:
    """Metrics collector for system monitoring"""
    
    def __init__(self):
        # Prometheus metrics
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_counter = Counter(
            'prompt_architect_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'prompt_architect_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_requests = Counter(
            'prompt_architect_llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_response_time = Histogram(
            'prompt_architect_llm_response_time_seconds',
            'LLM response time in seconds',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.llm_tokens = Counter(
            'prompt_architect_llm_tokens_total',
            'Total tokens processed',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        
        self.llm_cost = Counter(
            'prompt_architect_llm_cost_total',
            'Total LLM cost in USD',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Reasoning metrics
        self.reasoning_sessions = Counter(
            'prompt_architect_reasoning_sessions_total',
            'Total reasoning sessions',
            ['strategy', 'status'],
            registry=self.registry
        )
        
        self.reasoning_steps = Histogram(
            'prompt_architect_reasoning_steps_total',
            'Number of reasoning steps',
            ['strategy'],
            registry=self.registry
        )
        
        self.reasoning_confidence = Histogram(
            'prompt_architect_reasoning_confidence',
            'Reasoning confidence scores',
            ['strategy'],
            registry=self.registry
        )
        
        # Template metrics
        self.template_generations = Counter(
            'prompt_architect_template_generations_total',
            'Total template generations',
            ['category', 'complexity', 'status'],
            registry=self.registry
        )
        
        self.template_selections = Counter(
            'prompt_architect_template_selections_total',
            'Total template selections',
            ['category', 'confidence_level'],
            registry=self.registry
        )
        
        # System metrics
        self.active_users = Gauge(
            'prompt_architect_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.system_health = Gauge(
            'prompt_architect_system_health',
            'System health score (0-1)',
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'prompt_architect_errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_history: deque = deque(maxlen=100)
        
        logger.info("MetricsCollector initialized")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_llm_request(self, provider: str, model: str, status: str, 
                          response_time: float, tokens: Dict[str, int], cost: float):
        """Record LLM request metrics"""
        self.llm_requests.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        self.llm_response_time.labels(
            provider=provider,
            model=model
        ).observe(response_time)
        
        # Record token usage
        for token_type, count in tokens.items():
            self.llm_tokens.labels(
                provider=provider,
                model=model,
                type=token_type
            ).inc(count)
        
        # Record cost
        self.llm_cost.labels(
            provider=provider,
            model=model
        ).inc(cost)
    
    def record_reasoning_session(self, strategy: str, status: str, 
                                steps: int, confidence: float):
        """Record reasoning session metrics"""
        self.reasoning_sessions.labels(
            strategy=strategy,
            status=status
        ).inc()
        
        self.reasoning_steps.labels(strategy=strategy).observe(steps)
        self.reasoning_confidence.labels(strategy=strategy).observe(confidence)
    
    def record_template_generation(self, category: str, complexity: str, status: str):
        """Record template generation metrics"""
        self.template_generations.labels(
            category=category,
            complexity=complexity,
            status=status
        ).inc()
    
    def record_template_selection(self, category: str, confidence: float):
        """Record template selection metrics"""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        self.template_selections.labels(
            category=category,
            confidence_level=confidence_level
        ).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        self.error_counter.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def update_system_metrics(self, active_users: int, health_score: float):
        """Update system-level metrics"""
        self.active_users.set(active_users)
        self.system_health.set(health_score)
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record custom metric"""
        metric_data = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.custom_metrics[name].append(metric_data)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "timestamp": datetime.now(),
            "custom_metrics": {
                name: len(metrics) for name, metrics in self.custom_metrics.items()
            },
            "performance_history": len(self.performance_history),
            "prometheus_metrics": self._get_prometheus_summary()
        }
    
    def _get_prometheus_summary(self) -> Dict[str, Any]:
        """Get Prometheus metrics summary"""
        try:
            metrics_data = generate_latest(self.registry).decode('utf-8')
            return {"status": "available", "metrics_count": len(metrics_data.split('\n'))}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_performance_metrics(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get performance metrics for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.performance_history
            if metric.timestamp >= cutoff_time
        ]


class HealthChecker:
    """Health checker for system components"""
    
    def __init__(self):
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.last_check: Dict[str, datetime] = {}
        self.check_interval = 30  # seconds
        
    async def check_component_health(self, component: str, check_func) -> Dict[str, Any]:
        """Check health of a specific component"""
        try:
            start_time = time.time()
            result = await check_func()
            response_time = time.time() - start_time
            
            health_data = {
                "status": "healthy",
                "response_time": response_time,
                "last_check": datetime.now(),
                "details": result
            }
            
            self.health_status[component] = health_data
            self.last_check[component] = datetime.now()
            
            return health_data
            
        except Exception as e:
            health_data = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now()
            }
            
            self.health_status[component] = health_data
            self.last_check[component] = datetime.now()
            
            logger.error("Health check failed", component=component, error=str(e))
            return health_data
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_status:
            return {"status": "unknown", "components": {}}
        
        healthy_components = sum(1 for status in self.health_status.values() 
                               if status["status"] == "healthy")
        total_components = len(self.health_status)
        
        overall_health = healthy_components / total_components if total_components > 0 else 0
        
        return {
            "status": "healthy" if overall_health > 0.8 else "degraded" if overall_health > 0.5 else "unhealthy",
            "health_score": overall_health,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "components": self.health_status,
            "last_updated": datetime.now()
        }


class AlertManager:
    """Alert manager for system monitoring"""
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 5.0,  # 5 seconds
            "cpu_usage": 0.8,  # 80% CPU usage
            "memory_usage": 0.9,  # 90% memory usage
            "health_score": 0.7  # 70% health score
        }
        
    def add_alert_rule(self, name: str, condition: str, threshold: float, severity: str = "warning"):
        """Add alert rule"""
        self.alert_rules[name] = {
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "enabled": True
        }
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
                
            try:
                # Evaluate condition (simplified - in production, use proper expression evaluator)
                if self._evaluate_condition(rule["condition"], metrics, rule["threshold"]):
                    self._trigger_alert(rule_name, rule, metrics)
            except Exception as e:
                logger.error("Alert rule evaluation failed", rule=rule_name, error=str(e))
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any], threshold: float) -> bool:
        """Evaluate alert condition"""
        # Simplified condition evaluation
        if condition == "error_rate":
            return metrics.get("error_rate", 0) > threshold
        elif condition == "response_time":
            return metrics.get("avg_response_time", 0) > threshold
        elif condition == "cpu_usage":
            return metrics.get("cpu_usage", 0) > threshold
        elif condition == "memory_usage":
            return metrics.get("memory_usage", 0) > threshold
        elif condition == "health_score":
            return metrics.get("health_score", 1) < threshold
        
        return False
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger alert"""
        alert = {
            "id": f"alert_{int(time.time())}_{rule_name}",
            "rule_name": rule_name,
            "severity": rule["severity"],
            "message": f"Alert triggered: {rule_name}",
            "timestamp": datetime.now(),
            "metrics": metrics,
            "resolved": False
        }
        
        self.alerts.append(alert)
        logger.warning("Alert triggered", alert=alert)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return [alert for alert in self.alerts if not alert["resolved"]]
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now()
                break


class MonitoringSystem:
    """Main monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.is_running = False
        self.monitoring_task = None
        
        # Initialize default alert rules
        self._setup_default_alerts()
        
        logger.info("MonitoringSystem initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        for condition, threshold in self.alert_manager.alert_thresholds.items():
            severity = "critical" if condition in ["error_rate", "health_score"] else "warning"
            self.alert_manager.add_alert_rule(
                f"{condition}_alert",
                condition,
                threshold,
                severity
            )
    
    async def start_monitoring(self):
        """Start monitoring system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring system started")
    
    async def stop_monitoring(self):
        """Stop monitoring system"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check health
                await self._check_health()
                
                # Check alerts
                await self._check_alerts()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # This would integrate with actual system metrics collection
            # For now, we'll simulate some metrics
            import psutil
            
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Record custom metrics
            self.metrics_collector.record_custom_metric("system_cpu_usage", cpu_usage)
            self.metrics_collector.record_custom_metric("system_memory_usage", memory_usage)
            
            # Update performance history
            performance_metric = PerformanceMetrics(
                response_time=0.0,  # Would be calculated from actual requests
                throughput=0.0,     # Would be calculated from actual requests
                error_rate=0.0,     # Would be calculated from actual requests
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                timestamp=datetime.now()
            )
            
            self.metrics_collector.performance_history.append(performance_metric)
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    async def _check_health(self):
        """Check system health"""
        # This would check various system components
        # For now, we'll just update the health score based on available metrics
        health_score = 1.0  # Default healthy
        
        # Check if we have recent metrics
        if self.metrics_collector.performance_history:
            latest_metric = self.metrics_collector.performance_history[-1]
            if latest_metric.cpu_usage > 90 or latest_metric.memory_usage > 90:
                health_score = 0.5  # Degraded
            if latest_metric.cpu_usage > 95 or latest_metric.memory_usage > 95:
                health_score = 0.2  # Unhealthy
        
        self.metrics_collector.update_system_metrics(active_users=0, health_score=health_score)
    
    async def _check_alerts(self):
        """Check alerts"""
        try:
            # Get current metrics
            metrics = {
                "error_rate": 0.0,  # Would be calculated from actual error data
                "avg_response_time": 0.0,  # Would be calculated from actual request data
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "health_score": 1.0
            }
            
            # Update with actual system metrics if available
            if self.metrics_collector.performance_history:
                latest_metric = self.metrics_collector.performance_history[-1]
                metrics["cpu_usage"] = latest_metric.cpu_usage / 100.0
                metrics["memory_usage"] = latest_metric.memory_usage / 100.0
            
            # Check alerts
            self.alert_manager.check_alerts(metrics)
            
        except Exception as e:
            logger.error("Failed to check alerts", error=str(e))
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "health_status": self.health_checker.get_overall_health(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "performance_metrics": [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "cpu_usage": metric.cpu_usage,
                    "memory_usage": metric.memory_usage,
                    "response_time": metric.response_time,
                    "throughput": metric.throughput,
                    "error_rate": metric.error_rate
                }
                for metric in self.metrics_collector.performance_history
            ]
        }
