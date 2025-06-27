"""
Model Resource Management Layer
Handles model lifecycle, resource monitoring, and load balancing
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from llm_cooperation.config import SystemConfig
from llm_cooperation.engines.openai_engine import engine, InferenceRequest, InferenceResponse

class ModelStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class ModelMetrics:
    """Model performance and resource metrics"""
    model_name: str
    status: ModelStatus
    request_count: int = 0
    avg_latency: float = 0.0
    queue_length: int = 0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    # Performance history for trend analysis
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass 
class LoadBalanceStrategy:
    """Load balancing strategy configuration"""
    strategy_type: str = "least_loaded"  # least_loaded, round_robin, weighted
    weights: Dict[str, float] = field(default_factory=dict)
    health_threshold: float = 0.95
    max_queue_length: int = 50

class ModelResourceManager:
    """Manages model resources, monitoring, and load balancing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_queues: Dict[str, asyncio.Queue] = {}
        self.load_balance_strategy = LoadBalanceStrategy()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.request_history: deque = deque(maxlen=1000)
        
        # Initialize model metrics
        for model_name in SystemConfig.MODELS.keys():
            self.model_metrics[model_name] = ModelMetrics(
                model_name=model_name,
                status=ModelStatus.STARTING
            )
            self.model_queues[model_name] = asyncio.Queue(maxsize=100)
    
    async def initialize(self):
        """Initialize the model resource manager"""
        self.logger.info("Initializing Model Resource Manager...")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Wait for models to be ready
        await self._wait_for_models_ready()
        
        self.logger.info("Model Resource Manager initialized")
    
    async def _wait_for_models_ready(self, timeout: int = 300):
        """Wait for all models to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = 0
            
            for model_name in self.model_metrics.keys():
                if await self._check_model_ready(model_name):
                    self.model_metrics[model_name].status = ModelStatus.READY
                    ready_count += 1
                else:
                    self.model_metrics[model_name].status = ModelStatus.STARTING
            
            if ready_count == len(self.model_metrics):
                self.logger.info("All models are ready")
                return
            
            await asyncio.sleep(5)
        
        raise TimeoutError("Models not ready within timeout")
    
    async def _check_model_ready(self, model_name: str) -> bool:
        """Check if a model is ready"""
        try:
            # Check if model is available in OpenAI engine
            model_info = await engine.get_model_info(model_name)
            return model_info.get("status") == "healthy"
        except Exception:
            return False
    
    async def _monitoring_loop(self):
        """Continuous monitoring of model metrics"""
        while True:
            try:
                await self._update_all_metrics()
                await asyncio.sleep(SystemConfig.MONITORING_CONFIG["metrics_interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _update_all_metrics(self):
        """Update metrics for all models"""
        tasks = [self._update_model_metrics(model_name) 
                for model_name in self.model_metrics.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update system-wide metrics
        await self._update_system_metrics()
    
    async def _update_model_metrics(self, model_name: str):
        """Update metrics for a specific model"""
        try:
            # Get model info from OpenAI engine
            model_info = await engine.get_model_info(model_name)
            
            metrics = self.model_metrics[model_name]
            
            # Update status based on model info
            if model_info.get("status") == "healthy":
                metrics.status = ModelStatus.READY
            elif model_info.get("status") == "available":
                metrics.status = ModelStatus.STARTING
            else:
                metrics.status = ModelStatus.ERROR
            
            # For API models, we don't have detailed metrics like vLLM
            # We track basic performance through request history
            current_time = time.time()
            time_diff = current_time - metrics.last_updated
            
            # Estimate throughput from request history
            if hasattr(self, 'request_history') and time_diff > 0:
                recent_requests = [
                    r for r in self.request_history 
                    if r.get('model') == model_name and 
                       r.get('timestamp', 0) > current_time - 60  # Last minute
                ]
                metrics.throughput = len(recent_requests) / min(time_diff, 60)
                metrics.throughput_history.append(metrics.throughput)
            
            # Queue length is not directly available for API models
            # We can estimate based on recent response times
            if metrics.latency_history:
                avg_latency = statistics.mean(metrics.latency_history)
                # Simple heuristic: if latency is increasing, assume queue is building
                if len(metrics.latency_history) > 10:
                    recent_avg = statistics.mean(list(metrics.latency_history)[-10:])
                    if recent_avg > avg_latency * 1.5:
                        metrics.queue_length = min(int(recent_avg / avg_latency), 20)
                    else:
                        metrics.queue_length = 0
            
            # API models don't have GPU metrics, set to 0
            metrics.gpu_utilization = 0.0
            metrics.memory_usage = 0.0
            
            metrics.last_updated = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics for {model_name}: {e}")
            self.model_metrics[model_name].status = ModelStatus.ERROR
    
    async def _update_system_metrics(self):
        """Update system-wide performance metrics"""
        # Calculate overall system health
        ready_models = sum(1 for m in self.model_metrics.values() 
                          if m.status == ModelStatus.READY)
        total_models = len(self.model_metrics)
        
        system_health = ready_models / total_models if total_models > 0 else 0
        
        self.logger.debug(f"System health: {system_health:.2%} ({ready_models}/{total_models} models ready)")
    
    def select_best_model(self, task_type: str, requirements: Dict = None) -> str:
        """Select the best model for a given task"""
        suitable_models = SystemConfig.get_models_for_task(task_type)
        
        if not suitable_models:
            return list(SystemConfig.MODELS.keys())[0]  # Default fallback
        
        # Filter by availability and health
        available_models = [
            model for model in suitable_models
            if self.model_metrics[model].status == ModelStatus.READY
        ]
        
        if not available_models:
            # If no models are ready, return the first suitable one
            return suitable_models[0]
        
        return self._apply_load_balance_strategy(available_models)
    
    def _apply_load_balance_strategy(self, models: List[str]) -> str:
        """Apply load balancing strategy to select from available models"""
        strategy = self.load_balance_strategy.strategy_type
        
        if strategy == "least_loaded":
            return min(models, key=lambda m: self.model_metrics[m].queue_length)
        
        elif strategy == "round_robin":
            # Simple round-robin based on request count
            return min(models, key=lambda m: self.model_metrics[m].request_count)
        
        elif strategy == "weighted":
            # Weighted selection based on performance
            best_model = None
            best_score = float('inf')
            
            for model in models:
                metrics = self.model_metrics[model]
                weight = self.load_balance_strategy.weights.get(model, 1.0)
                
                # Score based on latency and queue length
                score = (metrics.avg_latency * metrics.queue_length) / weight
                
                if score < best_score:
                    best_score = score
                    best_model = model
            
            return best_model or models[0]
        
        return models[0]  # Default to first model
    
    async def get_model_recommendations(self, task_type: str) -> List[Tuple[str, float]]:
        """Get ranked model recommendations for a task"""
        suitable_models = SystemConfig.get_models_for_task(task_type)
        recommendations = []
        
        for model in suitable_models:
            metrics = self.model_metrics[model]
            
            # Calculate recommendation score
            if metrics.status != ModelStatus.READY:
                score = 0.0
            else:
                # Score based on performance metrics (higher is better)
                latency_score = 1.0 / max(metrics.avg_latency, 0.1)
                queue_score = 1.0 / max(metrics.queue_length + 1, 1)
                throughput_score = metrics.throughput
                
                score = (latency_score + queue_score + throughput_score) / 3
            
            recommendations.append((model, score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    async def execute_request(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference request with resource management"""
        start_time = time.time()
        
        try:
            # Update request metrics
            metrics = self.model_metrics[request.model_name]
            metrics.request_count += 1
            
            # Execute inference
            response = await engine.inference(request)
            
            # Update performance metrics
            latency = time.time() - start_time
            metrics.latency_history.append(latency)
            
            if metrics.latency_history:
                metrics.avg_latency = statistics.mean(metrics.latency_history)
            
            # Update error rate
            if not response.success:
                # Simple error rate calculation (last 100 requests)
                error_count = sum(1 for r in self.request_history 
                                if not r.get('success', True))
                metrics.error_rate = error_count / len(self.request_history)
            
            # Record request in history
            self.request_history.append({
                'model': request.model_name,
                'latency': latency,
                'success': response.success,
                'timestamp': start_time
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request execution failed: {e}")
            # Update error metrics
            metrics = self.model_metrics[request.model_name]
            metrics.status = ModelStatus.ERROR
            
            return InferenceResponse(
                text="",
                model_name=request.model_name,
                usage={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        model_status = {}
        for model_name, metrics in self.model_metrics.items():
            model_status[model_name] = {
                "status": metrics.status.value,
                "request_count": metrics.request_count,
                "avg_latency": metrics.avg_latency,
                "queue_length": metrics.queue_length,
                "memory_usage": metrics.memory_usage,
                "gpu_utilization": metrics.gpu_utilization,
                "error_rate": metrics.error_rate,
                "throughput": metrics.throughput
            }
        
        return {
            "models": model_status,
            "total_requests": len(self.request_history),
            "system_uptime": time.time(),
            "load_balance_strategy": self.load_balance_strategy.strategy_type
        }
    
    async def shutdown(self):
        """Shutdown model resource manager"""
        self.logger.info("Shutting down Model Resource Manager...")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Model Resource Manager shutdown complete")

# Global manager instance
model_manager = ModelResourceManager()