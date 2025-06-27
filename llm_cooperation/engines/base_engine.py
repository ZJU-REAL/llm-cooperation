"""
Base engine interface for inference engines
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class InferenceRequest:
    """Base inference request structure"""
    prompt: str
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

@dataclass
class InferenceResponse:
    """Base inference response structure"""
    text: str
    model_name: str
    usage: Dict[str, int]
    latency: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseEngine(ABC):
    """Abstract base class for inference engines"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the inference engine"""
        pass
    
    @abstractmethod
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference on specified model"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the inference engine"""
        pass
    
    async def batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Perform batch inference (default implementation)"""
        import asyncio
        tasks = [self.inference(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self.initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Simple test inference
            test_request = InferenceRequest(
                prompt="Test",
                model_name="default",
                max_tokens=1
            )
            
            start_time = time.time()
            response = await self.inference(test_request)
            latency = time.time() - start_time
            
            return {
                "status": "healthy" if response.success else "error",
                "healthy": response.success,
                "latency": latency,
                "error": response.error if not response.success else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "error": str(e)
            }
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "type": self.__class__.__module__ + "." + self.__class__.__name__
        }