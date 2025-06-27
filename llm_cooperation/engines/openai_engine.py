"""
OpenAI API Engine Layer
Provides unified inference services using OpenAI-compatible APIs
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import time
import json
from openai import AsyncOpenAI
from llm_cooperation.config import SystemConfig, ModelConfig

@dataclass
class InferenceRequest:
    """Inference request structure"""
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
    """Inference response structure"""
    text: str
    model_name: str
    usage: Dict[str, int]
    latency: float
    success: bool = True
    error: Optional[str] = None

class OpenAIEngine:
    """OpenAI API inference engine with multi-provider support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clients: Dict[str, AsyncOpenAI] = {}
        self.model_configs = SystemConfig.MODELS
        self.session_timeout = aiohttp.ClientTimeout(total=120)
        
    async def initialize(self):
        """Initialize the OpenAI API engine"""
        self.logger.info("Initializing OpenAI API inference engine...")
        
        # Create clients for each unique API endpoint
        unique_endpoints = {}
        for model_name, config in self.model_configs.items():
            if config.deployment_type == "openai_api":
                endpoint_key = f"{config.api_base_url}:{config.api_key}"
                if endpoint_key not in unique_endpoints:
                    unique_endpoints[endpoint_key] = {
                        "base_url": config.api_base_url,
                        "api_key": config.api_key,
                        "models": []
                    }
                unique_endpoints[endpoint_key]["models"].append(model_name)
        
        # Create clients
        for endpoint_key, endpoint_info in unique_endpoints.items():
            try:
                client = AsyncOpenAI(
                    base_url=endpoint_info["base_url"],
                    api_key=endpoint_info["api_key"],
                    timeout=60.0,
                    max_retries=3
                )
                
                # Store client for each model
                for model_name in endpoint_info["models"]:
                    self.clients[model_name] = client
                
                self.logger.info(f"Created client for {endpoint_info['base_url']} "
                               f"serving models: {endpoint_info['models']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create client for {endpoint_info['base_url']}: {e}")
        
        # Test connectivity
        await self._test_connectivity()
        
        self.logger.info("OpenAI API inference engine initialized successfully")
    
    async def _test_connectivity(self):
        """Test connectivity to all configured models"""
        self.logger.info("Testing connectivity to configured models...")
        
        test_tasks = []
        for model_name in self.clients.keys():
            task = asyncio.create_task(self._test_model_connectivity(model_name))
            test_tasks.append(task)
        
        if test_tasks:
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            successful_tests = sum(1 for r in results if r is True)
            total_tests = len(results)
            
            self.logger.info(f"Connectivity test results: {successful_tests}/{total_tests} models accessible")
    
    async def _test_model_connectivity(self, model_name: str) -> bool:
        """Test connectivity to a specific model"""
        try:
            test_request = InferenceRequest(
                prompt="Hello",
                model_name=model_name,
                max_tokens=5,
                temperature=0.1
            )
            
            response = await self.inference(test_request)
            
            if response.success:
                self.logger.debug(f"Model {model_name} is accessible")
                return True
            else:
                self.logger.warning(f"Model {model_name} test failed: {response.error}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Model {model_name} connectivity test failed: {e}")
            return False
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference using OpenAI API"""
        start_time = time.time()
        
        try:
            if request.model_name not in self.clients:
                raise ValueError(f"Model {request.model_name} not configured")
            
            client = self.clients[request.model_name]
            model_config = self.model_configs[request.model_name]
            
            # Prepare messages
            messages = [{"role": "user", "content": request.prompt}]
            
            # Add system message for thinking models
            if model_config.enable_thinking:
                system_message = {
                    "role": "system", 
                    "content": "You are a helpful assistant. Think step by step and provide detailed reasoning for complex problems."
                }
                messages.insert(0, system_message)
            
            # Prepare API parameters
            api_params = {
                "model": model_config.model_path,
                "messages": messages,
                "max_tokens": min(request.max_tokens, model_config.max_tokens),
                "temperature": request.temperature if request.temperature is not None else model_config.temperature,
                "top_p": request.top_p,
                "stream": request.stream
            }
            
            # Add extra parameters
            if request.extra_params:
                api_params.update(request.extra_params)
            
            # Make API call
            response = await client.chat.completions.create(**api_params)
            
            # Extract response
            if request.stream:
                # Handle streaming response
                response_text = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                
                # Estimate usage for streaming
                usage = {
                    "prompt_tokens": len(request.prompt.split()) * 1.3,
                    "completion_tokens": len(response_text.split()) * 1.3,
                    "total_tokens": len(request.prompt.split()) * 1.3 + len(response_text.split()) * 1.3
                }
            else:
                # Handle regular response
                response_text = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            
            return InferenceResponse(
                text=response_text,
                model_name=request.model_name,
                usage=usage,
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Inference failed for {request.model_name}: {e}")
            return InferenceResponse(
                text="",
                model_name=request.model_name,
                usage={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Perform batch inference"""
        tasks = [self.inference(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        if model_name not in self.model_configs:
            return {"error": f"Model {model_name} not found"}
        
        config = self.model_configs[model_name]
        client = self.clients.get(model_name)
        
        info = {
            "name": model_name,
            "model_path": config.model_path,
            "deployment_type": config.deployment_type,
            "api_base_url": config.api_base_url,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "supported_tasks": config.supported_tasks,
            "status": "available" if client else "unavailable"
        }
        
        # Test model availability
        try:
            if client:
                test_response = await self._test_model_connectivity(model_name)
                info["status"] = "healthy" if test_response else "error"
        except:
            info["status"] = "error"
        
        return info
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models_info = []
        
        for model_name in self.model_configs.keys():
            model_info = await self.get_model_info(model_name)
            models_info.append(model_info)
        
        return models_info
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        total_models = len(self.model_configs)
        available_models = len(self.clients)
        
        # Test healthy models
        health_tasks = [self._test_model_connectivity(model_name) 
                       for model_name in self.clients.keys()]
        
        if health_tasks:
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            healthy_models = sum(1 for r in health_results if r is True)
        else:
            healthy_models = 0
        
        return {
            "total_models": total_models,
            "available_models": available_models,
            "healthy_models": healthy_models,
            "health_percentage": (healthy_models / max(total_models, 1)) * 100,
            "deployment_types": list(set(config.deployment_type for config in self.model_configs.values())),
            "api_endpoints": list(set(config.api_base_url for config in self.model_configs.values() if config.api_base_url))
        }
    
    def update_model_config(self, model_name: str, **kwargs):
        """Update model configuration"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.model_configs[model_name]
        
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # If API configuration changed, recreate client
        if any(key in kwargs for key in ['api_base_url', 'api_key']):
            asyncio.create_task(self._recreate_client(model_name))
    
    async def _recreate_client(self, model_name: str):
        """Recreate client for a model"""
        try:
            config = self.model_configs[model_name]
            
            if config.deployment_type == "openai_api":
                client = AsyncOpenAI(
                    base_url=config.api_base_url,
                    api_key=config.api_key,
                    timeout=60.0,
                    max_retries=3
                )
                
                # Test the new client
                old_client = self.clients.get(model_name)
                test_request = InferenceRequest(
                    prompt="Test",
                    model_name=model_name,
                    max_tokens=5
                )
                
                # Temporarily use new client for test
                self.clients[model_name] = client
                test_result = await self._test_model_connectivity(model_name)
                
                if test_result:
                    self.logger.info(f"Successfully updated client for {model_name}")
                    # Close old client if it exists
                    if old_client:
                        await old_client.close()
                else:
                    # Restore old client if test failed
                    if old_client:
                        self.clients[model_name] = old_client
                    else:
                        del self.clients[model_name]
                    raise Exception("Client test failed")
                
        except Exception as e:
            self.logger.error(f"Failed to recreate client for {model_name}: {e}")
    
    async def add_model(self, model_name: str, model_path: str, 
                       api_base_url: str, api_key: str,
                       supported_tasks: List[str] = None, **kwargs):
        """Add a new model configuration"""
        # Add to system config
        SystemConfig.add_custom_model(
            model_name=model_name,
            model_path=model_path,
            api_base_url=api_base_url,
            api_key=api_key,
            supported_tasks=supported_tasks,
            **kwargs
        )
        
        # Update local config reference
        self.model_configs = SystemConfig.MODELS
        
        # Create client
        try:
            client = AsyncOpenAI(
                base_url=api_base_url,
                api_key=api_key,
                timeout=60.0,
                max_retries=3
            )
            
            self.clients[model_name] = client
            
            # Test connectivity
            test_result = await self._test_model_connectivity(model_name)
            
            if test_result:
                self.logger.info(f"Successfully added model {model_name}")
                return True
            else:
                self.logger.warning(f"Model {model_name} added but not accessible")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to add model {model_name}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all API clients"""
        self.logger.info("Shutting down OpenAI API engine...")
        
        # Close all clients
        for model_name, client in self.clients.items():
            try:
                await client.close()
                self.logger.debug(f"Closed client for {model_name}")
            except Exception as e:
                self.logger.error(f"Error closing client for {model_name}: {e}")
        
        self.clients.clear()
        self.logger.info("OpenAI API engine shutdown complete")

# Global engine instance
engine = OpenAIEngine()