"""
Configuration module for LLM Cooperation System
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv("API_Key_DeepSeek.env")

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    model_path: str
    deployment_type: str = "openai_api"  # "openai_api" or "vllm_local"
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    # vLLM specific configs (for local deployment)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 32768
    enable_thinking: bool = False
    supported_tasks: List[str] = None
    # API specific configs
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        if self.supported_tasks is None:
            self.supported_tasks = []

@dataclass
class OpenAIAPIConfig:
    """OpenAI API configuration"""
    base_url: str
    api_key: str
    timeout: float = 60.0
    max_retries: int = 3
    default_model: str = "gpt-3.5-turbo"
    
@dataclass
class VLLMConfig:
    """vLLM engine configuration (for local deployment)"""
    host: str = "0.0.0.0"
    port_start: int = 8000
    max_concurrent_requests: int = 1000
    enable_metrics: bool = True
    distributed_executor_backend: str = "mp"
    
class SystemConfig:
    """Main system configuration"""
    
    # OpenAI API configuration (default from environment)
    DEFAULT_OPENAI_CONFIG = OpenAIAPIConfig(
        base_url=os.getenv("BASE_URL", "https://api2.aigcbest.top/v1"),
        api_key=os.getenv("API_KEY", "sk-BHJwrDHeR1CXL83svRkwZx0Z9OF4K9LsQDrtEQSbQCCOPA7K"),
        timeout=60.0,
        max_retries=3
    )
    
    # Model configurations - using OpenAI API by default
    MODELS = {
        "qwen3_32b_router": ModelConfig(
            name="qwen3_32b_router",
            model_path="Qwen/Qwen3-32B",
            deployment_type="openai_api",
            api_base_url=DEFAULT_OPENAI_CONFIG.base_url,
            api_key=DEFAULT_OPENAI_CONFIG.api_key,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for routing decisions
            enable_thinking=False,
            supported_tasks=["routing", "analysis", "classification"]
        ),
        "qwen3_32b_reasoning": ModelConfig(
            name="qwen3_32b_reasoning", 
            model_path="Qwen/Qwen3-32B",
            deployment_type="openai_api",
            api_base_url=DEFAULT_OPENAI_CONFIG.base_url,
            api_key=DEFAULT_OPENAI_CONFIG.api_key,
            max_tokens=8192,
            temperature=0.7,
            enable_thinking=True,
            supported_tasks=["reasoning", "math", "logic", "code_debug"]
        ),
        "qwen3_8b": ModelConfig(
            name="qwen3_8b",
            model_path="Qwen/Qwen3-8B",
            deployment_type="openai_api",
            api_base_url=DEFAULT_OPENAI_CONFIG.base_url,
            api_key=DEFAULT_OPENAI_CONFIG.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["text_generation", "summarization", "translation", "extraction"]
        ),
        "qwen2_5_vl_72b": ModelConfig(
            name="qwen2_5_vl_72b",
            model_path="Qwen/Qwen2.5-VL-72B-Instruct", 
            deployment_type="openai_api",
            api_base_url=DEFAULT_OPENAI_CONFIG.base_url,
            api_key=DEFAULT_OPENAI_CONFIG.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["vision", "multimodal", "image_analysis", "chart_analysis"]
        ),
        "qwen2_5_7b": ModelConfig(
            name="qwen2_5_7b",
            model_path="Qwen/Qwen2.5-7B-Instruct",
            deployment_type="openai_api",
            api_base_url=DEFAULT_OPENAI_CONFIG.base_url,
            api_key=DEFAULT_OPENAI_CONFIG.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["text_generation", "summarization", "translation", "extraction"]
        ),
        # Example of vLLM local deployment (commented out)
        # "local_model": ModelConfig(
        #     name="local_model",
        #     model_path="Qwen/Qwen2.5-7B-Instruct",
        #     deployment_type="vllm_local",
        #     tensor_parallel_size=1,
        #     gpu_memory_utilization=0.7,
        #     max_model_len=32768,
        #     supported_tasks=["text_generation"]
        # )
    }
    
    # vLLM engine configuration
    VLLM_CONFIG = VLLMConfig()
    
    # Routing configuration
    ROUTING_CONFIG = {
        "default_timeout": 30.0,
        "max_retries": 3,
        "load_balance_strategy": "least_loaded",
        "performance_threshold": 0.95,
        "cooperation_modes": ["sequential", "parallel", "voting"]
    }
    
    # Task type mappings
    TASK_MAPPINGS = {
        "math": ["qwen3_32b_reasoning"],
        "code": ["qwen3_32b_reasoning"],
        "vision": ["qwen2_5_vl_72b"],
        "multimodal": ["qwen2_5_vl_72b"],
        "text": ["qwen3_8b", "qwen2_5_7b"],
        "complex_reasoning": ["qwen3_32b_reasoning"],
        "simple_qa": ["qwen3_8b", "qwen2_5_7b"],
        "routing": ["qwen3_32b_router"],
        "cooperation": ["qwen3_32b_router", "qwen3_32b_reasoning", "qwen3_8b"]
    }
    
    # Performance monitoring
    MONITORING_CONFIG = {
        "metrics_interval": 10,
        "health_check_interval": 30,
        "performance_log_path": "./logs/performance.log",
        "enable_prometheus": True,
        "prometheus_port": 9090
    }
    
    @classmethod
    def get_model_port(cls, model_name: str) -> int:
        """Get assigned port for a model (only for vLLM local deployment)"""
        model_names = list(cls.MODELS.keys())
        if model_name not in model_names:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.VLLM_CONFIG.port_start + model_names.index(model_name)
    
    @classmethod
    def get_models_for_task(cls, task_type: str) -> List[str]:
        """Get suitable models for a task type"""
        return cls.TASK_MAPPINGS.get(task_type, ["qwen3_8b"])
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.MODELS[model_name]
    
    @classmethod
    def update_api_config(cls, base_url: str, api_key: str):
        """Update API configuration for all models"""
        cls.DEFAULT_OPENAI_CONFIG.base_url = base_url
        cls.DEFAULT_OPENAI_CONFIG.api_key = api_key
        
        # Update all model configurations
        for model_config in cls.MODELS.values():
            if model_config.deployment_type == "openai_api":
                model_config.api_base_url = base_url
                model_config.api_key = api_key
    
    @classmethod
    def add_custom_model(cls, model_name: str, model_path: str, 
                        deployment_type: str = "openai_api",
                        api_base_url: str = None, api_key: str = None,
                        supported_tasks: List[str] = None, **kwargs):
        """Add a custom model configuration"""
        if api_base_url is None:
            api_base_url = cls.DEFAULT_OPENAI_CONFIG.base_url
        if api_key is None:
            api_key = cls.DEFAULT_OPENAI_CONFIG.api_key
        
        cls.MODELS[model_name] = ModelConfig(
            name=model_name,
            model_path=model_path,
            deployment_type=deployment_type,
            api_base_url=api_base_url,
            api_key=api_key,
            supported_tasks=supported_tasks or ["text_generation"],
            **kwargs
        )