"""
Model configuration templates and presets
"""
from typing import Dict, List
from .base_config import ModelConfig, OpenAIAPIConfig

def create_default_models(openai_config: OpenAIAPIConfig) -> Dict[str, ModelConfig]:
    """Create default model configurations"""
    
    models = {
        "qwen3_32b_router": ModelConfig(
            name="qwen3_32b_router",
            model_path="Qwen/Qwen3-32B",
            deployment_type="openai_api",
            api_base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for routing decisions
            enable_thinking=False,
            supported_tasks=["routing", "analysis", "classification"]
        ),
        "qwen3_32b_reasoning": ModelConfig(
            name="qwen3_32b_reasoning", 
            model_path="Qwen/Qwen3-32B",
            deployment_type="openai_api",
            api_base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            max_tokens=8192,
            temperature=0.7,
            enable_thinking=True,
            supported_tasks=["reasoning", "math", "logic", "code_debug"]
        ),
        "qwen3_8b": ModelConfig(
            name="qwen3_8b",
            model_path="Qwen/Qwen3-8B",
            deployment_type="openai_api",
            api_base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["text_generation", "summarization", "translation", "extraction"]
        ),
        "qwen2_5_vl_72b": ModelConfig(
            name="qwen2_5_vl_72b",
            model_path="Qwen/Qwen2.5-VL-72B-Instruct", 
            deployment_type="openai_api",
            api_base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["vision", "multimodal", "image_analysis", "chart_analysis"]
        ),
        "qwen2_5_7b": ModelConfig(
            name="qwen2_5_7b",
            model_path="Qwen/Qwen2.5-7B-Instruct",
            deployment_type="openai_api",
            api_base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            max_tokens=4096,
            temperature=0.7,
            supported_tasks=["text_generation", "summarization", "translation", "extraction"]
        )
    }
    
    return models

def create_preset_configs() -> Dict[str, Dict]:
    """Create preset configurations for common providers"""
    
    presets = {
        "aigcbest": {
            "name": "AIGC Best",
            "base_url": "https://api2.aigcbest.top/v1",
            "description": "AIGC Best API service with multiple model providers",
            "models": {
                "qwen3_32b": {
                    "path": "Qwen/Qwen3-32B",
                    "tasks": ["reasoning", "math", "logic", "code_debug"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "qwen3_8b": {
                    "path": "Qwen/Qwen3-8B",
                    "tasks": ["text_generation", "summarization", "translation"],
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "qwen2_5_7b": {
                    "path": "Qwen/Qwen2.5-7B-Instruct",
                    "tasks": ["text_generation", "extraction"],
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "qwen2_5_vl_72b": {
                    "path": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "tasks": ["vision", "multimodal", "image_analysis"],
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "claude_3_5_sonnet": {
                    "path": "anthropic/claude-3-5-sonnet-20241022",
                    "tasks": ["reasoning", "analysis", "writing"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "gpt4o": {
                    "path": "openai/gpt-4o",
                    "tasks": ["reasoning", "analysis", "code_debug"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "deepseek_v3": {
                    "path": "deepseek/deepseek-v3",
                    "tasks": ["reasoning", "math", "code_debug"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                }
            }
        },
        "openai": {
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "description": "Official OpenAI API",
            "models": {
                "gpt4o": {
                    "path": "gpt-4o",
                    "tasks": ["reasoning", "analysis", "code_debug"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "gpt4_turbo": {
                    "path": "gpt-4-turbo",
                    "tasks": ["reasoning", "analysis"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "gpt3_5_turbo": {
                    "path": "gpt-3.5-turbo",
                    "tasks": ["text_generation", "summarization"],
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            }
        },
        "anthropic": {
            "name": "Anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "description": "Anthropic Claude API",
            "models": {
                "claude_3_5_sonnet": {
                    "path": "claude-3-5-sonnet-20241022",
                    "tasks": ["reasoning", "analysis", "writing"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "claude_3_haiku": {
                    "path": "claude-3-haiku-20240307",
                    "tasks": ["text_generation", "summarization"],
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            }
        },
        "deepseek": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "description": "DeepSeek API service",
            "models": {
                "deepseek_v3": {
                    "path": "deepseek-v3",
                    "tasks": ["reasoning", "math", "code_debug"],
                    "max_tokens": 8192,
                    "temperature": 0.7
                },
                "deepseek_coder": {
                    "path": "deepseek-coder",
                    "tasks": ["code_debug", "code_generation"],
                    "max_tokens": 8192,
                    "temperature": 0.3
                }
            }
        }
    }
    
    return presets

def create_model_from_preset(preset_name: str, model_name: str, 
                           api_key: str, custom_config: Dict = None) -> ModelConfig:
    """Create a model configuration from preset"""
    
    presets = create_preset_configs()
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = presets[preset_name]
    
    if model_name not in preset["models"]:
        raise ValueError(f"Model {model_name} not found in preset {preset_name}")
    
    model_info = preset["models"][model_name]
    
    # Merge custom config if provided
    if custom_config:
        model_info.update(custom_config)
    
    return ModelConfig(
        name=model_name,
        model_path=model_info["path"],
        deployment_type="openai_api",
        api_base_url=preset["base_url"],
        api_key=api_key,
        supported_tasks=model_info.get("tasks", ["text_generation"]),
        max_tokens=model_info.get("max_tokens", 4096),
        temperature=model_info.get("temperature", 0.7)
    )

def validate_model_config(config: ModelConfig) -> bool:
    """Validate model configuration"""
    
    required_fields = ["name", "model_path", "deployment_type"]
    
    for field in required_fields:
        if not getattr(config, field):
            raise ValueError(f"Missing required field: {field}")
    
    if config.deployment_type == "openai_api":
        if not config.api_base_url or not config.api_key:
            raise ValueError("OpenAI API deployment requires api_base_url and api_key")
    
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    if not 0 <= config.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    
    return True