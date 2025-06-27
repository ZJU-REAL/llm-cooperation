#!/usr/bin/env python3
"""
API Configuration Utility
Allows dynamic configuration of OpenAI API endpoints and models
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

from llm_cooperation.config import SystemConfig, ModelConfig
from llm_cooperation.engines.openai_engine import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIConfigManager:
    """Manages API configurations and model setups"""
    
    def __init__(self):
        self.config_file = Path("api_configs.json")
    
    def load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        return {}
    
    def save_config_file(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
    
    def update_global_api_config(self, base_url: str, api_key: str):
        """Update global API configuration"""
        logger.info(f"Updating global API config to {base_url}")
        
        # Update system configuration
        SystemConfig.update_api_config(base_url, api_key)
        
        # Save to file
        config = self.load_config_file()
        config.update({
            "global": {
                "base_url": base_url,
                "api_key": api_key
            }
        })
        self.save_config_file(config)
        
        logger.info("Global API configuration updated successfully")
    
    def add_model_config(self, model_name: str, model_path: str,
                        api_base_url: str = None, api_key: str = None,
                        supported_tasks: List[str] = None,
                        max_tokens: int = 4096, temperature: float = 0.7,
                        **kwargs):
        """Add or update a model configuration"""
        
        # Use global config if not specified
        if api_base_url is None:
            api_base_url = SystemConfig.DEFAULT_OPENAI_CONFIG.base_url
        if api_key is None:
            api_key = SystemConfig.DEFAULT_OPENAI_CONFIG.api_key
        
        logger.info(f"Adding model {model_name} with path {model_path}")
        
        # Add to system config
        SystemConfig.add_custom_model(
            model_name=model_name,
            model_path=model_path,
            api_base_url=api_base_url,
            api_key=api_key,
            supported_tasks=supported_tasks or ["text_generation"],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Save to file
        config = self.load_config_file()
        if "models" not in config:
            config["models"] = {}
        
        config["models"][model_name] = {
            "model_path": model_path,
            "api_base_url": api_base_url,
            "api_key": api_key,
            "supported_tasks": supported_tasks or ["text_generation"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        self.save_config_file(config)
        logger.info(f"Model {model_name} configuration saved")
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all configured models"""
        models = {}
        
        for model_name, config in SystemConfig.MODELS.items():
            models[model_name] = {
                "model_path": config.model_path,
                "deployment_type": config.deployment_type,
                "api_base_url": config.api_base_url,
                "api_key": config.api_key[:10] + "..." if config.api_key else None,
                "supported_tasks": config.supported_tasks,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature
            }
        
        return models
    
    async def test_model_connectivity(self, model_name: str = None) -> Dict[str, Any]:
        """Test connectivity for one or all models"""
        if model_name:
            models_to_test = [model_name]
        else:
            models_to_test = list(SystemConfig.MODELS.keys())
        
        results = {}
        
        for model in models_to_test:
            try:
                # Initialize engine if not already done
                if not hasattr(engine, 'clients') or not engine.clients:
                    await engine.initialize()
                
                # Test model
                result = await engine._test_model_connectivity(model)
                results[model] = {
                    "status": "success" if result else "failed",
                    "accessible": result
                }
                
            except Exception as e:
                results[model] = {
                    "status": "error",
                    "error": str(e),
                    "accessible": False
                }
        
        return results
    
    def create_preset_configs(self):
        """Create preset configurations for common providers"""
        presets = {
            "aigcbest": {
                "name": "AIGC Best",
                "base_url": "https://api2.aigcbest.top/v1",
                "models": {
                    "qwen3_32b": "Qwen/Qwen3-32B",
                    "qwen3_8b": "Qwen/Qwen3-8B", 
                    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
                    "qwen2_5_vl_72b": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "claude_3_5_sonnet": "anthropic/claude-3-5-sonnet-20241022",
                    "gpt4o": "openai/gpt-4o",
                    "deepseek_v3": "deepseek/deepseek-v3"
                }
            },
            "openai": {
                "name": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "models": {
                    "gpt4o": "gpt-4o",
                    "gpt4_turbo": "gpt-4-turbo",
                    "gpt3_5_turbo": "gpt-3.5-turbo"
                }
            },
            "anthropic": {
                "name": "Anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "models": {
                    "claude_3_5_sonnet": "claude-3-5-sonnet-20241022",
                    "claude_3_haiku": "claude-3-haiku-20240307"
                }
            }
        }
        
        return presets
    
    def apply_preset(self, preset_name: str, api_key: str):
        """Apply a preset configuration"""
        presets = self.create_preset_configs()
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset = presets[preset_name]
        base_url = preset["base_url"]
        
        logger.info(f"Applying preset: {preset['name']}")
        
        # Update global config
        self.update_global_api_config(base_url, api_key)
        
        # Add models from preset
        for model_name, model_path in preset["models"].items():
            # Define supported tasks based on model type
            if "vl" in model_name.lower() or "vision" in model_name.lower():
                supported_tasks = ["vision", "multimodal", "image_analysis"]
            elif "32b" in model_name or "claude" in model_name or "gpt4" in model_name:
                supported_tasks = ["reasoning", "math", "logic", "code_debug", "complex_reasoning"]
            else:
                supported_tasks = ["text_generation", "summarization", "translation"]
            
            self.add_model_config(
                model_name=model_name,
                model_path=model_path,
                api_base_url=base_url,
                api_key=api_key,
                supported_tasks=supported_tasks
            )
        
        logger.info(f"Applied preset {preset_name} with {len(preset['models'])} models")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="API Configuration Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Update global config
    global_parser = subparsers.add_parser("global", help="Update global API configuration")
    global_parser.add_argument("--base-url", required=True, help="API base URL")
    global_parser.add_argument("--api-key", required=True, help="API key")
    
    # Add model
    model_parser = subparsers.add_parser("add-model", help="Add model configuration")
    model_parser.add_argument("--name", required=True, help="Model name")
    model_parser.add_argument("--path", required=True, help="Model path/identifier")
    model_parser.add_argument("--base-url", help="API base URL (optional)")
    model_parser.add_argument("--api-key", help="API key (optional)")
    model_parser.add_argument("--tasks", nargs="+", help="Supported tasks")
    model_parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    model_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    
    # List models
    subparsers.add_parser("list", help="List all configured models")
    
    # Test connectivity
    test_parser = subparsers.add_parser("test", help="Test model connectivity")
    test_parser.add_argument("--model", help="Specific model to test (optional)")
    
    # Apply preset
    preset_parser = subparsers.add_parser("preset", help="Apply preset configuration")
    preset_parser.add_argument("--name", required=True, 
                              choices=["aigcbest", "openai", "anthropic"],
                              help="Preset name")
    preset_parser.add_argument("--api-key", required=True, help="API key")
    
    # List presets
    subparsers.add_parser("presets", help="List available presets")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config_manager = APIConfigManager()
    
    try:
        if args.command == "global":
            config_manager.update_global_api_config(args.base_url, args.api_key)
            
        elif args.command == "add-model":
            config_manager.add_model_config(
                model_name=args.name,
                model_path=args.path,
                api_base_url=args.base_url,
                api_key=args.api_key,
                supported_tasks=args.tasks,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
        elif args.command == "list":
            models = config_manager.list_models()
            print("\nConfigured Models:")
            print("=" * 50)
            for name, config in models.items():
                print(f"\nModel: {name}")
                print(f"  Path: {config['model_path']}")
                print(f"  API: {config['api_base_url']}")
                print(f"  Tasks: {config['supported_tasks']}")
                print(f"  Max Tokens: {config['max_tokens']}")
                print(f"  Temperature: {config['temperature']}")
            
        elif args.command == "test":
            print("Testing model connectivity...")
            results = await config_manager.test_model_connectivity(args.model)
            
            print("\nConnectivity Test Results:")
            print("=" * 50)
            for model, result in results.items():
                status = "✓" if result['accessible'] else "✗"
                print(f"{status} {model}: {result['status']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
            
        elif args.command == "preset":
            config_manager.apply_preset(args.name, args.api_key)
            
        elif args.command == "presets":
            presets = config_manager.create_preset_configs()
            print("\nAvailable Presets:")
            print("=" * 50)
            for name, preset in presets.items():
                print(f"\n{name}: {preset['name']}")
                print(f"  URL: {preset['base_url']}")
                print(f"  Models: {list(preset['models'].keys())}")
        
    except Exception as e:
        logger.error(f"Command failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())