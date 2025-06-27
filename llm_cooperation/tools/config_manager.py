"""
API Configuration Manager
Handles dynamic configuration of OpenAI API endpoints and models
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..config import SystemConfig, ModelConfig, create_preset_configs

logger = logging.getLogger(__name__)

class APIConfigManager:
    """Manages API configurations and model setups"""
    
    def __init__(self, config_file: str = "api_configs.json"):
        self.config_file = Path(config_file)
        self.system_config = SystemConfig()
    
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
        self.system_config.update_api_config(base_url, api_key)
        
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
            api_base_url = self.system_config.DEFAULT_OPENAI_CONFIG.base_url
        if api_key is None:
            api_key = self.system_config.DEFAULT_OPENAI_CONFIG.api_key
        
        logger.info(f"Adding model {model_name} with path {model_path}")
        
        # Add to system config
        self.system_config.add_custom_model(
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
        
        for model_name, config in self.system_config.MODELS.items():
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
        from ..engines.openai_engine import OpenAIEngine
        
        if model_name:
            models_to_test = [model_name]
        else:
            models_to_test = list(self.system_config.MODELS.keys())
        
        results = {}
        engine = OpenAIEngine(self.system_config)
        
        try:
            await engine.initialize()
            
            for model in models_to_test:
                try:
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
        finally:
            await engine.shutdown()
        
        return results
    
    def create_preset_configs(self):
        """Create preset configurations for common providers"""
        return create_preset_configs()
    
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
        for model_name, model_info in preset["models"].items():
            self.add_model_config(
                model_name=model_name,
                model_path=model_info["path"],
                api_base_url=base_url,
                api_key=api_key,
                supported_tasks=model_info.get("tasks", ["text_generation"]),
                max_tokens=model_info.get("max_tokens", 4096),
                temperature=model_info.get("temperature", 0.7)
            )
        
        logger.info(f"Applied preset {preset_name} with {len(preset['models'])} models")
    
    def remove_model(self, model_name: str):
        """Remove a model configuration"""
        # Remove from system config
        self.system_config.remove_model(model_name)
        
        # Remove from file
        config = self.load_config_file()
        if "models" in config and model_name in config["models"]:
            del config["models"][model_name]
            self.save_config_file(config)
        
        logger.info(f"Removed model {model_name}")
    
    def export_config(self, output_file: str = None) -> Dict[str, Any]:
        """Export current configuration"""
        config_data = self.system_config.to_dict()
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration exported to {output_file}")
        
        return config_data
    
    def import_config(self, config_file: str):
        """Import configuration from file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update global config if present
        if "default_openai_config" in config_data:
            openai_config = config_data["default_openai_config"]
            self.update_global_api_config(
                openai_config["base_url"],
                openai_config.get("api_key", "")
            )
        
        # Add models
        if "models" in config_data:
            for model_name, model_info in config_data["models"].items():
                self.add_model_config(
                    model_name=model_name,
                    **model_info
                )
        
        logger.info(f"Configuration imported from {config_file}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        from ..config.model_configs import validate_model_config
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate each model
        for model_name, config in self.system_config.MODELS.items():
            try:
                validate_model_config(config)
                validation_results["warnings"].append(f"Model {model_name}: OK")
            except ValueError as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Model {model_name}: {e}")
        
        # Check API keys
        if not self.system_config.DEFAULT_OPENAI_CONFIG.api_key:
            validation_results["warnings"].append("No global API key configured")
        
        return validation_results