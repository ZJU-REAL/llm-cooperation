"""
Tests for configuration module
"""
import pytest
from llm_cooperation.config import SystemConfig, ModelConfig, OpenAIAPIConfig

class TestSystemConfig:
    
    def test_initialization(self):
        """Test SystemConfig initialization"""
        config = SystemConfig()
        assert len(config.MODELS) > 0
        assert config.DEFAULT_OPENAI_CONFIG is not None
        assert isinstance(config.ROUTING_CONFIG, dict)
    
    def test_model_management(self):
        """Test adding and removing models"""
        config = SystemConfig()
        initial_count = len(config.MODELS)
        
        # Add custom model
        config.add_custom_model(
            model_name="test_model",
            model_path="test/model",
            supported_tasks=["test"]
        )
        
        assert len(config.MODELS) == initial_count + 1
        assert "test_model" in config.MODELS
        
        # Remove model
        config.remove_model("test_model")
        assert len(config.MODELS) == initial_count
        assert "test_model" not in config.MODELS
    
    def test_model_config_validation(self):
        """Test ModelConfig validation"""
        from llm_cooperation.config.model_configs import validate_model_config
        
        # Valid config
        config = ModelConfig(
            name="test",
            model_path="test/path",
            api_base_url="https://api.test.com",
            api_key="test-key"
        )
        assert validate_model_config(config) is True
        
        # Invalid config - missing required fields
        with pytest.raises(ValueError):
            invalid_config = ModelConfig(
                name="",
                model_path="test/path"
            )
            validate_model_config(invalid_config)

class TestModelConfig:
    
    def test_model_config_creation(self):
        """Test ModelConfig creation"""
        config = ModelConfig(
            name="test_model",
            model_path="test/model/path",
            api_base_url="https://api.test.com",
            api_key="test-key"
        )
        
        assert config.name == "test_model"
        assert config.model_path == "test/model/path"
        assert config.deployment_type == "openai_api"
        assert config.supported_tasks == []
    
    def test_model_config_post_init(self):
        """Test ModelConfig post-initialization"""
        config = ModelConfig(
            name="test",
            model_path="test/path"
        )
        
        # Check default values
        assert config.supported_tasks == []
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

class TestOpenAIAPIConfig:
    
    def test_api_config_creation(self):
        """Test OpenAIAPIConfig creation"""
        config = OpenAIAPIConfig(
            base_url="https://api.test.com",
            api_key="test-key"
        )
        
        assert config.base_url == "https://api.test.com"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_retries == 3