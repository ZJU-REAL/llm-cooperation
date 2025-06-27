"""
Tests for inference engines
"""
import pytest
from unittest.mock import AsyncMock, patch
from llm_cooperation.engines import OpenAIEngine, InferenceRequest, InferenceResponse
from llm_cooperation.config import SystemConfig

class TestInferenceRequest:
    
    def test_request_creation(self):
        """Test InferenceRequest creation"""
        request = InferenceRequest(
            prompt="Test prompt",
            model_name="test_model"
        )
        
        assert request.prompt == "Test prompt"
        assert request.model_name == "test_model"
        assert request.max_tokens == 1024
        assert request.extra_params == {}

class TestInferenceResponse:
    
    def test_response_creation(self):
        """Test InferenceResponse creation"""
        response = InferenceResponse(
            text="Test response",
            model_name="test_model",
            usage={"total_tokens": 100},
            latency=1.5
        )
        
        assert response.text == "Test response"
        assert response.model_name == "test_model"
        assert response.success is True
        assert response.metadata == {}

@pytest.mark.asyncio
class TestOpenAIEngine:
    
    @pytest.fixture
    def mock_config(self):
        """Mock SystemConfig for testing"""
        config = SystemConfig()
        # Override with test configuration
        config._models = {
            "test_model": config._models["qwen3_8b"]  # Use existing model config
        }
        return config
    
    @pytest.fixture
    def engine(self, mock_config):
        """Create OpenAIEngine instance for testing"""
        return OpenAIEngine(mock_config)
    
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert not engine.initialized
        assert engine.name == "OpenAIEngine"
    
    async def test_get_engine_info(self, engine):
        """Test engine info retrieval"""
        info = engine.get_engine_info()
        
        assert "name" in info
        assert "initialized" in info
        assert "type" in info
        assert info["name"] == "OpenAIEngine"
    
    @patch('llm_cooperation.engines.openai_engine.AsyncOpenAI')
    async def test_inference_success(self, mock_openai, engine):
        """Test successful inference"""
        # Mock OpenAI client response
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Add mock client to engine
        engine.clients["test_model"] = mock_client
        
        # Test inference
        request = InferenceRequest(
            prompt="Test prompt",
            model_name="test_model"
        )
        
        response = await engine.inference(request)
        
        assert response.success is True
        assert response.text == "Test response"
        assert response.model_name == "test_model"
        assert response.usage["total_tokens"] == 30
    
    async def test_inference_failure(self, engine):
        """Test inference failure handling"""
        request = InferenceRequest(
            prompt="Test prompt",
            model_name="nonexistent_model"
        )
        
        response = await engine.inference(request)
        
        assert response.success is False
        assert response.error is not None
        assert "not configured" in response.error
    
    async def test_get_model_info(self, engine, mock_config):
        """Test model info retrieval"""
        info = await engine.get_model_info("test_model")
        
        assert "name" in info
        assert "model_path" in info
        assert "status" in info
        assert info["name"] == "test_model"
    
    async def test_get_model_info_not_found(self, engine):
        """Test model info for non-existent model"""
        info = await engine.get_model_info("nonexistent_model")
        
        assert "error" in info
        assert "not found" in info["error"]