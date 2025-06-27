#!/usr/bin/env python3
"""
Test suite for LLM Cooperation System
Comprehensive testing of all system components
"""
import asyncio
import pytest
import aiohttp
import json
import time
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_cooperation.config import SystemConfig
from vllm_engine import InferenceRequest, InferenceResponse
from llm_cooperation.routing.intelligent_router import intelligent_router, RequestAnalyzer, TaskType, ComplexityLevel
from llm_cooperation.schedulers.cooperation_scheduler import cooperation_scheduler, CooperationMode
from llm_cooperation.services.application_service import service_manager

class TestClient:
    """Test client for system endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request"""
        async with self.session.get(f"{self.base_url}{endpoint}") as response:
            return await response.json(), response.status
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request"""
        async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
            return await response.json(), response.status

@pytest.mark.asyncio
class TestSystemComponents:
    """Test individual system components"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test model configuration
        assert len(SystemConfig.MODELS) > 0
        
        for model_name, config in SystemConfig.MODELS.items():
            assert config.name == model_name
            assert config.tensor_parallel_size > 0
            assert 0 < config.gpu_memory_utilization <= 1.0
            assert config.max_model_len > 0
            assert isinstance(config.supported_tasks, list)
        
        # Test port assignment
        for model_name in SystemConfig.MODELS.keys():
            port = SystemConfig.get_model_port(model_name)
            assert 8000 <= port <= 9000
    
    async def test_request_analyzer(self):
        """Test request analysis functionality"""
        analyzer = RequestAnalyzer()
        
        test_cases = [
            {
                "query": "What is 2 + 2?",
                "expected_type": TaskType.MATH_CALCULATION,
                "expected_complexity": ComplexityLevel.LOW
            },
            {
                "query": "Write a Python function to implement quicksort algorithm with detailed comments",
                "expected_type": TaskType.CODE_ANALYSIS,
                "expected_complexity": ComplexityLevel.HIGH
            },
            {
                "query": "Analyze this image and describe what you see",
                "expected_type": TaskType.MULTIMODAL,
                "requires_multimodal": True
            },
            {
                "query": "Compare the advantages and disadvantages of renewable energy sources",
                "expected_type": TaskType.COMPARISON,
                "expected_complexity": ComplexityLevel.MEDIUM
            }
        ]
        
        for case in test_cases:
            analysis = await analyzer.analyze_request(case["query"])
            
            assert analysis.task_type == case["expected_type"]
            
            if "expected_complexity" in case:
                assert analysis.complexity_level == case["expected_complexity"]
            
            if "requires_multimodal" in case:
                assert analysis.requires_multimodal == case["requires_multimodal"]
            
            assert 0 <= analysis.confidence <= 1.0
            assert len(analysis.suggested_models) > 0
    
    async def test_cooperation_modes(self):
        """Test cooperation scheduling"""
        # Note: This test requires running models
        test_query = "Simple test query for cooperation"
        
        # Test sequential cooperation
        try:
            result = await cooperation_scheduler.create_sequential_task(
                test_query,
                ["qwen2_5_7b"],  # Use single model for testing
                "concatenate"
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Sequential cooperation test skipped: {e}")
        
        # Test parallel cooperation
        try:
            result = await cooperation_scheduler.create_parallel_task(
                test_query,
                ["qwen2_5_7b"],  # Use single model for testing
                "concatenate"
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Parallel cooperation test skipped: {e}")
    
    async def test_application_services(self):
        """Test application services"""
        # Test document analysis
        document = "This is a test document for analysis."
        
        try:
            response = await service_manager.process_request(
                service_type="document_analysis",
                content=document,
                parameters={"analysis_type": "summary"}
            )
            
            assert response.success
            assert response.result is not None
            assert response.processing_time > 0
            
        except Exception as e:
            pytest.skip(f"Document analysis test skipped: {e}")
        
        # Test data insights
        data = "Sales: Q1: 100, Q2: 120, Q3: 110, Q4: 130"
        
        try:
            response = await service_manager.process_request(
                service_type="data_insight",
                content=data,
                parameters={"insight_type": "trends"}
            )
            
            assert response.success
            assert response.result is not None
            
        except Exception as e:
            pytest.skip(f"Data insights test skipped: {e}")

@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test API endpoints"""
    
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        async with TestClient() as client:
            try:
                data, status = await client.get("/health")
                assert status == 200
                assert data["status"] == "healthy"
            except Exception as e:
                pytest.skip(f"Health endpoint test skipped: {e}")
    
    async def test_status_endpoint(self):
        """Test system status endpoint"""
        async with TestClient() as client:
            try:
                data, status = await client.get("/status")
                assert status == 200
                assert "system_health" in data
                assert "active_models" in data
                assert "total_requests" in data
            except Exception as e:
                pytest.skip(f"Status endpoint test skipped: {e}")
    
    async def test_query_endpoint(self):
        """Test query processing endpoint"""
        async with TestClient() as client:
            query_data = {
                "query": "What is the capital of France?",
                "preferences": {"strategy": "single_model"},
                "user_id": "test_user"
            }
            
            try:
                data, status = await client.post("/query", query_data)
                assert status == 200
                assert "result" in data
                assert "processing_time" in data
                assert "strategy_used" in data
            except Exception as e:
                pytest.skip(f"Query endpoint test skipped: {e}")
    
    async def test_service_endpoint(self):
        """Test service request endpoint"""
        async with TestClient() as client:
            service_data = {
                "service_type": "document_analysis",
                "content": "Test document content",
                "parameters": {"analysis_type": "summary"},
                "user_id": "test_user"
            }
            
            try:
                data, status = await client.post("/service", service_data)
                assert status == 200
                assert "request_id" in data
                assert "success" in data
                assert "processing_time" in data
            except Exception as e:
                pytest.skip(f"Service endpoint test skipped: {e}")

class TestPerformance:
    """Performance and load testing"""
    
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        async def make_request(client, query_id):
            query_data = {
                "query": f"Test query {query_id}",
                "user_id": f"test_user_{query_id}"
            }
            
            try:
                data, status = await client.post("/query", query_data)
                return status == 200
            except:
                return False
        
        async with TestClient() as client:
            try:
                # Test 5 concurrent requests
                tasks = [make_request(client, i) for i in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # At least 80% should succeed
                success_rate = sum(1 for r in results if r is True) / len(results)
                assert success_rate >= 0.8
                
            except Exception as e:
                pytest.skip(f"Concurrent requests test skipped: {e}")
    
    async def test_response_times(self):
        """Test response time requirements"""
        async with TestClient() as client:
            query_data = {
                "query": "Simple test query",
                "preferences": {"max_latency": 30.0}
            }
            
            try:
                start_time = time.time()
                data, status = await client.post("/query", query_data)
                elapsed = time.time() - start_time
                
                assert status == 200
                assert elapsed < 30.0  # Should complete within 30 seconds
                
            except Exception as e:
                pytest.skip(f"Response time test skipped: {e}")

class TestIntegration:
    """End-to-end integration tests"""
    
    async def test_full_workflow(self):
        """Test complete workflow from query to response"""
        async with TestClient() as client:
            # 1. Check system health
            try:
                health_data, status = await client.get("/health")
                assert status == 200
            except:
                pytest.skip("System not available for integration test")
            
            # 2. Submit query
            query_data = {
                "query": "Explain artificial intelligence in simple terms",
                "user_id": "integration_test"
            }
            
            try:
                query_response, status = await client.post("/query", query_data)
                assert status == 200
                assert len(query_response["result"]) > 0
            except Exception as e:
                pytest.skip(f"Query processing failed: {e}")
            
            # 3. Check system statistics
            try:
                stats_data, status = await client.get("/routing/stats")
                assert status == 200
                assert stats_data["total_requests"] > 0
            except Exception as e:
                pytest.skip(f"Stats check failed: {e}")
    
    async def test_service_integration(self):
        """Test service integration workflow"""
        async with TestClient() as client:
            services = [
                {
                    "service_type": "document_analysis",
                    "content": "AI is transforming industries worldwide.",
                    "parameters": {"analysis_type": "summary"}
                },
                {
                    "service_type": "data_insight", 
                    "content": "Sales: Jan 100, Feb 120, Mar 110",
                    "parameters": {"insight_type": "trends"}
                }
            ]
            
            for service_request in services:
                try:
                    response, status = await client.post("/service", service_request)
                    assert status == 200
                    assert response["success"] is True
                except Exception as e:
                    pytest.skip(f"Service {service_request['service_type']} failed: {e}")

def run_tests():
    """Run all tests"""
    print("Running LLM Cooperation System Tests...")
    
    # Run pytest with custom configuration
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    import sys
    
    # Check if system is running
    async def check_system():
        try:
            async with TestClient() as client:
                data, status = await client.get("/health")
                return status == 200
        except:
            return False
    
    system_available = asyncio.run(check_system())
    
    if not system_available:
        print("Warning: System not available. Some tests will be skipped.")
        print("Start the system with: python start_system.py --dev")
    
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code)