#!/usr/bin/env python3
"""
Basic usage examples for LLM Cooperation System
"""
import asyncio
import os
from llm_cooperation import (
    SystemConfig, 
    OpenAIEngine, 
    IntelligentRouter,
    CooperationScheduler,
    ApplicationServiceManager
)

async def basic_inference_example():
    """Example of basic inference with a single model"""
    print("🔹 Basic Inference Example")
    print("-" * 40)
    
    # Initialize configuration
    config = SystemConfig()
    config.update_api_config(
        base_url=os.getenv("BASE_URL", "https://api2.aigcbest.top/v1"),
        api_key=os.getenv("API_KEY", "your-api-key")
    )
    
    # Create and initialize engine
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    try:
        # Create inference request
        from llm_cooperation.engines import InferenceRequest
        request = InferenceRequest(
            prompt="What is artificial intelligence?",
            model_name="qwen3_8b",
            max_tokens=200
        )
        
        # Perform inference
        response = await engine.inference(request)
        
        if response.success:
            print(f"Model: {response.model_name}")
            print(f"Response: {response.text[:200]}...")
            print(f"Latency: {response.latency:.2f}s")
            print(f"Tokens: {response.usage.get('total_tokens', 'N/A')}")
        else:
            print(f"Error: {response.error}")
    
    finally:
        await engine.shutdown()

async def intelligent_routing_example():
    """Example of intelligent routing"""
    print("\n🔹 Intelligent Routing Example")
    print("-" * 40)
    
    # Initialize system
    config = SystemConfig()
    engine = OpenAIEngine(config)
    router = IntelligentRouter()
    
    await engine.initialize()
    
    try:
        # Different types of queries
        queries = [
            "Solve: 2x² + 5x - 3 = 0",  # Math - should use reasoning model
            "Translate 'hello' to Spanish",  # Simple - should use lightweight model
            "Explain quantum computing in detail"  # Complex - might use cooperation
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            
            # Route query automatically
            result = await router.route_request(query)
            
            print(f"Result: {result[:150]}...")
    
    finally:
        await engine.shutdown()

async def cooperation_example():
    """Example of multi-model cooperation"""
    print("\n🔹 Multi-Model Cooperation Example")
    print("-" * 40)
    
    # Initialize system
    config = SystemConfig()
    engine = OpenAIEngine(config)
    scheduler = CooperationScheduler()
    
    await engine.initialize()
    
    try:
        complex_query = """
        Analyze the environmental and economic impacts of electric vehicles.
        Consider manufacturing, operation, and end-of-life phases.
        """
        
        print(f"Complex Query: {complex_query[:100]}...")
        
        # Sequential cooperation
        print("\n📋 Sequential Cooperation:")
        result = await scheduler.create_sequential_task(
            query=complex_query,
            models=["qwen3_32b", "qwen3_8b"],
            integration_strategy="ensemble"
        )
        print(f"Result: {result[:200]}...")
        
        # Parallel cooperation
        print("\n📋 Parallel Cooperation:")
        result = await scheduler.create_parallel_task(
            query=complex_query,
            models=["qwen3_32b", "qwen3_8b"],
            integration_strategy="ensemble"
        )
        print(f"Result: {result[:200]}...")
    
    finally:
        await engine.shutdown()

async def application_services_example():
    """Example of application services"""
    print("\n🔹 Application Services Example")
    print("-" * 40)
    
    # Initialize system
    config = SystemConfig()
    engine = OpenAIEngine(config) 
    service_manager = ApplicationServiceManager()
    
    await engine.initialize()
    
    try:
        # Document analysis
        document = """
        Q3 2024 Financial Report Summary:
        Revenue increased 25% to $5.2M. Customer growth of 30% with 1,200 new customers.
        Key challenges: supply chain delays and increased competition.
        """
        
        print("📄 Document Analysis:")
        response = await service_manager.process_request(
            service_type="document_analysis",
            content=document,
            parameters={"analysis_type": "summary"}
        )
        
        if response.success:
            print(f"Analysis: {response.result['analysis'][:200]}...")
            print(f"Processing time: {response.processing_time:.2f}s")
        
        # Data insights
        data = "Sales: Jan 100K, Feb 120K, Mar 110K, Apr 130K"
        
        print("\n📊 Data Insights:")
        response = await service_manager.process_request(
            service_type="data_insight",
            content=data,
            parameters={"insight_type": "trends"}
        )
        
        if response.success:
            print(f"Insights: {response.result['trend_analysis'][:200]}...")
    
    finally:
        await engine.shutdown()

async def configuration_example():
    """Example of dynamic configuration"""
    print("\n🔹 Configuration Example")
    print("-" * 40)
    
    from llm_cooperation.tools import APIConfigManager
    
    config_manager = APIConfigManager()
    
    # Show current models
    print("Current Models:")
    models = config_manager.list_models()
    for name, info in models.items():
        print(f"  • {name}: {info['model_path']}")
    
    # Add custom model
    print("\nAdding custom model...")
    config_manager.add_model_config(
        model_name="custom_test",
        model_path="test/custom-model",
        supported_tasks=["test", "demo"],
        max_tokens=2048
    )
    
    # Test connectivity (will fail without real API)
    print("\nTesting connectivity...")
    try:
        results = await config_manager.test_model_connectivity("qwen3_8b")
        for model, result in results.items():
            status = "✅" if result['accessible'] else "❌"
            print(f"  {status} {model}")
    except Exception as e:
        print(f"  Connection test failed: {e}")
    
    # Remove test model
    config_manager.remove_model("custom_test")
    print("Custom model removed")

async def error_handling_example():
    """Example of error handling"""
    print("\n🔹 Error Handling Example")
    print("-" * 40)
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    
    try:
        await engine.initialize()
        
        # Test with invalid model
        from llm_cooperation.engines import InferenceRequest
        request = InferenceRequest(
            prompt="Test",
            model_name="nonexistent_model"
        )
        
        response = await engine.inference(request)
        
        if not response.success:
            print(f"Expected error handled: {response.error}")
        
        # Test health check
        health = await engine.health_check()
        print(f"Engine health: {health['status']}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        await engine.shutdown()

async def main():
    """Run all examples"""
    print("🚀 LLM Cooperation System - Basic Usage Examples")
    print("=" * 60)
    
    try:
        await basic_inference_example()
        await intelligent_routing_example() 
        await cooperation_example()
        await application_services_example()
        await configuration_example()
        await error_handling_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())