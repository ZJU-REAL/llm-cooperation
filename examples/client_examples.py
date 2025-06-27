#!/usr/bin/env python3
"""
Example client for LLM Cooperation System
Demonstrates various usage patterns and capabilities
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
import argparse

class LLMCooperationClient:
    """Client for interacting with LLM Cooperation System"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send query to the system"""
        payload = {
            "query": query,
            "preferences": preferences or {},
            "user_id": "example_client"
        }
        
        async with self.session.post(f"{self.base_url}/query", json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Query failed: {response.status} - {error_text}")
    
    async def service_request(self, service_type: str, content: str, 
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send service request"""
        payload = {
            "service_type": service_type,
            "content": content,
            "parameters": parameters or {},
            "user_id": "example_client"
        }
        
        async with self.session.post(f"{self.base_url}/service", json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Service request failed: {response.status} - {error_text}")
    
    async def cooperation_task(self, query: str, mode: str = "sequential",
                             models: List[str] = None, 
                             integration_strategy: str = "concatenate") -> Dict[str, Any]:
        """Create cooperation task"""
        params = {
            "query": query,
            "mode": mode,
            "integration_strategy": integration_strategy
        }
        
        if models:
            params["models"] = models
        
        async with self.session.post(f"{self.base_url}/cooperation/task", params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Cooperation task failed: {response.status} - {error_text}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        async with self.session.get(f"{self.base_url}/status") as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Status check failed: {response.status} - {error_text}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {}
        
        endpoints = [
            ("routing", "/routing/stats"),
            ("services", "/services/stats"),
            ("cooperation", "/cooperation/stats"),
            ("models", "/models")
        ]
        
        for name, endpoint in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        stats[name] = await response.json()
                    else:
                        stats[name] = {"error": f"HTTP {response.status}"}
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats

async def example_simple_queries():
    """Example: Simple query processing"""
    print("\n=== Simple Query Examples ===")
    
    async with LLMCooperationClient() as client:
        examples = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Write a Python function to calculate fibonacci numbers",
            "Solve this math problem: 2x + 5 = 15"
        ]
        
        for query in examples:
            print(f"\nQuery: {query}")
            try:
                start_time = time.time()
                result = await client.query(query)
                elapsed = time.time() - start_time
                
                print(f"Result: {result['result'][:200]}...")
                print(f"Strategy: {result['strategy_used']}")
                print(f"Time: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")

async def example_cooperation_modes():
    """Example: Different cooperation modes"""
    print("\n=== Cooperation Mode Examples ===")
    
    async with LLMCooperationClient() as client:
        complex_query = """
        Analyze the pros and cons of renewable energy adoption, considering:
        1. Economic factors
        2. Environmental impact
        3. Technical challenges
        4. Policy implications
        
        Provide a comprehensive analysis with recommendations.
        """
        
        modes = ["sequential", "parallel", "voting"]
        
        for mode in modes:
            print(f"\n--- {mode.upper()} Mode ---")
            try:
                start_time = time.time()
                result = await client.cooperation_task(
                    query=complex_query,
                    mode=mode,
                    models=["qwen3_32b_reasoning", "qwen2_5_7b"]
                )
                elapsed = time.time() - start_time
                
                print(f"Result: {result['result'][:300]}...")
                print(f"Models: {result['models']}")
                print(f"Time: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")

async def example_application_services():
    """Example: Application service usage"""
    print("\n=== Application Service Examples ===")
    
    async with LLMCooperationClient() as client:
        
        # Document Analysis
        print("\n--- Document Analysis ---")
        document = """
        Artificial Intelligence (AI) has transformed numerous industries in recent years. 
        From healthcare diagnostics to autonomous vehicles, AI applications continue to expand. 
        However, challenges remain in areas such as ethics, bias, and interpretability. 
        Organizations must balance innovation with responsible AI practices.
        """
        
        try:
            result = await client.service_request(
                service_type="document_analysis",
                content=document,
                parameters={"analysis_type": "comprehensive"}
            )
            print(f"Analysis: {result['result']['analysis'][:300]}...")
            print(f"Processing time: {result['processing_time']:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
        
        # Data Insights
        print("\n--- Data Insights ---")
        data = """
        Sales Data Q1-Q4:
        Q1: 100000, Q2: 120000, Q3: 135000, Q4: 150000
        Customer Satisfaction: Q1: 85%, Q2: 87%, Q3: 89%, Q4: 91%
        Market Share: Q1: 15%, Q2: 16%, Q3: 17%, Q4: 18%
        """
        
        try:
            result = await client.service_request(
                service_type="data_insight",
                content=data,
                parameters={"insight_type": "trends"}
            )
            print(f"Insights: {result['result']['trend_analysis'][:300]}...")
            print(f"Processing time: {result['processing_time']:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
        
        # Decision Support
        print("\n--- Decision Support ---")
        decision_problem = """
        We need to choose between three cloud providers for our infrastructure:
        - Provider A: Lower cost, limited features
        - Provider B: Medium cost, good features, reliable
        - Provider C: Higher cost, extensive features, premium support
        
        Budget: $50k annually, Priority: Reliability and scalability
        """
        
        try:
            result = await client.service_request(
                service_type="decision_support",
                content=decision_problem,
                parameters={"decision_type": "comparison"}
            )
            print(f"Recommendation: {result['result']['comparison'][:300]}...")
            print(f"Processing time: {result['processing_time']:.2f}s")
        except Exception as e:
            print(f"Error: {e}")

async def example_system_monitoring():
    """Example: System monitoring and statistics"""
    print("\n=== System Monitoring ===")
    
    async with LLMCooperationClient() as client:
        try:
            # Get system status
            status = await client.get_status()
            print(f"System Health: {status['system_health']}")
            print(f"Active Models: {status['active_models']}")
            print(f"Total Requests: {status['total_requests']}")
            print(f"Uptime: {status['uptime']}")
            
            # Get detailed statistics
            stats = await client.get_stats()
            
            print("\n--- Routing Statistics ---")
            routing_stats = stats.get('routing', {})
            if 'error' not in routing_stats:
                print(f"Total requests: {routing_stats.get('total_requests', 0)}")
                print(f"Success rate: {routing_stats.get('success_rate', 0):.2%}")
                print(f"Average latency: {routing_stats.get('average_latency', 0):.2f}s")
            
            print("\n--- Service Statistics ---")
            service_stats = stats.get('services', {})
            if 'error' not in service_stats:
                print(f"Total requests: {service_stats.get('total_requests', 0)}")
                print(f"Success rate: {service_stats.get('success_rate', 0):.2%}")
                print(f"Available services: {service_stats.get('available_services', [])}")
            
            print("\n--- Model Status ---")
            model_stats = stats.get('models', {})
            if 'error' not in model_stats and 'models' in model_stats:
                for model_name, model_info in model_stats['models'].items():
                    print(f"{model_name}: {model_info.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"Error getting system info: {e}")

async def example_performance_comparison():
    """Example: Performance comparison between strategies"""
    print("\n=== Performance Comparison ===")
    
    async with LLMCooperationClient() as client:
        test_query = "Explain the concept of machine learning and its applications in modern technology"
        
        strategies = [
            {"preferences": {"strategy": "single_model"}},
            {"preferences": {"strategy": "sequential"}},
            {"preferences": {"strategy": "parallel"}}
        ]
        
        results = []
        
        for i, strategy_config in enumerate(strategies):
            strategy_name = strategy_config["preferences"]["strategy"]
            print(f"\nTesting {strategy_name} strategy...")
            
            try:
                start_time = time.time()
                result = await client.query(test_query, strategy_config["preferences"])
                elapsed = time.time() - start_time
                
                results.append({
                    "strategy": strategy_name,
                    "time": elapsed,
                    "result_length": len(result['result']),
                    "success": True
                })
                
                print(f"Time: {elapsed:.2f}s")
                print(f"Result length: {len(result['result'])} chars")
                
            except Exception as e:
                results.append({
                    "strategy": strategy_name,
                    "time": 0,
                    "result_length": 0,
                    "success": False,
                    "error": str(e)
                })
                print(f"Error: {e}")
        
        # Summary
        print(f"\n--- Performance Summary ---")
        for result in results:
            if result["success"]:
                print(f"{result['strategy']}: {result['time']:.2f}s ({result['result_length']} chars)")
            else:
                print(f"{result['strategy']}: Failed - {result.get('error', 'Unknown error')}")

async def main():
    """Main example runner"""
    parser = argparse.ArgumentParser(description="LLM Cooperation System Examples")
    parser.add_argument("--url", default="http://localhost:8080", 
                       help="Base URL of the system")
    parser.add_argument("--example", choices=[
        "simple", "cooperation", "services", "monitoring", 
        "performance", "all"
    ], default="all", help="Which example to run")
    
    args = parser.parse_args()
    
    # Set global base URL
    global base_url
    base_url = args.url
    
    print(f"Running examples against: {base_url}")
    
    try:
        if args.example in ["simple", "all"]:
            await example_simple_queries()
        
        if args.example in ["cooperation", "all"]:
            await example_cooperation_modes()
        
        if args.example in ["services", "all"]:
            await example_application_services()
        
        if args.example in ["monitoring", "all"]:
            await example_system_monitoring()
        
        if args.example in ["performance", "all"]:
            await example_performance_comparison()
            
        print("\n=== Examples completed successfully! ===")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())