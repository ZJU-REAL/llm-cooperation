#!/usr/bin/env python3
"""
Demo script for AIGC Best API integration
Demonstrates LLM cooperation using models from aigcbest.top
"""
import asyncio
import json
from example_client import LLMCooperationClient
from api_config import APIConfigManager

async def setup_aigcbest_config():
    """Setup AIGC Best configuration"""
    print("Setting up AIGC Best configuration...")
    
    config_manager = APIConfigManager()
    
    # Apply AIGC Best preset
    api_key = "sk-BHJwrDHeR1CXL83svRkwZx0Z9OF4K9LsQDrtEQSbQCCOPA7K"
    config_manager.apply_preset("aigcbest", api_key)
    
    print("✓ Configuration applied")
    
    # Test connectivity
    print("\nTesting model connectivity...")
    results = await config_manager.test_model_connectivity()
    
    for model, result in results.items():
        status = "✓" if result.get('accessible') else "✗"
        print(f"  {status} {model}")
    
    print("\nConfiguration setup complete!")

async def demo_basic_queries():
    """Demo basic queries with different models"""
    print("\n" + "="*60)
    print("DEMO: Basic Queries with Model Selection")
    print("="*60)
    
    async with LLMCooperationClient() as client:
        
        # Test different types of queries with optimal model selection
        test_cases = [
            {
                "query": "Solve this equation: 2x² + 5x - 3 = 0",
                "description": "Math problem (should use reasoning model)"
            },
            {
                "query": "Write a Python function to reverse a linked list",
                "description": "Coding task (should use reasoning model)"  
            },
            {
                "query": "Summarize the benefits of renewable energy",
                "description": "Simple text task (should use lightweight model)"
            },
            {
                "query": "Translate 'Hello, how are you?' to Chinese",
                "description": "Translation task (should use lightweight model)"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {case['description']} ---")
            print(f"Query: {case['query']}")
            
            try:
                result = await client.query(case['query'])
                
                print(f"Strategy: {result['strategy_used']}")
                print(f"Processing time: {result['processing_time']:.2f}s")
                print(f"Result: {result['result'][:200]}...")
                
                if len(result['result']) > 200:
                    print("... (truncated)")
                
            except Exception as e:
                print(f"Error: {e}")

async def demo_cooperation_modes():
    """Demo different cooperation modes"""
    print("\n" + "="*60)
    print("DEMO: Multi-Model Cooperation")
    print("="*60)
    
    async with LLMCooperationClient() as client:
        
        complex_query = """
        I'm starting a new e-commerce business. Please provide a comprehensive analysis covering:
        
        1. Market research and target audience identification
        2. Technology stack recommendations for the platform
        3. Marketing strategy and customer acquisition
        4. Financial projections and funding requirements
        5. Risk assessment and mitigation strategies
        
        Please provide detailed insights for each area.
        """
        
        cooperation_modes = [
            {
                "mode": "sequential",
                "description": "Sequential processing (one model builds on another)",
                "models": ["qwen3_32b", "qwen3_8b"]
            },
            {
                "mode": "parallel", 
                "description": "Parallel processing (multiple models work simultaneously)",
                "models": ["qwen3_32b", "qwen3_8b"]
            },
            {
                "mode": "voting",
                "description": "Voting consensus (multiple models vote on best answer)",
                "models": ["qwen3_32b", "qwen3_8b", "qwen2_5_7b"]
            }
        ]
        
        print(f"Complex Query: {complex_query[:150]}...\n")
        
        for mode_config in cooperation_modes:
            print(f"--- {mode_config['description']} ---")
            
            try:
                result = await client.cooperation_task(
                    query=complex_query,
                    mode=mode_config['mode'],
                    models=mode_config['models'],
                    integration_strategy="ensemble"
                )
                
                print(f"Models used: {result['models']}")
                print(f"Result length: {len(result['result'])} characters")
                print(f"Preview: {result['result'][:300]}...")
                print()
                
            except Exception as e:
                print(f"Error: {e}\n")

async def demo_application_services():
    """Demo application services with real-world examples"""
    print("\n" + "="*60)
    print("DEMO: Application Services")
    print("="*60)
    
    async with LLMCooperationClient() as client:
        
        # Document Analysis Demo
        print("--- Document Analysis Service ---")
        
        sample_document = """
        Executive Summary: Q3 2024 Performance Report
        
        Our company has shown remarkable growth in Q3 2024, with revenue increasing by 34% 
        year-over-year to $12.5 million. Key performance indicators include:
        
        • Customer acquisition: 2,847 new customers (28% increase)
        • Customer retention rate: 94.2% (up from 91.8%)
        • Average revenue per user: $89.34 (15% increase)
        • Market share in key segments: 18.7% (up 3.2 percentage points)
        
        Challenges faced:
        - Supply chain disruptions affecting 12% of orders
        - Increased competition in the mobile segment
        - Rising customer acquisition costs (+22%)
        
        Strategic initiatives for Q4:
        1. Expansion into European markets
        2. Launch of premium product line
        3. Investment in AI-powered customer service
        4. Partnership with leading logistics provider
        
        Financial outlook remains positive with projected Q4 revenue of $14.2-15.1 million.
        """
        
        try:
            result = await client.service_request(
                service_type="document_analysis",
                content=sample_document,
                parameters={
                    "analysis_type": "comprehensive"
                }
            )
            
            print(f"Analysis completed in {result['processing_time']:.2f}s")
            print(f"Analysis preview: {result['result']['analysis'][:400]}...")
            print()
            
        except Exception as e:
            print(f"Document analysis error: {e}\n")
        
        # Data Insights Demo
        print("--- Data Insights Service ---")
        
        sample_data = """
        Monthly Sales Data (2024):
        Jan: $890K (units: 3,456), Feb: $945K (units: 3,678), Mar: $1,120K (units: 4,234)
        Apr: $1,050K (units: 4,012), May: $1,180K (units: 4,456), Jun: $1,340K (units: 4,987)
        Jul: $1,450K (units: 5,234), Aug: $1,380K (units: 5,123), Sep: $1,520K (units: 5,567)
        
        Customer Satisfaction Scores:
        Jan: 87%, Feb: 88%, Mar: 89%, Apr: 91%, May: 92%, Jun: 93%
        Jul: 94%, Aug: 93%, Sep: 95%
        
        Market Share by Region:
        North America: 23%, Europe: 18%, Asia-Pacific: 12%, Latin America: 8%
        """
        
        try:
            result = await client.service_request(
                service_type="data_insight",
                content=sample_data,
                parameters={
                    "insight_type": "trends"
                }
            )
            
            print(f"Insights generated in {result['processing_time']:.2f}s")
            print(f"Insights preview: {result['result']['trend_analysis'][:400]}...")
            print()
            
        except Exception as e:
            print(f"Data insights error: {e}\n")
        
        # Decision Support Demo
        print("--- Decision Support Service ---")
        
        decision_scenario = """
        Decision Required: Cloud Infrastructure Migration
        
        Current Situation:
        - Running on-premise servers with increasing maintenance costs
        - Growing data storage needs (currently 50TB, growing 20% monthly)
        - Need for better scalability and disaster recovery
        
        Options Being Considered:
        
        Option A: AWS Migration
        - Estimated cost: $45K/month
        - Benefits: Mature ecosystem, extensive services, good support
        - Concerns: Vendor lock-in, complex pricing, learning curve
        
        Option B: Google Cloud Platform
        - Estimated cost: $38K/month  
        - Benefits: Better AI/ML tools, competitive pricing, good analytics
        - Concerns: Smaller ecosystem, less enterprise experience
        
        Option C: Microsoft Azure
        - Estimated cost: $42K/month
        - Benefits: Good Windows integration, hybrid cloud options
        - Concerns: Complex licensing, mixed performance reviews
        
        Option D: Multi-cloud approach
        - Estimated cost: $52K/month
        - Benefits: Reduced vendor lock-in, optimized service selection
        - Concerns: Increased complexity, higher management overhead
        
        Decision criteria: Cost, scalability, security, ease of migration, long-term flexibility
        Timeline: Migration must complete within 6 months
        Budget: Maximum $50K/month operational cost
        """
        
        try:
            result = await client.service_request(
                service_type="decision_support",
                content=decision_scenario,
                parameters={
                    "decision_type": "comparison"
                }
            )
            
            print(f"Decision analysis completed in {result['processing_time']:.2f}s")
            print(f"Recommendation preview: {result['result']['comparison'][:400]}...")
            print()
            
        except Exception as e:
            print(f"Decision support error: {e}\n")

async def demo_performance_comparison():
    """Demo performance comparison between strategies"""
    print("\n" + "="*60)
    print("DEMO: Performance Comparison")
    print("="*60)
    
    async with LLMCooperationClient() as client:
        
        test_query = """
        Explain the concept of quantum computing and its potential applications 
        in cryptography, machine learning, and scientific simulation. Include 
        both the opportunities and current limitations.
        """
        
        strategies = [
            {"name": "Single Model (Fast)", "config": {"strategy": "single_model"}},
            {"name": "Sequential Cooperation", "config": {"strategy": "sequential"}},
            {"name": "Parallel Cooperation", "config": {"strategy": "parallel"}},
            {"name": "High Quality", "config": {"strategy": "auto", "quality_priority": 0.9}}
        ]
        
        print(f"Test Query: {test_query[:100]}...\n")
        print("Performance Comparison:")
        print("-" * 40)
        
        for strategy in strategies:
            try:
                import time
                start_time = time.time()
                
                result = await client.query(test_query, strategy["config"])
                
                elapsed = time.time() - start_time
                word_count = len(result['result'].split())
                
                print(f"{strategy['name']:<25} | {elapsed:>6.2f}s | {word_count:>4} words")
                
            except Exception as e:
                print(f"{strategy['name']:<25} | Error: {e}")
        
        print("-" * 40)

async def main():
    """Main demo function"""
    print("AIGC Best API Integration Demo")
    print("=" * 60)
    print("This demo showcases the LLM Cooperation System")
    print("using models from aigcbest.top")
    print()
    
    try:
        # Setup configuration
        await setup_aigcbest_config()
        
        # Run demos
        await demo_basic_queries()
        await demo_cooperation_modes()
        await demo_application_services() 
        await demo_performance_comparison()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Intelligent model routing based on task type")
        print("✓ Multi-model cooperation (sequential, parallel, voting)")
        print("✓ Enterprise application services")
        print("✓ Performance optimization and comparison")
        print("✓ Flexible API configuration")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())