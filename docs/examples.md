---
layout: default
title: Examples
nav_order: 4
description: "Comprehensive examples and tutorials for LLM Cooperation System"
---

# Examples
{: .no_toc }

Comprehensive examples showing various use cases and capabilities of the LLM Cooperation System.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic Usage Examples

### Simple Inference

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def basic_inference():
    """Basic single-model inference example"""
    
    # Initialize system
    config = SystemConfig()
    config.update_api_config(
        base_url="https://api2.aigcbest.top/v1",
        api_key="your-api-key"
    )
    
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    try:
        # Create request
        request = InferenceRequest(
            prompt="Explain the concept of machine learning in simple terms",
            model_name="qwen3_8b",
            max_tokens=300,
            temperature=0.7
        )
        
        # Get response
        response = await engine.inference(request)
        
        if response.success:
            print(f"✅ Model: {response.model_name}")
            print(f"📝 Response:\n{response.text}")
            print(f"⏱️ Latency: {response.latency:.2f}s")
            print(f"🔢 Tokens: {response.usage.get('total_tokens', 'N/A')}")
        else:
            print(f"❌ Error: {response.error}")
    
    finally:
        await engine.shutdown()

# Run the example
asyncio.run(basic_inference())
```

### Batch Processing

```python
async def batch_processing():
    """Process multiple queries efficiently"""
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    # Define queries
    queries = [
        "What is Python?",
        "Explain quantum computing",
        "How does blockchain work?",
        "What is artificial intelligence?",
        "Describe machine learning algorithms"
    ]
    
    try:
        # Process all queries
        tasks = []
        for i, query in enumerate(queries):
            request = InferenceRequest(
                prompt=query,
                model_name="qwen3_8b",
                max_tokens=150
            )
            tasks.append(engine.inference(request))
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks)
        
        # Display results
        for i, (query, response) in enumerate(zip(queries, responses)):
            print(f"\n{'='*50}")
            print(f"Query {i+1}: {query}")
            print(f"{'='*50}")
            if response.success:
                print(f"Response: {response.text[:200]}...")
                print(f"Latency: {response.latency:.2f}s")
            else:
                print(f"Error: {response.error}")
    
    finally:
        await engine.shutdown()

asyncio.run(batch_processing())
```

## Intelligent Routing Examples

### Automatic Model Selection

```python
from llm_cooperation import IntelligentRouter

async def intelligent_routing_demo():
    """Demonstrate automatic model selection based on query complexity"""
    
    router = IntelligentRouter()
    
    # Different types of queries
    test_queries = [
        # Simple queries (should use lightweight model)
        ("Simple Math", "What is 15 + 27?"),
        ("Basic Translation", "Translate 'good morning' to French"),
        ("Simple Definition", "What is HTTP?"),
        
        # Complex queries (should use reasoning model)  
        ("Mathematical Proof", "Prove that the square root of 2 is irrational"),
        ("Code Analysis", "Analyze this Python code for optimization opportunities:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
        ("Complex Analysis", "Compare the economic and environmental impacts of solar vs nuclear energy"),
        
        # Medium complexity (might use cooperation)
        ("Research Task", "Summarize the latest developments in quantum computing"),
        ("Creative Writing", "Write a short story about AI and humanity"),
    ]
    
    for category, query in test_queries:
        print(f"\n🔍 {category}")
        print(f"Query: {query[:100]}...")
        print("-" * 50)
        
        # Analyze query complexity
        analysis = router.analyze_query_complexity(query)
        print(f"📊 Complexity: {analysis['complexity_score']:.2f}")
        print(f"🏷️ Task Type: {analysis['task_type']}")
        print(f"🤖 Suggested Strategy: {analysis['suggested_strategy']}")
        
        # Route and get response
        start_time = time.time()
        result = await router.route_request(query)
        end_time = time.time()
        
        print(f"⚡ Response Time: {end_time - start_time:.2f}s")
        print(f"📝 Result: {result[:150]}...")

import time
asyncio.run(intelligent_routing_demo())
```

### Custom Routing Rules

```python
async def custom_routing_example():
    """Example of custom routing logic"""
    
    from llm_cooperation.routing import CustomRouter
    
    # Define custom routing rules
    routing_rules = {
        "code": {
            "keywords": ["python", "javascript", "code", "function", "class", "algorithm"],
            "preferred_model": "qwen3_32b",
            "min_tokens": 500
        },
        "math": {
            "keywords": ["equation", "solve", "prove", "theorem", "calculate"],
            "preferred_model": "qwen3_32b", 
            "temperature": 0.1
        },
        "creative": {
            "keywords": ["story", "poem", "creative", "imagine", "write"],
            "preferred_model": "qwen3_8b",
            "temperature": 0.9
        },
        "translation": {
            "keywords": ["translate", "translation", "language"],
            "preferred_model": "qwen3_8b",
            "max_tokens": 200
        }
    }
    
    router = CustomRouter(routing_rules)
    
    test_cases = [
        "Write a Python function to calculate fibonacci numbers",
        "Solve the equation: 2x² + 5x - 3 = 0", 
        "Write a creative story about space exploration",
        "Translate this sentence to Spanish: 'The weather is beautiful today'"
    ]
    
    for query in test_cases:
        route_info = router.determine_route(query)
        print(f"\nQuery: {query}")
        print(f"Detected Category: {route_info['category']}")
        print(f"Selected Model: {route_info['model']}")
        print(f"Parameters: {route_info['params']}")
        
        # Execute with determined routing
        result = await router.execute_with_routing(query, route_info)
        print(f"Result: {result[:100]}...")

asyncio.run(custom_routing_example())
```

## Multi-Model Cooperation Examples

### Sequential Cooperation

```python
from llm_cooperation import CooperationScheduler

async def sequential_cooperation_example():
    """Example of sequential model cooperation"""
    
    scheduler = CooperationScheduler()
    
    # Complex analysis task
    complex_query = """
    Analyze the potential impact of artificial general intelligence (AGI) on society.
    Consider technological, economic, social, and ethical dimensions.
    Provide specific recommendations for policy makers.
    """
    
    print("🔄 Sequential Cooperation Example")
    print("=" * 50)
    print(f"Query: {complex_query[:100]}...")
    
    # Step 1: Initial analysis with reasoning model
    print("\n📋 Step 1: Deep Analysis (GPT-4 level model)")
    result_step1 = await scheduler.create_sequential_task(
        query=complex_query,
        models=["qwen3_32b"],  # Start with powerful model
        integration_strategy="detailed_analysis"
    )
    
    # Step 2: Refinement and structuring
    print("\n📋 Step 2: Refinement and Organization")
    refinement_query = f"""
    Based on this analysis: {result_step1}
    
    Please:
    1. Organize the content into clear sections
    2. Add specific examples and case studies
    3. Provide actionable recommendations
    4. Ensure clarity and accessibility
    """
    
    final_result = await scheduler.create_sequential_task(
        query=refinement_query,
        models=["qwen3_8b"],  # Use efficient model for organization
        integration_strategy="structured_output"
    )
    
    print(f"\n✅ Final Result:\n{final_result}")
    print(f"\n📊 Total processing involved 2 models in sequence")

asyncio.run(sequential_cooperation_example())
```

### Parallel Cooperation with Voting

```python
async def parallel_voting_example():
    """Example of parallel cooperation with voting mechanism"""
    
    scheduler = CooperationScheduler()
    
    # Question that benefits from multiple perspectives
    question = """
    What are the most promising renewable energy technologies for the next decade?
    Consider efficiency, cost, scalability, and environmental impact.
    """
    
    print("🗳️ Parallel Cooperation with Voting")
    print("=" * 50)
    print(f"Question: {question}")
    
    # Get responses from multiple models in parallel
    print("\n📊 Getting responses from multiple models...")
    
    result = await scheduler.create_parallel_task(
        query=question,
        models=["qwen3_32b", "qwen3_8b", "qwen3_32b"],  # Use multiple instances
        integration_strategy="voting",
        voting_params={
            "consensus_threshold": 0.6,
            "weight_by_confidence": True,
            "include_dissenting_views": True
        }
    )
    
    print(f"\n✅ Consensus Result:\n{result['consensus']}")
    
    if result.get('dissenting_views'):
        print(f"\n🤔 Alternative Perspectives:\n{result['dissenting_views']}")
    
    print(f"\n📈 Confidence Score: {result['confidence_score']:.2f}")

asyncio.run(parallel_voting_example())
```

### Pipeline Cooperation

```python
async def pipeline_cooperation_example():
    """Example of pipeline-based model cooperation"""
    
    scheduler = CooperationScheduler()
    
    # Document processing pipeline
    document = """
    Artificial Intelligence (AI) is rapidly transforming industries worldwide. 
    From healthcare diagnostics to autonomous vehicles, AI applications are 
    becoming increasingly sophisticated. However, the technology also raises 
    important questions about employment, privacy, and ethical considerations.
    Companies are investing billions in AI research, while governments are 
    grappling with regulatory frameworks. The future of AI will depend on 
    how well we balance innovation with responsible development.
    """
    
    print("🔄 Pipeline Cooperation Example")
    print("=" * 50)
    
    # Define processing pipeline
    pipeline_config = [
        {
            "step": "extraction",
            "model": "qwen3_8b",
            "task": "Extract key facts and statistics from the document",
            "params": {"max_tokens": 300}
        },
        {
            "step": "analysis", 
            "model": "qwen3_32b",
            "task": "Analyze the implications and provide insights based on extracted facts",
            "params": {"max_tokens": 400, "temperature": 0.3}
        },
        {
            "step": "summary",
            "model": "qwen3_8b", 
            "task": "Create an executive summary combining facts and analysis",
            "params": {"max_tokens": 200, "temperature": 0.5}
        },
        {
            "step": "recommendations",
            "model": "qwen3_32b",
            "task": "Provide actionable recommendations based on the analysis",
            "params": {"max_tokens": 300, "temperature": 0.4}
        }
    ]
    
    # Execute pipeline
    result = await scheduler.create_pipeline_task(
        query=document,
        pipeline_config=pipeline_config
    )
    
    # Display results from each step
    for step_name, step_result in result['pipeline_results'].items():
        print(f"\n📋 {step_name.upper()}:")
        print("-" * 30)
        print(step_result['output'][:300] + "...")
        print(f"⏱️ Step time: {step_result['processing_time']:.2f}s")
    
    print(f"\n✅ FINAL INTEGRATED RESULT:")
    print("=" * 50)
    print(result['final_result'])
    
    print(f"\n📊 Pipeline Statistics:")
    print(f"Total steps: {len(pipeline_config)}")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Models used: {', '.join(set([step['model'] for step in pipeline_config]))}")

asyncio.run(pipeline_cooperation_example())
```

## Enterprise Application Examples

### Document Analysis Service

```python
from llm_cooperation import ApplicationServiceManager

async def document_analysis_example():
    """Enterprise document analysis example"""
    
    service_manager = ApplicationServiceManager()
    
    # Sample business document
    business_report = """
    Q3 2024 FINANCIAL PERFORMANCE REPORT
    
    Executive Summary:
    Our company achieved record revenue of $12.5M in Q3 2024, representing a 35% 
    increase over Q3 2023. Customer acquisition grew by 42% with 2,100 new enterprise 
    clients. However, operating expenses increased by 28% due to expanded R&D investments.
    
    Key Metrics:
    - Revenue: $12.5M (+35% YoY)
    - Gross Margin: 68% (+3% YoY) 
    - Customer Count: 7,200 (+42% YoY)
    - Employee Count: 145 (+15% YoY)
    - R&D Spending: $2.1M (+45% YoY)
    
    Challenges:
    - Supply chain disruptions affected 15% of orders
    - Competition intensified in our core market
    - Talent acquisition costs increased 30%
    
    Opportunities:
    - New product launch planned for Q4
    - Expansion into European markets approved
    - Strategic partnership with TechCorp signed
    """
    
    print("📄 Document Analysis Service Example")
    print("=" * 50)
    
    # Comprehensive analysis
    analysis_response = await service_manager.process_request(
        service_type="document_analysis",
        content=business_report,
        parameters={
            "analysis_type": "comprehensive",
            "include_metrics": True,
            "include_recommendations": True,
            "output_format": "structured"
        }
    )
    
    if analysis_response.success:
        result = analysis_response.result
        
        print("📊 KEY INSIGHTS:")
        for insight in result.get('key_insights', []):
            print(f"  • {insight}")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        for metric in result.get('extracted_metrics', []):
            print(f"  • {metric}")
        
        print(f"\n⚠️ RISK FACTORS:")
        for risk in result.get('risk_factors', []):
            print(f"  • {risk}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in result.get('recommendations', []):
            print(f"  • {rec}")
        
        print(f"\n⏱️ Analysis completed in {analysis_response.processing_time:.2f}s")
        print(f"🤖 Models used: {analysis_response.model_used}")
    
    else:
        print(f"❌ Analysis failed: {analysis_response.error}")

asyncio.run(document_analysis_example())
```

### Data Insights Service

```python
async def data_insights_example():
    """Data analysis and insights generation example"""
    
    service_manager = ApplicationServiceManager()
    
    # Sample sales data
    sales_data = """
    Monthly Sales Data (2024):
    January: $850K (650 customers)
    February: $920K (720 customers)  
    March: $1.1M (850 customers)
    April: $980K (780 customers)
    May: $1.2M (920 customers)
    June: $1.35M (1050 customers)
    July: $1.1M (880 customers)
    August: $1.45M (1100 customers)
    September: $1.6M (1200 customers)
    
    Product Categories:
    - Software Licenses: 45% of revenue
    - Professional Services: 30% of revenue  
    - Support & Maintenance: 20% of revenue
    - Training: 5% of revenue
    
    Geographic Distribution:
    - North America: 60%
    - Europe: 25% 
    - Asia-Pacific: 12%
    - Other: 3%
    """
    
    print("📊 Data Insights Service Example")
    print("=" * 50)
    
    # Trend analysis
    trend_response = await service_manager.process_request(
        service_type="data_insight",
        content=sales_data,
        parameters={
            "insight_type": "trends",
            "include_forecasting": True,
            "include_correlations": True,
            "confidence_intervals": True
        }
    )
    
    if trend_response.success:
        result = trend_response.result
        
        print("📈 TREND ANALYSIS:")
        print(f"  Overall Growth: {result.get('overall_trend')}")
        print(f"  Growth Rate: {result.get('growth_rate')}")
        print(f"  Seasonality: {result.get('seasonality_pattern')}")
        
        print(f"\n🔮 FORECASTING:")
        for period, forecast in result.get('forecasts', {}).items():
            print(f"  {period}: {forecast}")
        
        print(f"\n🔗 KEY CORRELATIONS:")
        for correlation in result.get('correlations', []):
            print(f"  • {correlation}")
        
        print(f"\n💡 STRATEGIC INSIGHTS:")
        for insight in result.get('strategic_insights', []):
            print(f"  • {insight}")
    
    # Comparative analysis
    print(f"\n{'='*50}")
    print("🔍 COMPARATIVE ANALYSIS")
    
    comparison_response = await service_manager.process_request(
        service_type="data_insight",
        content=sales_data,
        parameters={
            "insight_type": "comparative",
            "compare_periods": ["Q1", "Q2", "Q3"],
            "include_benchmarks": True
        }
    )
    
    if comparison_response.success:
        comp_result = comparison_response.result
        
        print("📊 QUARTERLY COMPARISON:")
        for quarter, metrics in comp_result.get('quarterly_breakdown', {}).items():
            print(f"  {quarter}: {metrics}")
        
        print(f"\n🎯 PERFORMANCE BENCHMARKS:")
        for benchmark in comp_result.get('benchmarks', []):
            print(f"  • {benchmark}")

asyncio.run(data_insights_example())
```

### Decision Support System

```python
async def decision_support_example():
    """Strategic decision support example"""
    
    service_manager = ApplicationServiceManager()
    
    # Business scenario
    decision_scenario = """
    STRATEGIC DECISION: Market Expansion
    
    Context:
    Our SaaS company (current revenue: $15M/year) is considering expansion 
    into the European market. We have strong product-market fit in North America
    with 85% customer satisfaction and 15% monthly churn rate.
    
    Expansion Options:
    1. Direct Sales Approach: Hire local sales teams in UK, Germany, France
       - Estimated investment: $2.5M in Year 1
       - Projected revenue: $3-5M by end of Year 2
       - Risk level: Medium-High
    
    2. Partner Channel Strategy: Work with local system integrators
       - Estimated investment: $800K in Year 1  
       - Projected revenue: $1.5-3M by end of Year 2
       - Risk level: Medium
    
    3. Acquisition Strategy: Acquire European competitor
       - Estimated investment: $8-12M
       - Projected revenue: $4-6M immediate + growth
       - Risk level: High
    
    Constraints:
    - Available capital: $10M
    - Timeline: Must show results within 18 months
    - Current team: 45 employees (8 in sales)
    - Regulatory: GDPR compliance required
    
    Market Intelligence:
    - European SaaS market growing 22% annually
    - 3 major competitors already established
    - Average customer acquisition cost: €1,200
    - Average deal size: €35K annually
    """
    
    print("🎯 Decision Support System Example")
    print("=" * 50)
    
    # Comprehensive decision analysis
    decision_response = await service_manager.process_request(
        service_type="decision_support",
        content=decision_scenario,
        parameters={
            "analysis_framework": "strategic_options",
            "include_risk_assessment": True,
            "include_financial_modeling": True,
            "decision_criteria": ["ROI", "risk", "timeline", "strategic_fit"],
            "recommendation_confidence": True
        }
    )
    
    if decision_response.success:
        result = decision_response.result
        
        print("⚖️ DECISION ANALYSIS:")
        print(f"Recommended Option: {result.get('recommended_option')}")
        print(f"Confidence Level: {result.get('confidence_score', 0)*100:.1f}%")
        
        print(f"\n📊 OPTION COMPARISON:")
        for option, analysis in result.get('option_analysis', {}).items():
            print(f"\n{option}:")
            print(f"  Pros: {', '.join(analysis.get('pros', []))}")
            print(f"  Cons: {', '.join(analysis.get('cons', []))}")
            print(f"  Risk Score: {analysis.get('risk_score', 'N/A')}/10")
            print(f"  Expected ROI: {analysis.get('expected_roi', 'N/A')}")
        
        print(f"\n🎯 KEY SUCCESS FACTORS:")
        for factor in result.get('success_factors', []):
            print(f"  • {factor}")
        
        print(f"\n⚠️ CRITICAL RISKS:")
        for risk in result.get('critical_risks', []):
            print(f"  • {risk['description']} (Impact: {risk['impact']}, Probability: {risk['probability']})")
        
        print(f"\n📋 IMPLEMENTATION ROADMAP:")
        for phase in result.get('implementation_plan', []):
            print(f"  Phase {phase['phase']}: {phase['description']}")
            print(f"    Timeline: {phase['timeline']}")
            print(f"    Resources: {phase['resources']}")
        
        print(f"\n📈 FINANCIAL PROJECTIONS:")
        for projection in result.get('financial_projections', []):
            print(f"  • {projection}")
    
    else:
        print(f"❌ Decision analysis failed: {decision_response.error}")

asyncio.run(decision_support_example())
```

## Advanced Configuration Examples

### Dynamic Model Management

```python
from llm_cooperation.tools import APIConfigManager

async def dynamic_model_management():
    """Example of dynamic model configuration and management"""
    
    config_manager = APIConfigManager()
    
    print("⚙️ Dynamic Model Management Example")
    print("=" * 50)
    
    # Add multiple API providers
    providers = [
        {
            "name": "aigc_best",
            "base_url": "https://api2.aigcbest.top/v1",
            "api_key": "your-aigc-key",
            "models": ["Qwen/Qwen3-32B", "Qwen/Qwen3-8B", "DeepSeek-V3"]
        },
        {
            "name": "openai", 
            "base_url": "https://api.openai.com/v1",
            "api_key": "your-openai-key",
            "models": ["gpt-4", "gpt-3.5-turbo"]
        },
        {
            "name": "custom_provider",
            "base_url": "https://api.custom.com/v1", 
            "api_key": "your-custom-key",
            "models": ["custom-model-1", "custom-model-2"]
        }
    ]
    
    # Configure providers
    for provider in providers:
        print(f"\n🔧 Configuring {provider['name']}...")
        
        config_manager.add_endpoint_config(
            name=provider['name'],
            base_url=provider['base_url'],
            api_key=provider['api_key']
        )
        
        # Add models for this provider
        for model in provider['models']:
            model_name = f"{provider['name']}_{model.split('/')[-1].lower().replace('-', '_')}"
            config_manager.add_model_config(
                model_name=model_name,
                model_path=model,
                provider=provider['name'],
                supported_tasks=["text", "chat", "completion"]
            )
            print(f"  ✅ Added model: {model_name} -> {model}")
    
    # Test connectivity
    print(f"\n🔍 Testing Model Connectivity...")
    connectivity_results = await config_manager.test_model_connectivity()
    
    for model, result in connectivity_results.items():
        status = "✅" if result['accessible'] else "❌"
        latency = f"{result['latency']:.0f}ms" if result.get('latency') else "N/A"
        print(f"  {status} {model}: {latency}")
    
    # Performance benchmarking
    print(f"\n📊 Performance Benchmarking...")
    benchmark_query = "What is the capital of France?"
    
    for model_name in ["aigc_best_qwen3_8b", "openai_gpt_3_5_turbo"]:
        if connectivity_results.get(model_name, {}).get('accessible'):
            start_time = time.time()
            try:
                # Simulate inference (would use actual engine in real scenario)
                await asyncio.sleep(0.1)  # Simulated response time
                end_time = time.time()
                print(f"  📈 {model_name}: {(end_time-start_time)*1000:.0f}ms")
            except Exception as e:
                print(f"  ❌ {model_name}: Error - {e}")
    
    # Dynamic load balancing configuration
    print(f"\n⚖️ Setting up Load Balancing...")
    load_balance_config = {
        "lightweight_tasks": {
            "models": ["aigc_best_qwen3_8b", "openai_gpt_3_5_turbo"],
            "strategy": "round_robin",
            "health_check_interval": 30
        },
        "reasoning_tasks": {
            "models": ["aigc_best_qwen3_32b", "openai_gpt_4"],
            "strategy": "least_latency",
            "fallback_models": ["aigc_best_qwen3_8b"]
        }
    }
    
    config_manager.configure_load_balancing(load_balance_config)
    print("  ✅ Load balancing configured")
    
    # Model health monitoring
    print(f"\n💓 Health Monitoring Setup...")
    health_config = {
        "check_interval": 60,  # seconds
        "failure_threshold": 3,
        "recovery_threshold": 2,
        "alert_webhooks": ["https://alerts.company.com/webhook"]
    }
    
    config_manager.setup_health_monitoring(health_config)
    print("  ✅ Health monitoring active")

import time
asyncio.run(dynamic_model_management())
```

### Performance Monitoring

```python
from llm_cooperation.monitoring import PerformanceMonitor
import matplotlib.pyplot as plt
import pandas as pd

async def performance_monitoring_example():
    """Example of comprehensive performance monitoring"""
    
    monitor = PerformanceMonitor()
    
    print("📊 Performance Monitoring Example")
    print("=" * 50)
    
    # Simulate various requests for monitoring
    test_scenarios = [
        {"type": "simple", "model": "qwen3_8b", "tokens": 100},
        {"type": "complex", "model": "qwen3_32b", "tokens": 500},
        {"type": "batch", "model": "qwen3_8b", "tokens": 200},
        {"type": "cooperation", "model": "multi", "tokens": 800},
    ]
    
    # Collect performance data
    performance_data = []
    
    for scenario in test_scenarios * 10:  # Run each scenario 10 times
        start_time = time.time()
        
        # Simulate request processing
        processing_time = {
            "simple": 0.5 + random.uniform(-0.2, 0.3),
            "complex": 2.0 + random.uniform(-0.5, 1.0), 
            "batch": 1.2 + random.uniform(-0.3, 0.5),
            "cooperation": 3.5 + random.uniform(-1.0, 2.0)
        }[scenario["type"]]
        
        await asyncio.sleep(processing_time)
        end_time = time.time()
        
        # Record metrics
        metrics = {
            "timestamp": time.time(),
            "request_type": scenario["type"],
            "model": scenario["model"],
            "tokens": scenario["tokens"],
            "latency": end_time - start_time,
            "success": random.choice([True, True, True, False]),  # 75% success rate
            "memory_usage": random.uniform(100, 500),  # MB
            "cpu_usage": random.uniform(20, 80)  # %
        }
        
        performance_data.append(metrics)
        await monitor.record_metrics(metrics)
    
    # Generate performance report
    print(f"\n📈 PERFORMANCE SUMMARY:")
    df = pd.DataFrame(performance_data)
    
    # Overall statistics
    print(f"Total Requests: {len(df)}")
    print(f"Success Rate: {(df['success'].sum() / len(df) * 100):.1f}%")
    print(f"Average Latency: {df['latency'].mean():.2f}s")
    print(f"95th Percentile Latency: {df['latency'].quantile(0.95):.2f}s")
    
    # By request type
    print(f"\n📊 BY REQUEST TYPE:")
    type_stats = df.groupby('request_type').agg({
        'latency': ['mean', 'std', 'count'],
        'success': 'mean',
        'tokens': 'mean'
    }).round(2)
    
    for req_type in type_stats.index:
        stats = type_stats.loc[req_type]
        print(f"  {req_type.upper()}:")
        print(f"    Avg Latency: {stats[('latency', 'mean')]:.2f}s ± {stats[('latency', 'std')]:.2f}s")
        print(f"    Success Rate: {stats[('success', 'mean')]*100:.1f}%")
        print(f"    Requests: {stats[('latency', 'count')]}")
    
    # Performance alerts
    print(f"\n🚨 PERFORMANCE ALERTS:")
    alerts = await monitor.check_performance_thresholds({
        "max_latency": 3.0,
        "min_success_rate": 0.90,
        "max_error_rate": 0.10
    })
    
    for alert in alerts:
        print(f"  ⚠️ {alert['severity'].upper()}: {alert['message']}")
    
    # Resource utilization
    print(f"\n💾 RESOURCE UTILIZATION:")
    print(f"Average Memory Usage: {df['memory_usage'].mean():.1f} MB")
    print(f"Peak Memory Usage: {df['memory_usage'].max():.1f} MB")
    print(f"Average CPU Usage: {df['cpu_usage'].mean():.1f}%")
    print(f"Peak CPU Usage: {df['cpu_usage'].max():.1f}%")
    
    # Trend analysis
    print(f"\n📈 TREND ANALYSIS:")
    recent_data = df.tail(20)  # Last 20 requests
    older_data = df.head(20)   # First 20 requests
    
    latency_trend = recent_data['latency'].mean() - older_data['latency'].mean()
    success_trend = recent_data['success'].mean() - older_data['success'].mean()
    
    print(f"Latency Trend: {'↗️ +' if latency_trend > 0 else '↘️ '}{latency_trend:.2f}s")
    print(f"Success Rate Trend: {'↗️ +' if success_trend > 0 else '↘️ '}{success_trend*100:.1f}%")

import random
import time
asyncio.run(performance_monitoring_example())
```

## Error Handling and Resilience Examples

### Comprehensive Error Handling

```python
from llm_cooperation.exceptions import *
import logging

async def error_handling_example():
    """Comprehensive error handling and resilience example"""
    
    print("🛡️ Error Handling and Resilience Example")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    
    # Test scenarios that might fail
    error_scenarios = [
        {
            "name": "Invalid Model",
            "request": InferenceRequest(
                prompt="Test", 
                model_name="nonexistent_model"
            ),
            "expected_error": ModelNotFoundError
        },
        {
            "name": "Empty Prompt", 
            "request": InferenceRequest(
                prompt="",
                model_name="qwen3_8b"
            ),
            "expected_error": ValidationError
        },
        {
            "name": "Excessive Token Request",
            "request": InferenceRequest(
                prompt="Test" * 1000,
                model_name="qwen3_8b", 
                max_tokens=50000  # Unrealistic number
            ),
            "expected_error": APILimitError
        }
    ]
    
    try:
        await engine.initialize()
        
        for scenario in error_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            print("-" * 30)
            
            try:
                response = await engine.inference(scenario['request'])
                
                if not response.success:
                    print(f"✅ Expected error handled gracefully: {response.error}")
                else:
                    print(f"⚠️ Unexpected success: {response.text[:50]}...")
                    
            except scenario['expected_error'] as e:
                print(f"✅ Caught expected error: {type(e).__name__}: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {type(e).__name__}: {e}")
                logger.error(f"Unexpected error in scenario {scenario['name']}", exc_info=True)
        
        # Test resilience with retry mechanism
        print(f"\n🔄 Testing Retry Mechanism")
        print("-" * 30)
        
        from llm_cooperation.resilience import RetryHandler
        
        retry_handler = RetryHandler(
            max_retries=3,
            backoff_strategy="exponential",
            base_delay=1.0,
            max_delay=10.0
        )
        
        async def flaky_function():
            """Simulates a function that fails randomly"""
            if random.random() < 0.7:  # 70% chance of failure
                raise APIConnectionError("Simulated network error")
            return "Success!"
        
        try:
            result = await retry_handler.execute_with_retry(flaky_function)
            print(f"✅ Retry mechanism succeeded: {result}")
        except Exception as e:
            print(f"❌ Retry mechanism exhausted: {e}")
        
        # Test circuit breaker
        print(f"\n⚡ Testing Circuit Breaker")
        print("-" * 30)
        
        from llm_cooperation.resilience import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            expected_exception=APIConnectionError
        )
        
        async def unreliable_service():
            """Simulates an unreliable external service"""
            if random.random() < 0.8:  # 80% failure rate
                raise APIConnectionError("Service unavailable")
            return "Service response"
        
        # Test circuit breaker behavior
        for i in range(10):
            try:
                result = await circuit_breaker.call(unreliable_service)
                print(f"  Request {i+1}: ✅ {result}")
            except CircuitBreakerOpenError:
                print(f"  Request {i+1}: 🚫 Circuit breaker open")
            except APIConnectionError as e:
                print(f"  Request {i+1}: ❌ {e}")
            
            await asyncio.sleep(0.5)  # Brief delay between requests
        
        # Test graceful degradation
        print(f"\n🎭 Testing Graceful Degradation")
        print("-" * 30)
        
        from llm_cooperation.resilience import GracefulDegradation
        
        degradation_handler = GracefulDegradation(
            fallback_models=["qwen3_8b", "backup_model"],
            cache_enabled=True,
            cache_ttl=300  # 5 minutes
        )
        
        # Primary model fails, should fallback
        primary_request = InferenceRequest(
            prompt="What is machine learning?",
            model_name="primary_model_that_fails"
        )
        
        try:
            response = await degradation_handler.handle_request(primary_request)
            print(f"✅ Graceful degradation worked: Used {response.model_name}")
            print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"❌ Graceful degradation failed: {e}")
    
    finally:
        await engine.shutdown()

asyncio.run(error_handling_example())
```

This comprehensive examples documentation provides practical, real-world usage patterns for the LLM Cooperation System, covering everything from basic inference to advanced enterprise applications with error handling and performance monitoring.

---

**Languages**: [English](/) | [中文](zh/)