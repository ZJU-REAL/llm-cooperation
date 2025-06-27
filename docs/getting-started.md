---
layout: default
title: Getting Started
nav_order: 2
description: "Installation and setup guide for LLM Cooperation System"
---

# Getting Started
{: .no_toc }

Complete guide to installing and configuring the LLM Cooperation System.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager
- Internet connection for API access

### Install from GitHub

```bash
# Basic installation
pip install git+https://github.com/ZJU-REAL/llm-cooperation.git

# With development dependencies
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[dev]"

# With server components
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[server]"

# Full installation with all features
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[all]"
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/ZJU-REAL/llm-cooperation.git
cd llm-cooperation

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/
```

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
# API Configuration
BASE_URL=https://api2.aigcbest.top/v1
API_KEY=your-api-key-here

# Logging
LOG_LEVEL=INFO

# Server Configuration (optional)
SERVER_HOST=0.0.0.0
SERVER_PORT=8080

# Model Preferences (optional)
DEFAULT_REASONING_MODEL=qwen3_32b
DEFAULT_LIGHTWEIGHT_MODEL=qwen3_8b
```

### Using the Configuration CLI

The system includes a powerful CLI tool for configuration management:

```bash
# Quick setup with AIGC Best preset
llm-config preset --name aigcbest --api-key YOUR_API_KEY

# Add custom API endpoint
llm-config add-endpoint --name custom --base-url https://api.example.com/v1 --api-key YOUR_KEY

# Add custom model
llm-config add-model --name custom_model --path "provider/model-name" --max-tokens 4096

# Test connectivity
llm-config test

# List all configured models
llm-config list

# Show current configuration
llm-config show
```

### Supported API Providers

The system supports any OpenAI-compatible API. Here are some popular providers:

#### AIGC Best (Recommended)
```bash
llm-config preset --name aigcbest --api-key YOUR_KEY
```
- Base URL: `https://api2.aigcbest.top/v1`
- Available models: Qwen/Qwen3-32B, Qwen/Qwen3-8B, DeepSeek-V3, and more

#### OpenAI
```bash
llm-config preset --name openai --api-key YOUR_KEY
```
- Base URL: `https://api.openai.com/v1`
- Available models: gpt-4, gpt-3.5-turbo, etc.

#### Custom Provider
```bash
llm-config add-endpoint \
  --name custom \
  --base-url https://your-api.com/v1 \
  --api-key YOUR_KEY \
  --models "model1,model2,model3"
```

## Basic Usage

### Simple Inference

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def basic_example():
    # Initialize configuration
    config = SystemConfig()
    
    # Create engine
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    try:
        # Create request
        request = InferenceRequest(
            prompt="Explain quantum computing in simple terms",
            model_name="qwen3_8b",
            max_tokens=200
        )
        
        # Get response
        response = await engine.inference(request)
        
        if response.success:
            print(f"Model: {response.model_name}")
            print(f"Response: {response.text}")
            print(f"Tokens used: {response.usage.get('total_tokens')}")
            print(f"Latency: {response.latency:.2f}s")
        else:
            print(f"Error: {response.error}")
    
    finally:
        await engine.shutdown()

# Run the example
asyncio.run(basic_example())
```

### Intelligent Routing

The system can automatically select the best model for your task:

```python
from llm_cooperation import IntelligentRouter

async def routing_example():
    router = IntelligentRouter()
    
    # The router will analyze the query and select the appropriate model
    queries = [
        "What's 2+2?",  # Simple math -> lightweight model
        "Prove the Pythagorean theorem",  # Complex math -> reasoning model
        "Translate 'hello' to Spanish",  # Simple task -> lightweight model
    ]
    
    for query in queries:
        result = await router.route_request(query)
        print(f"Query: {query}")
        print(f"Selected model and response: {result[:100]}...")
        print()
```

### Multi-Model Cooperation

For complex tasks, you can use multiple models working together:

```python
from llm_cooperation import CooperationScheduler

async def cooperation_example():
    scheduler = CooperationScheduler()
    
    complex_task = """
    Analyze the pros and cons of renewable energy adoption.
    Consider economic, environmental, and technological factors.
    """
    
    # Sequential cooperation: models build on each other's work
    result = await scheduler.create_sequential_task(
        query=complex_task,
        models=["qwen3_32b", "qwen3_8b"],
        integration_strategy="ensemble"
    )
    
    print("Sequential cooperation result:")
    print(result[:300] + "...")
    
    # Parallel cooperation: models work independently then integrate
    result = await scheduler.create_parallel_task(
        query=complex_task,
        models=["qwen3_32b", "qwen3_8b"],
        integration_strategy="voting"
    )
    
    print("\nParallel cooperation result:")
    print(result[:300] + "...")
```

## Command Line Interface

### Starting the Server

```bash
# Start with default settings
llm-cooperation server

# Start with custom configuration
llm-cooperation server --host 0.0.0.0 --port 8080 --config /path/to/config.json

# Start with debug logging
llm-cooperation server --log-level DEBUG
```

### CLI Commands

```bash
# Test a single inference
llm-cooperation infer --prompt "What is AI?" --model qwen3_8b

# Benchmark performance
llm-cooperation benchmark --model qwen3_8b --requests 10

# Health check
llm-cooperation health

# Show system status
llm-cooperation status

# Configuration management
llm-cooperation config show
llm-cooperation config test
llm-cooperation config list-models
```

## Verification

### Test Your Setup

```python
import asyncio
from llm_cooperation.tools import APIConfigManager

async def test_setup():
    manager = APIConfigManager()
    
    # Test model connectivity
    results = await manager.test_model_connectivity()
    
    for model, result in results.items():
        status = "✅" if result['accessible'] else "❌"
        print(f"{status} {model}: {result.get('latency', 'N/A')}ms")
    
    # Test inference
    from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    request = InferenceRequest(
        prompt="Hello, world!",
        model_name="qwen3_8b",
        max_tokens=50
    )
    
    response = await engine.inference(request)
    
    if response.success:
        print("✅ Setup verified successfully!")
        print(f"Test response: {response.text[:100]}...")
    else:
        print(f"❌ Setup verification failed: {response.error}")
    
    await engine.shutdown()

asyncio.run(test_setup())
```

## Troubleshooting

### Common Issues

**1. Import Error**
```bash
ModuleNotFoundError: No module named 'llm_cooperation'
```
Solution: Ensure the package is installed: `pip install git+https://github.com/ZJU-REAL/llm-cooperation.git`

**2. API Connection Error**
```bash
Connection failed: Invalid API key or endpoint
```
Solution: Check your API key and base URL configuration:
```bash
llm-config test
llm-config show
```

**3. Model Not Found**
```bash
Model 'custom_model' not found in configuration
```
Solution: Add the model to your configuration:
```bash
llm-config add-model --name custom_model --path "provider/model-name"
```

### Getting Help

- Check the [API Reference](api-reference.md) for detailed documentation
- Browse [Examples](examples.md) for more usage patterns
- Report issues on [GitHub](https://github.com/ZJU-REAL/llm-cooperation/issues)

## Next Steps

- Explore [Examples](examples.md) for advanced usage patterns
- Read the [API Reference](api-reference.md) for complete documentation
- Check out the [Chinese documentation](zh/) for additional resources

---

**Languages**: [English](/) | [中文](zh/)