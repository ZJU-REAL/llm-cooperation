---
layout: default
title: API Reference
nav_order: 3
description: "Complete API reference for LLM Cooperation System"
---

# API Reference
{: .no_toc }

Complete reference for all classes, methods, and functions in the LLM Cooperation System.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Core Classes

### SystemConfig

Central configuration management for the system.

```python
from llm_cooperation import SystemConfig

config = SystemConfig()
```

#### Methods

##### `update_api_config(base_url: str, api_key: str)`

Updates the OpenAI API configuration.

**Parameters:**
- `base_url` (str): API endpoint URL
- `api_key` (str): API authentication key

**Example:**
```python
config.update_api_config(
    base_url="https://api2.aigcbest.top/v1",
    api_key="your-api-key"
)
```

##### `add_model_config(model_name: str, model_path: str, **kwargs)`

Adds a new model configuration.

**Parameters:**
- `model_name` (str): Internal model name
- `model_path` (str): API model path (e.g., "Qwen/Qwen3-32B")
- `**kwargs`: Additional model parameters

**Example:**
```python
config.add_model_config(
    model_name="custom_model",
    model_path="provider/custom-model",
    max_tokens=4096,
    supported_tasks=["text", "code"]
)
```

##### `get_model_config(model_name: str) -> dict`

Retrieves configuration for a specific model.

**Returns:** Dictionary containing model configuration

### OpenAIEngine

Main inference engine for OpenAI-compatible APIs.

```python
from llm_cooperation import OpenAIEngine, SystemConfig

config = SystemConfig()
engine = OpenAIEngine(config)
```

#### Methods

##### `async initialize()`

Initializes the engine and sets up API clients.

**Example:**
```python
await engine.initialize()
```

##### `async inference(request: InferenceRequest) -> InferenceResponse`

Performs inference with the specified model.

**Parameters:**
- `request` (InferenceRequest): Inference request object

**Returns:** InferenceResponse object

**Example:**
```python
from llm_cooperation.engines import InferenceRequest

request = InferenceRequest(
    prompt="What is machine learning?",
    model_name="qwen3_8b",
    max_tokens=200
)

response = await engine.inference(request)
```

##### `async get_model_info(model_name: str) -> dict`

Gets information about a specific model.

**Returns:** Dictionary with model information and status

##### `async health_check() -> dict`

Performs a health check on the engine.

**Returns:** Dictionary with health status and metrics

##### `async shutdown()`

Gracefully shuts down the engine and closes connections.

## Data Classes

### InferenceRequest

Request object for model inference.

```python
from llm_cooperation.engines import InferenceRequest

request = InferenceRequest(
    prompt="Your prompt here",
    model_name="qwen3_8b",
    max_tokens=1024,
    temperature=0.7,
    extra_params={"top_p": 0.9}
)
```

#### Attributes

- `prompt` (str): Input text prompt
- `model_name` (str): Target model name
- `max_tokens` (int, optional): Maximum tokens to generate (default: 1024)
- `temperature` (float, optional): Sampling temperature (default: 0.7)
- `extra_params` (dict, optional): Additional model parameters

### InferenceResponse

Response object from model inference.

#### Attributes

- `text` (str): Generated response text
- `model_name` (str): Model that generated the response
- `success` (bool): Whether the inference was successful
- `error` (str, optional): Error message if unsuccessful
- `usage` (dict): Token usage statistics
- `latency` (float): Response latency in seconds
- `metadata` (dict): Additional response metadata

**Example:**
```python
if response.success:
    print(f"Response: {response.text}")
    print(f"Tokens: {response.usage.get('total_tokens')}")
    print(f"Latency: {response.latency:.2f}s")
else:
    print(f"Error: {response.error}")
```

## Intelligent Routing

### IntelligentRouter

Automatically selects optimal models based on task analysis.

```python
from llm_cooperation import IntelligentRouter

router = IntelligentRouter()
```

#### Methods

##### `async route_request(query: str, context: dict = None) -> str`

Analyzes query and routes to appropriate model.

**Parameters:**
- `query` (str): Input query
- `context` (dict, optional): Additional context for routing

**Returns:** Generated response from selected model

**Example:**
```python
# Simple query -> lightweight model
result = await router.route_request("What's 2+2?")

# Complex query -> reasoning model  
result = await router.route_request("Prove the fundamental theorem of calculus")
```

##### `analyze_query_complexity(query: str) -> dict`

Analyzes query complexity and characteristics.

**Returns:** Dictionary with complexity metrics and task type

## Multi-Model Cooperation

### CooperationScheduler

Coordinates multiple models for complex tasks.

```python
from llm_cooperation import CooperationScheduler

scheduler = CooperationScheduler()
```

#### Methods

##### `async create_sequential_task(query: str, models: List[str], integration_strategy: str = "ensemble") -> str`

Creates a sequential cooperation task where models build on each other's work.

**Parameters:**
- `query` (str): Input query
- `models` (List[str]): List of model names to use
- `integration_strategy` (str): How to integrate results ("ensemble", "weighted", "selective")

**Example:**
```python
result = await scheduler.create_sequential_task(
    query="Analyze climate change impacts on agriculture",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="ensemble"
)
```

##### `async create_parallel_task(query: str, models: List[str], integration_strategy: str = "voting") -> str`

Creates a parallel cooperation task where models work independently.

**Parameters:**
- `query` (str): Input query
- `models` (List[str]): List of model names to use
- `integration_strategy` (str): How to integrate results ("voting", "weighted", "consensus")

##### `async create_pipeline_task(query: str, pipeline_config: List[dict]) -> str`

Creates a pipeline where each model performs a specific step.

**Parameters:**
- `query` (str): Input query
- `pipeline_config` (List[dict]): Configuration for each pipeline step

**Example:**
```python
pipeline_config = [
    {"model": "qwen3_32b", "task": "analysis", "params": {}},
    {"model": "qwen3_8b", "task": "summary", "params": {"max_tokens": 200}}
]

result = await scheduler.create_pipeline_task(query, pipeline_config)
```

## Application Services

### ApplicationServiceManager

High-level application services for enterprise use cases.

```python
from llm_cooperation import ApplicationServiceManager

service_manager = ApplicationServiceManager()
```

#### Methods

##### `async process_request(service_type: str, content: str, parameters: dict = None) -> ServiceResponse`

Processes a request using the specified application service.

**Parameters:**
- `service_type` (str): Type of service ("document_analysis", "data_insight", "decision_support")
- `content` (str): Input content
- `parameters` (dict, optional): Service-specific parameters

**Example:**
```python
# Document analysis
response = await service_manager.process_request(
    service_type="document_analysis",
    content="Your document content here...",
    parameters={"analysis_type": "summary"}
)

# Data insights
response = await service_manager.process_request(
    service_type="data_insight", 
    content="Sales data: Q1: 100K, Q2: 120K, Q3: 110K",
    parameters={"insight_type": "trends"}
)
```

### ServiceResponse

Response object from application services.

#### Attributes

- `success` (bool): Whether the service request was successful
- `result` (dict): Service results
- `processing_time` (float): Time taken to process
- `model_used` (str): Model(s) used for processing
- `error` (str, optional): Error message if unsuccessful

## Configuration Tools

### APIConfigManager

Tools for managing API configurations and model setups.

```python
from llm_cooperation.tools import APIConfigManager

config_manager = APIConfigManager()
```

#### Methods

##### `add_model_config(model_name: str, model_path: str, **kwargs)`

Adds a new model configuration.

##### `remove_model(model_name: str)`

Removes a model from the configuration.

##### `list_models() -> dict`

Lists all configured models.

##### `async test_model_connectivity(model_name: str = None) -> dict`

Tests connectivity to configured models.

**Example:**
```python
# Test all models
results = await config_manager.test_model_connectivity()

# Test specific model
results = await config_manager.test_model_connectivity("qwen3_8b")
```

## CLI Commands

### llm-cooperation

Main CLI command for the system.

```bash
# Start server
llm-cooperation server [--host HOST] [--port PORT] [--config CONFIG]

# Single inference
llm-cooperation infer --prompt "TEXT" --model MODEL_NAME

# Health check
llm-cooperation health

# System status
llm-cooperation status

# Benchmark
llm-cooperation benchmark --model MODEL_NAME [--requests N]
```

### llm-config

Configuration management CLI.

```bash
# Quick presets
llm-config preset --name PRESET_NAME --api-key API_KEY

# Add endpoint
llm-config add-endpoint --name NAME --base-url URL --api-key KEY

# Add model
llm-config add-model --name NAME --path PATH [--max-tokens N]

# List models
llm-config list

# Test connectivity
llm-config test [MODEL_NAME]

# Show configuration
llm-config show
```

## Error Handling

### Common Exception Types

#### `ConfigurationError`
Raised when there are configuration issues.

#### `ModelNotFoundError`
Raised when a requested model is not available.

#### `InferenceError`
Raised when inference fails.

#### `APIConnectionError`
Raised when API connection fails.

**Example:**
```python
from llm_cooperation.exceptions import ModelNotFoundError

try:
    response = await engine.inference(request)
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Environment Variables

### Core Configuration

- `BASE_URL`: Default API base URL
- `API_KEY`: Default API key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Server Configuration

- `SERVER_HOST`: Server host (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 8080)

### Model Preferences

- `DEFAULT_REASONING_MODEL`: Default model for complex reasoning
- `DEFAULT_LIGHTWEIGHT_MODEL`: Default model for simple tasks
- `DEFAULT_MULTIMODAL_MODEL`: Default model for multimodal tasks

## Type Hints

The system includes comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
from llm_cooperation.types import (
    ModelConfig,
    APIConfig,
    InferenceParams,
    CooperationMode
)
```

---

**Languages**: [English](/) | [中文](zh/)