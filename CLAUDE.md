# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive LLM Cooperation System implementing intelligent multi-model routing and coordination using vLLM inference engine. The system provides enterprise-grade AI services with automatic model selection, distributed inference, and cooperative processing capabilities.

## Architecture

### Core System Layers

1. **vLLM Inference Engine Layer** (`vllm_engine.py`)
   - Unified inference services with PagedAttention memory management
   - Distributed inference across GPU clusters with tensor/pipeline parallelism
   - Async HTTP API with comprehensive health monitoring

2. **Model Resource Management Layer** (`model_manager.py`) 
   - Dynamic model lifecycle management and load balancing
   - Real-time performance metrics and resource monitoring
   - Intelligent model selection based on current system state

3. **Cooperation Scheduling Layer** (`cooperation_scheduler.py`)
   - Multi-model coordination with sequential/parallel/voting/pipeline modes
   - Task decomposition and result integration strategies
   - Advanced workflow orchestration with dependency management

4. **Intelligent Routing Layer** (`intelligent_router.py`)
   - Request analysis for task type and complexity detection
   - Performance-based routing decisions with caching
   - User preference integration and optimization strategies

5. **Application Service Layer** (`application_service.py`)
   - Enterprise services: document analysis, data insights, decision support
   - Structured API for industry applications
   - Comprehensive request/response tracking

### Model Configuration

- **Qwen3-32B Router**: Fast routing decisions (2 GPU tensor parallel)
- **Qwen3-32B Reasoning**: Complex logic with thinking chains (2 GPU tensor parallel) 
- **Qwen2.5-VL-72B**: Multimodal processing (4 GPU tensor parallel)
- **Qwen2.5-7B**: Lightweight tasks (1 GPU tensor parallel)

## Development Commands

### System Startup
```bash
# Development mode
python start_system.py --dev

# Production mode  
python start_system.py

# Docker deployment
docker-compose up -d
```

### Testing
```bash
# Run all tests
python test_system.py

# Run examples
python example_client.py --example all

# Specific example categories
python example_client.py --example cooperation
python example_client.py --example services
```

### Monitoring
```bash
# System status
curl http://localhost:8080/status

# Model metrics
curl http://localhost:8080/models

# Performance stats
curl http://localhost:8080/routing/stats
```

## Configuration

### Environment Setup
Configure in `API_Key_DeepSeek.env`:
- `BASE_URL`: AI service endpoint (default: https://api2.aigcbest.top/v1)
- `API_KEY`: Authentication key
- `MODEL`: Default model identifier

### OpenAI API Configuration
The system now supports flexible OpenAI-compatible API configuration:

```bash
# Quick setup with preset (AIGC Best)
python api_config.py preset --name aigcbest --api-key YOUR_API_KEY

# Custom API configuration
python api_config.py global --base-url https://api2.aigcbest.top/v1 --api-key YOUR_KEY

# Add individual models
python api_config.py add-model --name custom_model --path Qwen/Qwen3-32B --tasks reasoning math

# Test connectivity
python api_config.py test

# List all models
python api_config.py list
```

### Supported Model Providers
- **AIGC Best** (aigcbest.top): Qwen3-32B, Qwen3-8B, Claude, GPT-4o, DeepSeek-v3
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Haiku
- **Custom endpoints**: Any OpenAI-compatible API

### System Configuration (`config.py`)
- OpenAI API endpoints and model mappings
- Routing strategies and performance thresholds
- Task type mappings and cooperation modes
- Model-specific parameters (temperature, max_tokens)
- Monitoring and logging configuration

## API Usage

### Basic Query
```python
POST /query
{
  "query": "Your question here",
  "preferences": {"strategy": "auto", "quality_priority": 0.8}
}
```

### Application Services
```python
POST /service  
{
  "service_type": "document_analysis",
  "content": "Document text...",
  "parameters": {"analysis_type": "comprehensive"}
}
```

### Cooperation Tasks
```python
POST /cooperation/task
{
  "query": "Complex analysis request",
  "mode": "sequential",
  "models": ["qwen3_32b_reasoning", "qwen2_5_7b"]
}
```

## Development Notes

- System uses vLLM v0.6.0+ for inference engine foundation
- All components implement comprehensive async patterns
- Extensive logging and metrics collection throughout
- GPU memory and compute resources are automatically managed
- Hot-swappable model configurations without system restart
- Built-in performance optimization and load balancing