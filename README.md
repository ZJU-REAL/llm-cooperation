# LLM Cooperation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/llm-cooperation.svg)](https://badge.fury.io/py/llm-cooperation)

A comprehensive system for intelligent multi-model LLM routing and coordination using OpenAI-compatible APIs with cooperative processing capabilities.

## 🚀 Features

- **Intelligent Model Routing**: Automatic selection of optimal models based on task type and complexity
- **Multi-Model Cooperation**: Sequential, parallel, voting, and pipeline processing modes
- **OpenAI API Compatibility**: Works with any OpenAI-compatible API endpoint
- **Enterprise Services**: Document analysis, data insights, and decision support
- **Real-time Monitoring**: Performance metrics, health checks, and load balancing
- **Flexible Configuration**: Easy setup with multiple API providers

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install llm-cooperation
```

### From Source

```bash
git clone https://github.com/llm-cooperation/llm-cooperation.git
cd llm-cooperation
pip install -e .
```

### With Optional Dependencies

```bash
# For server functionality
pip install llm-cooperation[server]

# For monitoring capabilities  
pip install llm-cooperation[monitoring]

# Install everything
pip install llm-cooperation[all]
```

## 🏃 Quick Start

### 1. Basic Setup

```python
from llm_cooperation import SystemConfig, OpenAIEngine, IntelligentRouter

# Initialize with AIGC Best API
config = SystemConfig()
config.update_api_config(
    base_url="https://api2.aigcbest.top/v1",
    api_key="your-api-key"
)

# Create engine and router
engine = OpenAIEngine(config)
router = IntelligentRouter(engine)

# Initialize
await engine.initialize()
```

### 2. Simple Query

```python
# Basic query with automatic model selection
result = await router.route_request("Explain quantum computing")
print(result)
```

### 3. Multi-Model Cooperation

```python
from llm_cooperation import CooperationScheduler

scheduler = CooperationScheduler(engine)

# Sequential processing
result = await scheduler.create_sequential_task(
    "Analyze renewable energy trends",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="ensemble"
)
```

### 4. CLI Configuration

```bash
# Setup with preset
llm-config preset --name aigcbest --api-key your-key

# Add custom model
llm-config add-model --name custom --path "provider/model" 

# Test connectivity
llm-config test

# Start server
llm-server --host 0.0.0.0 --port 8080
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
BASE_URL=https://api2.aigcbest.top/v1
API_KEY=your-api-key-here
LOG_LEVEL=INFO
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

### Supported Providers

#### AIGC Best (Recommended)
- **Qwen3-32B**: Complex reasoning, math, code debugging
- **Qwen3-8B**: General text processing, moderate complexity  
- **Qwen2.5-VL-72B**: Multimodal, vision, image analysis
- **Claude 3.5 Sonnet**: High-quality analysis and writing
- **GPT-4o**: Advanced reasoning and problem solving
- **DeepSeek-v3**: Mathematical and logical reasoning

#### Other Providers
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Haiku
- **Any OpenAI-compatible API**

## 📋 Usage Examples

### Enterprise Application Services

```python
from llm_cooperation import ApplicationServiceManager

service_manager = ApplicationServiceManager(engine)

# Document Analysis
result = await service_manager.process_request(
    service_type="document_analysis",
    content="Your document text here...",
    parameters={"analysis_type": "comprehensive"}
)

# Data Insights
result = await service_manager.process_request(
    service_type="data_insight", 
    content="Sales data: Q1: 100K, Q2: 120K...",
    parameters={"insight_type": "trends"}
)

# Decision Support
result = await service_manager.process_request(
    service_type="decision_support",
    content="Which cloud provider should we choose?...",
    parameters={"decision_type": "comparison"}
)
```

### REST API Server

```python
from llm_cooperation.server import create_app

app = create_app()

# POST /query
{
  "query": "Explain machine learning",
  "preferences": {"strategy": "parallel", "quality_priority": 0.8}
}

# POST /cooperation/task  
{
  "query": "Complex analysis request",
  "mode": "voting",
  "models": ["qwen3_32b", "claude_3_5_sonnet"]
}
```

### Performance Monitoring

```python
# Get system status
status = await engine.get_system_metrics()

# Model performance
recommendations = await model_manager.get_model_recommendations("reasoning")

# Routing statistics  
stats = router.get_routing_stats()
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Application Service Layer                   │
│           Document Analysis │ Data Insights │ Decisions    │
├─────────────────────────────────────────────────────────────┤
│                 Intelligent Routing Layer                   │
│     Request Analysis │ Model Selection │ Performance Opt   │
├─────────────────────────────────────────────────────────────┤
│               Cooperation Scheduling Layer                  │
│   Sequential │ Parallel │ Voting │ Pipeline │ Ensemble     │
├─────────────────────────────────────────────────────────────┤
│               Model Resource Management                     │
│    Load Balancing │ Health Monitoring │ Metrics Collection │
├─────────────────────────────────────────────────────────────┤
│                 OpenAI API Engine Layer                     │
│        HTTP Clients │ Error Handling │ Response Parsing    │
├─────────────────────────────────────────────────────────────┤
│                    API Providers                            │
│   AIGC Best │ OpenAI │ Anthropic │ DeepSeek │ Custom APIs   │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 API Reference

### Core Classes

- **SystemConfig**: Main configuration management
- **OpenAIEngine**: OpenAI-compatible API inference engine  
- **ModelResourceManager**: Model lifecycle and performance management
- **IntelligentRouter**: Request analysis and model routing
- **CooperationScheduler**: Multi-model cooperation orchestration
- **ApplicationServiceManager**: Enterprise application services

### Key Methods

```python
# Engine
await engine.initialize()
response = await engine.inference(request)
models = await engine.list_available_models()

# Router  
result = await router.route_request(query, preferences)
analysis = await router.analyze_request(query)

# Cooperation
result = await scheduler.create_parallel_task(query, models)
status = scheduler.get_task_status(task_id)

# Services
response = await service_manager.process_request(type, content, params)
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=llm_cooperation

# Integration tests (requires running system)
pytest tests/integration/

# Load testing
python tests/load_test.py
```

## 📊 Monitoring & Metrics

- **System Health**: Model availability and response times
- **Performance Metrics**: Latency, throughput, error rates
- **Resource Usage**: API usage, token consumption
- **Request Analytics**: Task types, routing decisions
- **Cooperation Stats**: Mode usage, integration success

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/llm-cooperation/llm-cooperation.git
cd llm-cooperation

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the API specification
- AIGC Best for providing diverse model access
- The open-source community for various dependencies

## 📞 Support

- **Documentation**: [https://llm-cooperation.readthedocs.io/](https://llm-cooperation.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/llm-cooperation/llm-cooperation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llm-cooperation/llm-cooperation/discussions)

---

**Made with ❤️ by the LLM Cooperation Team**