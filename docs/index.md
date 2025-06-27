---
layout: default
title: LLM Cooperation System
description: "Intelligent multi-model routing and cooperation system for enterprise AI applications"
---

# LLM Cooperation System

Intelligent multi-model routing and cooperation system for enterprise AI applications

[Get started now](#quick-start) | [View on GitHub](https://github.com/ZJU-REAL/llm-cooperation)

---

## Overview

**LLM Cooperation System** is a sophisticated multi-model orchestration platform that intelligently routes requests to optimal language models and coordinates their responses for complex tasks. Built with enterprise-grade reliability and OpenAI-compatible APIs.

### 🎯 Key Features

- **🧠 Intelligent Routing**: Automatically selects the best model based on task complexity and type
- **🤝 Multi-Model Cooperation**: Sequential, parallel, voting, and pipeline coordination modes
- **🔌 Universal API Support**: Works with any OpenAI-compatible endpoint (OpenAI, Anthropic, DeepSeek, AIGC Best, etc.)
- **🏢 Enterprise Services**: Document analysis, data insights, decision support applications
- **📊 Real-time Monitoring**: Performance metrics, health checks, and load balancing
- **⚙️ Flexible Configuration**: Easy setup and deployment with multiple providers

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Application Services Layer                    │
│     Document Analysis  │  Data Insights  │  Decision Support │
├─────────────────────────────────────────────────────────────┤
│                Intelligent Routing Layer                     │
│   Request Analysis  │  Model Selection  │  Performance Opt   │
├─────────────────────────────────────────────────────────────┤
│              Cooperation Scheduling Layer                    │
│ Sequential │ Parallel │ Voting │ Pipeline │ Integration      │
├─────────────────────────────────────────────────────────────┤
│               Model Resource Management                      │
│  Load Balancing  │  Health Monitoring  │  Metrics Collection │
├─────────────────────────────────────────────────────────────┤
│                    OpenAI Engine Layer                      │
│     HTTP Clients    │    Error Handling    │    Parsing      │
├─────────────────────────────────────────────────────────────┤
│                      API Providers                          │
│ AIGC Best │ OpenAI │ Anthropic │ DeepSeek │ Custom Endpoints │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install from GitHub
pip install git+https://github.com/ZJU-REAL/llm-cooperation.git

# Development installation
git clone https://github.com/ZJU-REAL/llm-cooperation.git
cd llm-cooperation
pip install -e ".[dev]"
```

### Basic Configuration

```bash
# Configure with AIGC Best (recommended)
llm-config preset --name aigcbest --api-key YOUR_API_KEY

# Or set environment variables
export BASE_URL=https://api2.aigcbest.top/v1
export API_KEY=your-api-key-here
```

### Simple Usage

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def main():
    # Initialize system
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    # Create request
    request = InferenceRequest(
        prompt="What is artificial intelligence?",
        model_name="qwen3_8b"
    )
    
    # Get response
    response = await engine.inference(request)
    print(f"Response: {response.text}")
    
    await engine.shutdown()

asyncio.run(main())
```

## Documentation

- [Getting Started](getting-started.md) - Installation and basic setup
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - Usage examples and tutorials
- [中文文档](zh/) - Chinese documentation

## Support

- [GitHub Issues](https://github.com/ZJU-REAL/llm-cooperation/issues) - Bug reports and feature requests
- [Discussions](https://github.com/ZJU-REAL/llm-cooperation/discussions) - Community support

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ZJU-REAL/llm-cooperation/blob/main/LICENSE) file for details.

---

**Languages**: [English](/) | [中文](zh/)