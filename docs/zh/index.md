---
layout: default
title: 首页
nav_order: 1
description: "LLM协作系统 - 企业级AI应用的智能多模型路由和协作系统"
permalink: /zh/
---

# LLM协作系统
{: .fs-9 }

企业级AI应用的智能多模型路由和协作系统
{: .fs-6 .fw-300 }

[立即开始](#快速开始){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [查看GitHub](https://github.com/ZJU-REAL/llm-cooperation){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## 系统概述

**LLM协作系统**是一个先进的多模型协调平台，能够智能地将请求路由到最优的语言模型，并协调多个模型的响应来处理复杂任务。该系统采用企业级可靠性设计，支持OpenAI兼容的API接口。

### 🎯 核心特性

- **🧠 智能路由**: 基于任务复杂度和类型自动选择最佳模型
- **🤝 多模型协作**: 支持顺序、并行、投票和流水线协作模式
- **🔌 通用API支持**: 兼容任何OpenAI格式的API端点（OpenAI、Anthropic、DeepSeek、AIGC Best等）
- **🏢 企业服务**: 提供文档分析、数据洞察、决策支持等应用服务
- **📊 实时监控**: 性能指标、健康检查和负载均衡
- **⚙️ 灵活配置**: 支持多个提供商，易于设置和部署

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   应用服务层 (Application Layer)              │
│        文档分析    │    数据洞察    │    决策支持               │
├─────────────────────────────────────────────────────────────┤
│                   智能路由层 (Intelligent Routing)           │
│    请求分析    │    模型选择    │    性能优化                  │
├─────────────────────────────────────────────────────────────┤
│                 协作调度层 (Cooperation Scheduling)          │
│   顺序执行  │  并行处理  │  投票机制  │  流水线  │  集成        │
├─────────────────────────────────────────────────────────────┤
│                 模型资源管理层 (Model Management)             │
│    负载均衡    │    健康监控    │    指标收集                  │
├─────────────────────────────────────────────────────────────┤
│                 OpenAI引擎层 (OpenAI Engine)                 │
│        HTTP客户端    │    错误处理    │    响应解析              │
├─────────────────────────────────────────────────────────────┤
│                      API提供商                               │
│   AIGC Best  │  OpenAI  │  Anthropic  │  DeepSeek  │  自定义  │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 从GitHub安装
pip install git+https://github.com/ZJU-REAL/llm-cooperation.git

# 开发模式安装
git clone https://github.com/ZJU-REAL/llm-cooperation.git
cd llm-cooperation
pip install -e ".[dev]"
```

### 基础配置

```bash
# 使用AIGC Best预设配置（推荐）
llm-config preset --name aigcbest --api-key 您的API密钥

# 或者设置环境变量
export BASE_URL=https://api2.aigcbest.top/v1
export API_KEY=您的API密钥
```

### 简单使用示例

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def main():
    # 初始化系统
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    # 创建请求
    request = InferenceRequest(
        prompt="什么是人工智能？",
        model_name="qwen3_8b"
    )
    
    # 获取响应
    response = await engine.inference(request)
    print(f"响应: {response.text}")
    
    await engine.shutdown()

asyncio.run(main())
```

## 详细功能介绍

### 智能路由系统

系统能够自动分析查询的复杂度和类型，选择最适合的模型：

- **轻量级任务**: 简单翻译、基础问答 → 使用高效模型（如Qwen3-8B）
- **推理任务**: 数学证明、代码分析 → 使用推理模型（如Qwen3-32B）
- **复杂分析**: 多维度分析、策略制定 → 使用多模型协作

### 多模型协作模式

#### 1. 顺序协作 (Sequential)
模型依次处理，后续模型基于前面的结果继续优化：
```python
result = await scheduler.create_sequential_task(
    query="分析人工智能对教育的影响",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="ensemble"
)
```

#### 2. 并行协作 (Parallel) 
多个模型同时处理相同任务，通过投票或加权平均整合结果：
```python
result = await scheduler.create_parallel_task(
    query="评估新能源汽车的发展前景",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="voting"
)
```

#### 3. 流水线协作 (Pipeline)
不同模型负责不同处理阶段，形成完整的处理链：
```python
pipeline_config = [
    {"model": "qwen3_32b", "task": "analysis"},
    {"model": "qwen3_8b", "task": "summary"}
]
result = await scheduler.create_pipeline_task(query, pipeline_config)
```

### 企业级应用服务

#### 文档分析服务
```python
response = await service_manager.process_request(
    service_type="document_analysis",
    content="您的文档内容...",
    parameters={"analysis_type": "comprehensive"}
)
```

#### 数据洞察服务
```python
response = await service_manager.process_request(
    service_type="data_insight",
    content="销售数据: Q1: 100万, Q2: 120万...",
    parameters={"insight_type": "trends"}
)
```

#### 决策支持服务
```python
response = await service_manager.process_request(
    service_type="decision_support",
    content="市场扩张决策场景...",
    parameters={"analysis_framework": "strategic_options"}
)
```

## 支持的API提供商

### AIGC Best（推荐）
- 基础URL: `https://api2.aigcbest.top/v1`
- 可用模型: Qwen/Qwen3-32B, Qwen/Qwen3-8B, DeepSeek-V3等
- 性价比高，稳定性好

### OpenAI
- 基础URL: `https://api.openai.com/v1`
- 可用模型: gpt-4, gpt-3.5-turbo等
- 行业标杆，质量稳定

### 自定义提供商
支持任何OpenAI兼容的API端点，只需配置相应的URL和API密钥。

## 配置管理

### 命令行工具

```bash
# 快速配置预设
llm-config preset --name aigcbest --api-key 您的密钥

# 添加自定义端点
llm-config add-endpoint --name custom --base-url https://api.example.com/v1

# 添加自定义模型
llm-config add-model --name 自定义模型 --path "provider/model-name"

# 测试连接
llm-config test

# 列出所有模型
llm-config list
```

### 环境变量配置

创建`.env`文件：
```env
BASE_URL=https://api2.aigcbest.top/v1
API_KEY=您的API密钥
LOG_LEVEL=INFO
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

## 性能监控

系统提供全面的性能监控和健康检查：

- **实时指标**: 延迟、吞吐量、成功率
- **资源监控**: CPU、内存、网络使用情况
- **模型状态**: 每个模型的健康状态和性能表现
- **告警系统**: 异常情况自动告警
- **负载均衡**: 智能分配请求到最优模型

## 部署和运维

### 服务器模式
```bash
# 启动服务器
llm-cooperation server --host 0.0.0.0 --port 8080

# 健康检查
llm-cooperation health

# 系统状态
llm-cooperation status
```

### 容器化部署
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["llm-cooperation", "server"]
```

### 生产环境配置
- 使用负载均衡器分发请求
- 配置多个API密钥轮换使用
- 设置监控和日志系统
- 配置自动故障切换

## 文档导航

- [快速入门](getting-started.md) - 安装和基础配置指南
- [API参考](api-reference.md) - 完整的API文档
- [使用示例](examples.md) - 各种使用场景的详细示例
- [English Documentation](../) - 英文文档

## 技术支持

- [GitHub Issues](https://github.com/ZJU-REAL/llm-cooperation/issues) - 问题反馈和功能请求
- [Discussions](https://github.com/ZJU-REAL/llm-cooperation/discussions) - 社区讨论

## 开源许可

本项目采用MIT许可证 - 详见[LICENSE](https://github.com/ZJU-REAL/llm-cooperation/blob/main/LICENSE)文件。

## 版本信息

- **当前版本**: 1.0.0
- **最后更新**: 2024年6月
- **维护团队**: ZJU-REAL

---

**语言选择**: [English](/) | [中文](/zh/)