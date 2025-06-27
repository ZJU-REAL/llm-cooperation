---
layout: default
title: 快速入门
nav_order: 2
description: "LLM协作系统的安装和配置指南"
parent: 中文文档
---

# 快速入门
{: .no_toc }

LLM协作系统的完整安装和配置指南。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 安装要求

### 系统要求

- Python 3.8 或更高版本
- pip 包管理器
- 互联网连接（用于API访问）

### 推荐配置

- **开发环境**: Python 3.9+, 4GB RAM
- **生产环境**: Python 3.9+, 8GB RAM, SSD存储
- **并发场景**: 16GB RAM, 多核CPU

## 安装方法

### 从GitHub安装

```bash
# 基础安装
pip install git+https://github.com/ZJU-REAL/llm-cooperation.git

# 包含开发依赖
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[dev]"

# 包含服务器组件
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[server]"

# 完整安装（包含所有功能）
pip install "git+https://github.com/ZJU-REAL/llm-cooperation.git[all]"
```

### 开发环境安装

```bash
# 克隆仓库
git clone https://github.com/ZJU-REAL/llm-cooperation.git
cd llm-cooperation

# 开发模式安装
pip install -e ".[dev]"

# 运行测试验证安装
pytest tests/

# 检查代码质量
flake8 llm_cooperation/
black --check llm_cooperation/
```

## 基础配置

### 环境变量配置

在项目目录创建`.env`文件：

```env
# API配置
BASE_URL=https://api2.aigcbest.top/v1
API_KEY=您的API密钥

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# 服务器配置（可选）
SERVER_HOST=0.0.0.0
SERVER_PORT=8080

# 模型偏好设置（可选）
DEFAULT_REASONING_MODEL=qwen3_32b
DEFAULT_LIGHTWEIGHT_MODEL=qwen3_8b
DEFAULT_MULTIMODAL_MODEL=qwen2_vl_72b

# 性能配置
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
RETRY_ATTEMPTS=3
```

### 使用配置CLI工具

系统提供了强大的CLI工具来管理配置：

```bash
# AIGC Best快速配置（推荐）
llm-config preset --name aigcbest --api-key 您的API密钥

# 添加自定义API端点
llm-config add-endpoint \
  --name custom \
  --base-url https://api.example.com/v1 \
  --api-key 您的密钥

# 添加自定义模型
llm-config add-model \
  --name custom_model \
  --path "provider/model-name" \
  --max-tokens 4096 \
  --tasks "text,code,analysis"

# 测试连接
llm-config test

# 列出所有已配置的模型
llm-config list

# 显示当前配置
llm-config show

# 导出配置到文件
llm-config export --output config.json

# 从文件导入配置
llm-config import --input config.json
```

### 支持的API提供商

系统支持任何OpenAI兼容的API。以下是一些热门提供商的配置：

#### AIGC Best（推荐）
```bash
llm-config preset --name aigcbest --api-key 您的密钥
```
- 基础URL: `https://api2.aigcbest.top/v1`
- 可用模型: Qwen/Qwen3-32B, Qwen/Qwen3-8B, DeepSeek-V3等
- 优势: 性价比高，模型丰富，稳定性好

#### OpenAI官方
```bash
llm-config preset --name openai --api-key 您的密钥
```
- 基础URL: `https://api.openai.com/v1`
- 可用模型: gpt-4, gpt-3.5-turbo等
- 优势: 行业标杆，质量稳定

#### DeepSeek
```bash
llm-config preset --name deepseek --api-key 您的密钥
```
- 基础URL: `https://api.deepseek.com/v1`
- 可用模型: deepseek-chat, deepseek-coder等
- 优势: 代码生成能力强

#### 自定义提供商
```bash
llm-config add-endpoint \
  --name 自定义名称 \
  --base-url https://您的API.com/v1 \
  --api-key 您的密钥 \
  --models "model1,model2,model3"
```

## 基础使用

### 简单推理示例

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def basic_example():
    # 初始化配置
    config = SystemConfig()
    
    # 如果没有设置环境变量，可以在代码中配置
    config.update_api_config(
        base_url="https://api2.aigcbest.top/v1",
        api_key="您的API密钥"
    )
    
    # 创建引擎
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    try:
        # 创建请求
        request = InferenceRequest(
            prompt="用简单的语言解释什么是量子计算",
            model_name="qwen3_8b",
            max_tokens=300,
            temperature=0.7
        )
        
        # 获取响应
        response = await engine.inference(request)
        
        if response.success:
            print(f"模型: {response.model_name}")
            print(f"响应: {response.text}")
            print(f"使用的令牌数: {response.usage.get('total_tokens')}")
            print(f"延迟: {response.latency:.2f}秒")
        else:
            print(f"错误: {response.error}")
    
    finally:
        await engine.shutdown()

# 运行示例
asyncio.run(basic_example())
```

### 智能路由示例

系统可以自动为您的任务选择最佳模型：

```python
from llm_cooperation import IntelligentRouter

async def routing_example():
    router = IntelligentRouter()
    
    # 路由器会分析查询并选择适当的模型
    queries = [
        "2+2等于多少？",  # 简单数学 -> 轻量级模型
        "证明勾股定理",  # 复杂数学 -> 推理模型
        "将'你好'翻译成英文",  # 简单任务 -> 轻量级模型
        "分析人工智能对教育的深远影响",  # 复杂分析 -> 可能使用协作
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        result = await router.route_request(query)
        print(f"结果: {result[:100]}...")
        print()
```

### 多模型协作示例

对于复杂任务，您可以使用多个模型协同工作：

```python
from llm_cooperation import CooperationScheduler

async def cooperation_example():
    scheduler = CooperationScheduler()
    
    complex_task = """
    分析可再生能源采用的利弊。
    考虑经济、环境和技术因素。
    为政策制定者提供具体建议。
    """
    
    # 顺序协作：模型相互基于彼此的工作
    print("📋 顺序协作:")
    result = await scheduler.create_sequential_task(
        query=complex_task,
        models=["qwen3_32b", "qwen3_8b"],
        integration_strategy="ensemble"
    )
    print(f"结果: {result[:200]}...")
    
    # 并行协作：模型独立工作然后整合
    print("\n📋 并行协作:")
    result = await scheduler.create_parallel_task(
        query=complex_task,
        models=["qwen3_32b", "qwen3_8b"],
        integration_strategy="voting"
    )
    print(f"结果: {result[:200]}...")
```

## 命令行界面

### 启动服务器

```bash
# 使用默认设置启动
llm-cooperation server

# 使用自定义配置启动
llm-cooperation server --host 0.0.0.0 --port 8080 --config /path/to/config.json

# 使用调试日志启动
llm-cooperation server --log-level DEBUG

# 后台运行
nohup llm-cooperation server > server.log 2>&1 &
```

### CLI命令

```bash
# 单次推理测试
llm-cooperation infer --prompt "什么是AI？" --model qwen3_8b

# 性能基准测试
llm-cooperation benchmark --model qwen3_8b --requests 10 --concurrent 2

# 健康检查
llm-cooperation health

# 显示系统状态
llm-cooperation status

# 配置管理
llm-cooperation config show
llm-cooperation config test
llm-cooperation config list-models

# 日志查看
llm-cooperation logs --tail 100 --follow
```

## 配置验证

### 测试您的设置

```python
import asyncio
from llm_cooperation.tools import APIConfigManager

async def test_setup():
    manager = APIConfigManager()
    
    # 测试模型连接
    print("🔍 测试模型连接...")
    results = await manager.test_model_connectivity()
    
    for model, result in results.items():
        status = "✅" if result['accessible'] else "❌"
        latency = f"{result.get('latency', 'N/A')}ms"
        print(f"{status} {model}: {latency}")
    
    # 测试推理功能
    print("\n🧪 测试推理功能...")
    from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    request = InferenceRequest(
        prompt="你好，世界！",
        model_name="qwen3_8b",
        max_tokens=50
    )
    
    response = await engine.inference(request)
    
    if response.success:
        print("✅ 设置验证成功！")
        print(f"测试响应: {response.text[:100]}...")
    else:
        print(f"❌ 设置验证失败: {response.error}")
    
    await engine.shutdown()

asyncio.run(test_setup())
```

### 配置文件示例

创建`config.json`配置文件：

```json
{
  "api_config": {
    "base_url": "https://api2.aigcbest.top/v1",
    "api_key": "您的API密钥",
    "timeout": 30,
    "max_retries": 3
  },
  "models": {
    "qwen3_32b": {
      "model_path": "Qwen/Qwen3-32B",
      "max_tokens": 4096,
      "supported_tasks": ["reasoning", "analysis", "code"],
      "temperature": 0.7
    },
    "qwen3_8b": {
      "model_path": "Qwen/Qwen3-8B", 
      "max_tokens": 2048,
      "supported_tasks": ["text", "translation", "simple"],
      "temperature": 0.7
    }
  },
  "routing": {
    "default_strategy": "intelligent",
    "complexity_threshold": 0.6,
    "fallback_model": "qwen3_8b"
  },
  "cooperation": {
    "max_parallel_models": 3,
    "integration_timeout": 60,
    "consensus_threshold": 0.7
  },
  "monitoring": {
    "enabled": true,
    "metrics_interval": 60,
    "health_check_interval": 30
  }
}
```

## 常见问题解决

### 常见问题

**1. 导入错误**
```bash
ModuleNotFoundError: No module named 'llm_cooperation'
```
解决方案：确保包已安装：`pip install git+https://github.com/ZJU-REAL/llm-cooperation.git`

**2. API连接错误**
```bash
Connection failed: Invalid API key or endpoint
```
解决方案：检查您的API密钥和基础URL配置：
```bash
llm-config test
llm-config show
```

**3. 模型未找到**
```bash
Model 'custom_model' not found in configuration
```
解决方案：将模型添加到配置中：
```bash
llm-config add-model --name custom_model --path "provider/model-name"
```

**4. 内存不足**
```bash
OutOfMemoryError: Cannot allocate memory
```
解决方案：
- 减少`max_concurrent_requests`
- 降低`max_tokens`设置
- 使用更小的模型进行简单任务

**5. 网络超时**
```bash
TimeoutError: Request timed out
```
解决方案：
- 增加`REQUEST_TIMEOUT`设置
- 检查网络连接
- 使用更快的API端点

### 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 查看详细错误信息
try:
    response = await engine.inference(request)
except Exception as e:
    logging.error(f"推理失败: {e}", exc_info=True)
```

### 性能优化建议

1. **模型选择优化**
   - 简单任务使用小模型
   - 复杂任务使用大模型
   - 启用智能路由

2. **并发设置**
   - 根据服务器性能调整并发数
   - 监控内存和CPU使用率
   - 使用连接池

3. **缓存策略**
   - 启用响应缓存
   - 设置合理的缓存过期时间
   - 使用Redis进行分布式缓存

## 下一步

- 探索[使用示例](examples.md)了解高级用法
- 阅读[API参考](api-reference.md)获取完整文档
- 查看[English documentation](../)获取更多资源

---

**语言选择**: [English](/) | [中文](/zh/)