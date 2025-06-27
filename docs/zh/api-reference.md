---
layout: default
title: API参考
nav_order: 3
description: "LLM协作系统的完整API参考文档"
parent: 中文文档
---

# API参考
{: .no_toc }

LLM协作系统中所有类、方法和函数的完整参考文档。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 核心类

### SystemConfig

系统的中央配置管理类。

```python
from llm_cooperation import SystemConfig

config = SystemConfig()
```

#### 方法

##### `update_api_config(base_url: str, api_key: str)`

更新OpenAI API配置。

**参数:**
- `base_url` (str): API端点URL
- `api_key` (str): API认证密钥

**示例:**
```python
config.update_api_config(
    base_url="https://api2.aigcbest.top/v1",
    api_key="您的API密钥"
)
```

##### `add_model_config(model_name: str, model_path: str, **kwargs)`

添加新的模型配置。

**参数:**
- `model_name` (str): 内部模型名称
- `model_path` (str): API模型路径（如 "Qwen/Qwen3-32B"）
- `**kwargs`: 额外的模型参数

**示例:**
```python
config.add_model_config(
    model_name="自定义模型",
    model_path="provider/custom-model",
    max_tokens=4096,
    supported_tasks=["文本", "代码"]
)
```

##### `get_model_config(model_name: str) -> dict`

获取特定模型的配置。

**返回:** 包含模型配置的字典

##### `list_models() -> List[str]`

列出所有已配置的模型名称。

**返回:** 模型名称列表

##### `remove_model(model_name: str)`

从配置中移除模型。

**参数:**
- `model_name` (str): 要移除的模型名称

### OpenAIEngine

用于OpenAI兼容API的主要推理引擎。

```python
from llm_cooperation import OpenAIEngine, SystemConfig

config = SystemConfig()
engine = OpenAIEngine(config)
```

#### 方法

##### `async initialize()`

初始化引擎并设置API客户端。

**示例:**
```python
await engine.initialize()
```

##### `async inference(request: InferenceRequest) -> InferenceResponse`

使用指定模型执行推理。

**参数:**
- `request` (InferenceRequest): 推理请求对象

**返回:** InferenceResponse对象

**示例:**
```python
from llm_cooperation.engines import InferenceRequest

request = InferenceRequest(
    prompt="什么是机器学习？",
    model_name="qwen3_8b",
    max_tokens=200
)

response = await engine.inference(request)
```

##### `async get_model_info(model_name: str) -> dict`

获取特定模型的信息。

**返回:** 包含模型信息和状态的字典

##### `async health_check() -> dict`

对引擎执行健康检查。

**返回:** 包含健康状态和指标的字典

##### `async shutdown()`

优雅地关闭引擎并关闭连接。

##### `get_engine_info() -> dict`

获取引擎的基本信息。

**返回:** 包含引擎信息的字典

## 数据类

### InferenceRequest

模型推理的请求对象。

```python
from llm_cooperation.engines import InferenceRequest

request = InferenceRequest(
    prompt="您的提示文本",
    model_name="qwen3_8b",
    max_tokens=1024,
    temperature=0.7,
    extra_params={"top_p": 0.9}
)
```

#### 属性

- `prompt` (str): 输入文本提示
- `model_name` (str): 目标模型名称
- `max_tokens` (int, 可选): 最大生成令牌数（默认: 1024）
- `temperature` (float, 可选): 采样温度（默认: 0.7）
- `top_p` (float, 可选): 核采样参数（默认: 0.9）
- `extra_params` (dict, 可选): 额外的模型参数

### InferenceResponse

模型推理的响应对象。

#### 属性

- `text` (str): 生成的响应文本
- `model_name` (str): 生成响应的模型
- `success` (bool): 推理是否成功
- `error` (str, 可选): 失败时的错误消息
- `usage` (dict): 令牌使用统计
- `latency` (float): 响应延迟（秒）
- `metadata` (dict): 额外的响应元数据

**示例:**
```python
if response.success:
    print(f"响应: {response.text}")
    print(f"令牌数: {response.usage.get('total_tokens')}")
    print(f"延迟: {response.latency:.2f}秒")
else:
    print(f"错误: {response.error}")
```

## 智能路由

### IntelligentRouter

基于任务分析自动选择最优模型。

```python
from llm_cooperation import IntelligentRouter

router = IntelligentRouter()
```

#### 方法

##### `async route_request(query: str, context: dict = None) -> str`

分析查询并路由到适当的模型。

**参数:**
- `query` (str): 输入查询
- `context` (dict, 可选): 路由的额外上下文

**返回:** 从选定模型生成的响应

**示例:**
```python
# 简单查询 -> 轻量级模型
result = await router.route_request("2+2等于多少？")

# 复杂查询 -> 推理模型  
result = await router.route_request("证明微积分基本定理")
```

##### `analyze_query_complexity(query: str) -> dict`

分析查询复杂度和特征。

**返回:** 包含复杂度指标和任务类型的字典

**示例:**
```python
analysis = router.analyze_query_complexity("解这个方程: x² + 5x - 6 = 0")
print(f"复杂度分数: {analysis['complexity_score']}")
print(f"任务类型: {analysis['task_type']}")
print(f"建议策略: {analysis['suggested_strategy']}")
```

##### `set_routing_rules(rules: dict)`

设置自定义路由规则。

**参数:**
- `rules` (dict): 路由规则配置

**示例:**
```python
routing_rules = {
    "数学": {
        "关键词": ["方程", "证明", "计算", "定理"],
        "首选模型": "qwen3_32b",
        "温度": 0.1
    },
    "翻译": {
        "关键词": ["翻译", "translate", "语言"],
        "首选模型": "qwen3_8b",
        "最大令牌": 200
    }
}
router.set_routing_rules(routing_rules)
```

## 多模型协作

### CooperationScheduler

协调多个模型处理复杂任务。

```python
from llm_cooperation import CooperationScheduler

scheduler = CooperationScheduler()
```

#### 方法

##### `async create_sequential_task(query: str, models: List[str], integration_strategy: str = "ensemble") -> str`

创建顺序协作任务，其中模型基于彼此的工作构建。

**参数:**
- `query` (str): 输入查询
- `models` (List[str]): 要使用的模型名称列表
- `integration_strategy` (str): 如何整合结果（"ensemble", "weighted", "selective"）

**示例:**
```python
result = await scheduler.create_sequential_task(
    query="分析气候变化对农业的影响",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="ensemble"
)
```

##### `async create_parallel_task(query: str, models: List[str], integration_strategy: str = "voting") -> str`

创建并行协作任务，其中模型独立工作。

**参数:**
- `query` (str): 输入查询
- `models` (List[str]): 要使用的模型名称列表
- `integration_strategy` (str): 如何整合结果（"voting", "weighted", "consensus"）

**示例:**
```python
result = await scheduler.create_parallel_task(
    query="评估电动汽车的环境影响",
    models=["qwen3_32b", "qwen3_8b"],
    integration_strategy="voting"
)
```

##### `async create_pipeline_task(query: str, pipeline_config: List[dict]) -> str`

创建流水线，其中每个模型执行特定步骤。

**参数:**
- `query` (str): 输入查询
- `pipeline_config` (List[dict]): 每个流水线步骤的配置

**示例:**
```python
pipeline_config = [
    {"model": "qwen3_32b", "task": "分析", "params": {}},
    {"model": "qwen3_8b", "task": "总结", "params": {"max_tokens": 200}}
]

result = await scheduler.create_pipeline_task(query, pipeline_config)
```

##### `async create_voting_task(query: str, models: List[str], voting_params: dict = None) -> dict`

创建投票任务，多个模型提供意见并达成共识。

**参数:**
- `query` (str): 输入查询
- `models` (List[str]): 参与投票的模型列表
- `voting_params` (dict, 可选): 投票参数

**示例:**
```python
result = await scheduler.create_voting_task(
    query="未来十年最有前景的可再生能源技术是什么？",
    models=["qwen3_32b", "qwen3_8b", "qwen3_32b"],
    voting_params={
        "consensus_threshold": 0.6,
        "weight_by_confidence": True,
        "include_dissenting_views": True
    }
)

print(f"共识结果: {result['consensus']}")
print(f"置信度: {result['confidence_score']}")
```

## 应用服务

### ApplicationServiceManager

企业用例的高级应用服务。

```python
from llm_cooperation import ApplicationServiceManager

service_manager = ApplicationServiceManager()
```

#### 方法

##### `async process_request(service_type: str, content: str, parameters: dict = None) -> ServiceResponse`

使用指定的应用服务处理请求。

**参数:**
- `service_type` (str): 服务类型（"document_analysis", "data_insight", "decision_support"）
- `content` (str): 输入内容
- `parameters` (dict, 可选): 服务特定参数

**示例:**
```python
# 文档分析
response = await service_manager.process_request(
    service_type="document_analysis",
    content="您的文档内容...",
    parameters={"analysis_type": "comprehensive"}
)

# 数据洞察
response = await service_manager.process_request(
    service_type="data_insight", 
    content="销售数据: Q1: 100万, Q2: 120万, Q3: 110万",
    parameters={"insight_type": "trends"}
)

# 决策支持
response = await service_manager.process_request(
    service_type="decision_support",
    content="战略决策场景...",
    parameters={"analysis_framework": "strategic_options"}
)
```

##### `register_custom_service(service_name: str, service_handler: callable)`

注册自定义应用服务。

**参数:**
- `service_name` (str): 服务名称
- `service_handler` (callable): 服务处理函数

**示例:**
```python
async def custom_analysis_service(content: str, parameters: dict):
    # 自定义分析逻辑
    return {"analysis": "自定义分析结果"}

service_manager.register_custom_service(
    "custom_analysis", 
    custom_analysis_service
)
```

### ServiceResponse

应用服务的响应对象。

#### 属性

- `success` (bool): 服务请求是否成功
- `result` (dict): 服务结果
- `processing_time` (float): 处理耗时
- `model_used` (str): 用于处理的模型
- `error` (str, 可选): 失败时的错误消息
- `metadata` (dict): 额外的响应元数据

**示例:**
```python
if response.success:
    print(f"分析结果: {response.result}")
    print(f"处理时间: {response.processing_time:.2f}秒")
    print(f"使用的模型: {response.model_used}")
else:
    print(f"处理失败: {response.error}")
```

## 配置工具

### APIConfigManager

管理API配置和模型设置的工具。

```python
from llm_cooperation.tools import APIConfigManager

config_manager = APIConfigManager()
```

#### 方法

##### `add_model_config(model_name: str, model_path: str, **kwargs)`

添加新的模型配置。

##### `remove_model(model_name: str)`

从配置中移除模型。

##### `list_models() -> dict`

列出所有已配置的模型。

##### `async test_model_connectivity(model_name: str = None) -> dict`

测试与已配置模型的连接。

**示例:**
```python
# 测试所有模型
results = await config_manager.test_model_connectivity()

# 测试特定模型
results = await config_manager.test_model_connectivity("qwen3_8b")

for model, result in results.items():
    status = "✅" if result['accessible'] else "❌" 
    print(f"{status} {model}: {result.get('latency', 'N/A')}ms")
```

##### `export_config(file_path: str)`

将配置导出到文件。

##### `import_config(file_path: str)`

从文件导入配置。

##### `validate_config() -> dict`

验证当前配置的有效性。

**返回:** 包含验证结果的字典

## CLI命令

### llm-cooperation

系统的主要CLI命令。

```bash
# 启动服务器
llm-cooperation server [--host HOST] [--port PORT] [--config CONFIG]

# 单次推理
llm-cooperation infer --prompt "文本" --model 模型名称

# 健康检查
llm-cooperation health

# 系统状态
llm-cooperation status

# 性能基准测试
llm-cooperation benchmark --model 模型名称 [--requests N]

# 日志查看
llm-cooperation logs [--tail N] [--follow]
```

### llm-config

配置管理CLI。

```bash
# 快速预设
llm-config preset --name 预设名称 --api-key API密钥

# 添加端点
llm-config add-endpoint --name 名称 --base-url URL --api-key 密钥

# 添加模型
llm-config add-model --name 名称 --path 路径 [--max-tokens N]

# 列出模型
llm-config list

# 测试连接
llm-config test [模型名称]

# 显示配置
llm-config show

# 导出配置
llm-config export --output 文件路径

# 导入配置
llm-config import --input 文件路径

# 验证配置
llm-config validate
```

## 错误处理

### 常见异常类型

#### `ConfigurationError`
配置问题时抛出。

#### `ModelNotFoundError`
请求的模型不可用时抛出。

#### `InferenceError`
推理失败时抛出。

#### `APIConnectionError`
API连接失败时抛出。

#### `RateLimitError`
API速率限制时抛出。

#### `ValidationError`
输入验证失败时抛出。

**示例:**
```python
from llm_cooperation.exceptions import ModelNotFoundError, APIConnectionError

try:
    response = await engine.inference(request)
except ModelNotFoundError as e:
    print(f"模型未找到: {e}")
except APIConnectionError as e:
    print(f"API连接错误: {e}")
except Exception as e:
    print(f"意外错误: {e}")
```

## 环境变量

### 核心配置

- `BASE_URL`: 默认API基础URL
- `API_KEY`: 默认API密钥
- `LOG_LEVEL`: 日志级别（DEBUG, INFO, WARNING, ERROR）
- `LOG_FORMAT`: 日志格式字符串

### 服务器配置

- `SERVER_HOST`: 服务器主机（默认: 0.0.0.0）
- `SERVER_PORT`: 服务器端口（默认: 8080）
- `MAX_CONCURRENT_REQUESTS`: 最大并发请求数
- `REQUEST_TIMEOUT`: 请求超时时间（秒）

### 模型偏好

- `DEFAULT_REASONING_MODEL`: 复杂推理的默认模型
- `DEFAULT_LIGHTWEIGHT_MODEL`: 简单任务的默认模型
- `DEFAULT_MULTIMODAL_MODEL`: 多模态任务的默认模型

### 性能配置

- `RETRY_ATTEMPTS`: 重试次数
- `BACKOFF_FACTOR`: 退避因子
- `CACHE_TTL`: 缓存生存时间（秒）
- `HEALTH_CHECK_INTERVAL`: 健康检查间隔（秒）

## 类型提示

系统包含全面的类型提示以获得更好的IDE支持：

```python
from typing import Dict, List, Optional, Union, AsyncGenerator
from llm_cooperation.types import (
    ModelConfig,
    APIConfig,
    InferenceParams,
    CooperationMode,
    ServiceType,
    RoutingStrategy
)

# 示例类型化函数
async def process_batch(
    requests: List[InferenceRequest],
    config: SystemConfig,
    timeout: Optional[float] = None
) -> List[InferenceResponse]:
    """处理批量推理请求"""
    pass
```

## 监控和指标

### PerformanceMonitor

性能监控和指标收集。

```python
from llm_cooperation.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
```

#### 方法

##### `async record_metrics(metrics: dict)`

记录性能指标。

##### `async get_metrics(time_range: str = "1h") -> dict`

获取指定时间范围的指标。

##### `async check_performance_thresholds(thresholds: dict) -> List[dict]`

检查性能阈值并返回告警。

**示例:**
```python
# 记录指标
await monitor.record_metrics({
    "latency": 1.2,
    "tokens": 150,
    "model": "qwen3_8b",
    "success": True
})

# 获取指标
metrics = await monitor.get_metrics("24h")
print(f"平均延迟: {metrics['avg_latency']:.2f}s")
print(f"成功率: {metrics['success_rate']:.2f}%")

# 检查阈值
alerts = await monitor.check_performance_thresholds({
    "max_latency": 3.0,
    "min_success_rate": 0.95
})
```

## 扩展性

### 自定义引擎

```python
from llm_cooperation.engines.base import BaseEngine

class CustomEngine(BaseEngine):
    async def initialize(self):
        # 初始化逻辑
        pass
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        # 自定义推理逻辑
        pass
    
    async def shutdown(self):
        # 清理逻辑
        pass
```

### 自定义路由器

```python
from llm_cooperation.routing.base import BaseRouter

class CustomRouter(BaseRouter):
    async def route_request(self, query: str, context: dict = None) -> str:
        # 自定义路由逻辑
        pass
```

---

**语言选择**: [English](/) | [中文](/zh/)