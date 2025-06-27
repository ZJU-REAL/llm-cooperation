---
layout: default
title: 使用示例
nav_order: 4
description: "LLM协作系统的综合使用示例和教程"
parent: 中文文档
---

# 使用示例
{: .no_toc }

展示LLM协作系统各种用例和功能的综合示例。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 基础使用示例

### 简单推理

```python
import asyncio
from llm_cooperation import SystemConfig, OpenAIEngine, InferenceRequest

async def basic_inference():
    """基础单模型推理示例"""
    
    # 初始化系统
    config = SystemConfig()
    config.update_api_config(
        base_url="https://api2.aigcbest.top/v1",
        api_key="您的API密钥"
    )
    
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    try:
        # 创建请求
        request = InferenceRequest(
            prompt="用简单的语言解释机器学习的概念",
            model_name="qwen3_8b",
            max_tokens=300,
            temperature=0.7
        )
        
        # 获取响应
        response = await engine.inference(request)
        
        if response.success:
            print(f"✅ 模型: {response.model_name}")
            print(f"📝 响应:\n{response.text}")
            print(f"⏱️ 延迟: {response.latency:.2f}秒")
            print(f"🔢 令牌数: {response.usage.get('total_tokens', 'N/A')}")
        else:
            print(f"❌ 错误: {response.error}")
    
    finally:
        await engine.shutdown()

# 运行示例
asyncio.run(basic_inference())
```

### 批量处理

```python
async def batch_processing():
    """批量处理多个查询"""
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    await engine.initialize()
    
    # 定义查询
    queries = [
        "什么是Python？",
        "解释量子计算",
        "区块链是如何工作的？",
        "什么是人工智能？",
        "描述机器学习算法"
    ]
    
    try:
        # 处理所有查询
        tasks = []
        for i, query in enumerate(queries):
            request = InferenceRequest(
                prompt=query,
                model_name="qwen3_8b",
                max_tokens=150
            )
            tasks.append(engine.inference(request))
        
        # 等待所有响应
        responses = await asyncio.gather(*tasks)
        
        # 显示结果
        for i, (query, response) in enumerate(zip(queries, responses)):
            print(f"\n{'='*50}")
            print(f"查询 {i+1}: {query}")
            print(f"{'='*50}")
            if response.success:
                print(f"响应: {response.text[:200]}...")
                print(f"延迟: {response.latency:.2f}秒")
            else:
                print(f"错误: {response.error}")
    
    finally:
        await engine.shutdown()

asyncio.run(batch_processing())
```

## 智能路由示例

### 自动模型选择

```python
from llm_cooperation import IntelligentRouter

async def intelligent_routing_demo():
    """演示基于查询复杂度的自动模型选择"""
    
    router = IntelligentRouter()
    
    # 不同类型的查询
    test_queries = [
        # 简单查询（应使用轻量级模型）
        ("简单数学", "15 + 27 等于多少？"),
        ("基础翻译", "将'早上好'翻译成英文"),
        ("简单定义", "什么是HTTP？"),
        
        # 复杂查询（应使用推理模型）  
        ("数学证明", "证明根号2是无理数"),
        ("代码分析", "分析这段Python代码的优化机会：\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
        ("复杂分析", "比较太阳能和核能的经济与环境影响"),
        
        # 中等复杂度（可能使用协作）
        ("研究任务", "总结量子计算的最新发展"),
        ("创意写作", "写一个关于AI与人类的短故事"),
    ]
    
    for category, query in test_queries:
        print(f"\n🔍 {category}")
        print(f"查询: {query[:100]}...")
        print("-" * 50)
        
        # 分析查询复杂度
        analysis = router.analyze_query_complexity(query)
        print(f"📊 复杂度: {analysis['complexity_score']:.2f}")
        print(f"🏷️ 任务类型: {analysis['task_type']}")
        print(f"🤖 建议策略: {analysis['suggested_strategy']}")
        
        # 路由并获取响应
        start_time = time.time()
        result = await router.route_request(query)
        end_time = time.time()
        
        print(f"⚡ 响应时间: {end_time - start_time:.2f}秒")
        print(f"📝 结果: {result[:150]}...")

import time
asyncio.run(intelligent_routing_demo())
```

### 自定义路由规则

```python
async def custom_routing_example():
    """自定义路由逻辑示例"""
    
    from llm_cooperation.routing import CustomRouter
    
    # 定义自定义路由规则
    routing_rules = {
        "代码": {
            "关键词": ["python", "javascript", "代码", "函数", "类", "算法"],
            "首选模型": "qwen3_32b",
            "最小令牌": 500
        },
        "数学": {
            "关键词": ["方程", "解", "证明", "定理", "计算"],
            "首选模型": "qwen3_32b", 
            "温度": 0.1
        },
        "创意": {
            "关键词": ["故事", "诗歌", "创意", "想象", "写"],
            "首选模型": "qwen3_8b",
            "温度": 0.9
        },
        "翻译": {
            "关键词": ["翻译", "translation", "语言"],
            "首选模型": "qwen3_8b",
            "最大令牌": 200
        }
    }
    
    router = CustomRouter(routing_rules)
    
    test_cases = [
        "写一个Python函数来计算斐波那契数",
        "解方程: 2x² + 5x - 3 = 0", 
        "写一个关于太空探索的创意故事",
        "将这句话翻译成西班牙语: '今天天气很好'"
    ]
    
    for query in test_cases:
        route_info = router.determine_route(query)
        print(f"\n查询: {query}")
        print(f"检测到的类别: {route_info['category']}")
        print(f"选择的模型: {route_info['model']}")
        print(f"参数: {route_info['params']}")
        
        # 使用确定的路由执行
        result = await router.execute_with_routing(query, route_info)
        print(f"结果: {result[:100]}...")

asyncio.run(custom_routing_example())
```

## 多模型协作示例

### 顺序协作

```python
from llm_cooperation import CooperationScheduler

async def sequential_cooperation_example():
    """顺序模型协作示例"""
    
    scheduler = CooperationScheduler()
    
    # 复杂分析任务
    complex_query = """
    分析人工通用智能（AGI）对社会的潜在影响。
    考虑技术、经济、社会和伦理维度。
    为政策制定者提供具体建议。
    """
    
    print("🔄 顺序协作示例")
    print("=" * 50)
    print(f"查询: {complex_query[:100]}...")
    
    # 步骤1：使用推理模型进行初步分析
    print("\n📋 步骤1: 深度分析（GPT-4级别模型）")
    result_step1 = await scheduler.create_sequential_task(
        query=complex_query,
        models=["qwen3_32b"],  # 从强大的模型开始
        integration_strategy="detailed_analysis"
    )
    
    # 步骤2：优化和结构化
    print("\n📋 步骤2: 优化和组织")
    refinement_query = f"""
    基于这个分析: {result_step1}
    
    请：
    1. 将内容组织成清晰的部分
    2. 添加具体的例子和案例研究
    3. 提供可操作的建议
    4. 确保清晰度和可访问性
    """
    
    final_result = await scheduler.create_sequential_task(
        query=refinement_query,
        models=["qwen3_8b"],  # 使用高效模型进行组织
        integration_strategy="structured_output"
    )
    
    print(f"\n✅ 最终结果:\n{final_result}")
    print(f"\n📊 总处理涉及2个模型的顺序协作")

asyncio.run(sequential_cooperation_example())
```

### 并行协作与投票

```python
async def parallel_voting_example():
    """并行协作与投票机制示例"""
    
    scheduler = CooperationScheduler()
    
    # 受益于多个视角的问题
    question = """
    未来十年最有前景的可再生能源技术是什么？
    考虑效率、成本、可扩展性和环境影响。
    """
    
    print("🗳️ 并行协作与投票示例")
    print("=" * 50)
    print(f"问题: {question}")
    
    # 从多个模型并行获取响应
    print("\n📊 从多个模型获取响应...")
    
    result = await scheduler.create_parallel_task(
        query=question,
        models=["qwen3_32b", "qwen3_8b", "qwen3_32b"],  # 使用多个实例
        integration_strategy="voting",
        voting_params={
            "consensus_threshold": 0.6,
            "weight_by_confidence": True,
            "include_dissenting_views": True
        }
    )
    
    print(f"\n✅ 共识结果:\n{result['consensus']}")
    
    if result.get('dissenting_views'):
        print(f"\n🤔 不同观点:\n{result['dissenting_views']}")
    
    print(f"\n📈 置信度分数: {result['confidence_score']:.2f}")

asyncio.run(parallel_voting_example())
```

### 流水线协作

```python
async def pipeline_cooperation_example():
    """基于流水线的模型协作示例"""
    
    scheduler = CooperationScheduler()
    
    # 文档处理流水线
    document = """
    人工智能（AI）正在迅速改变全球各行各业。
    从医疗诊断到自动驾驶汽车，AI应用变得越来越复杂。
    然而，这项技术也提出了关于就业、隐私和伦理考虑的重要问题。
    公司正在AI研究上投资数十亿美元，而政府正在努力制定监管框架。
    AI的未来将取决于我们如何平衡创新与负责任的发展。
    """
    
    print("🔄 流水线协作示例")
    print("=" * 50)
    
    # 定义处理流水线
    pipeline_config = [
        {
            "step": "extraction",
            "model": "qwen3_8b",
            "task": "从文档中提取关键事实和统计数据",
            "params": {"max_tokens": 300}
        },
        {
            "step": "analysis", 
            "model": "qwen3_32b",
            "task": "基于提取的事实分析影响并提供洞察",
            "params": {"max_tokens": 400, "temperature": 0.3}
        },
        {
            "step": "summary",
            "model": "qwen3_8b", 
            "task": "结合事实和分析创建执行摘要",
            "params": {"max_tokens": 200, "temperature": 0.5}
        },
        {
            "step": "recommendations",
            "model": "qwen3_32b",
            "task": "基于分析提供可操作的建议",
            "params": {"max_tokens": 300, "temperature": 0.4}
        }
    ]
    
    # 执行流水线
    result = await scheduler.create_pipeline_task(
        query=document,
        pipeline_config=pipeline_config
    )
    
    # 显示每个步骤的结果
    for step_name, step_result in result['pipeline_results'].items():
        print(f"\n📋 {step_name.upper()}:")
        print("-" * 30)
        print(step_result['output'][:300] + "...")
        print(f"⏱️ 步骤时间: {step_result['processing_time']:.2f}秒")
    
    print(f"\n✅ 最终整合结果:")
    print("=" * 50)
    print(result['final_result'])
    
    print(f"\n📊 流水线统计:")
    print(f"总步骤数: {len(pipeline_config)}")
    print(f"总时间: {result['total_time']:.2f}秒")
    print(f"使用的模型: {', '.join(set([step['model'] for step in pipeline_config]))}")

asyncio.run(pipeline_cooperation_example())
```

## 企业应用示例

### 文档分析服务

```python
from llm_cooperation import ApplicationServiceManager

async def document_analysis_example():
    """企业文档分析示例"""
    
    service_manager = ApplicationServiceManager()
    
    # 样本商业文档
    business_report = """
    2024年第三季度财务表现报告
    
    执行摘要：
    我们公司在2024年第三季度实现了创纪录的1250万美元收入，
    比2023年第三季度增长35%。客户获取增长42%，新增2100家企业客户。
    然而，由于扩大的研发投资，运营费用增加了28%。
    
    关键指标：
    - 收入: 1250万美元（同比增长35%）
    - 毛利率: 68%（同比增长3%）
    - 客户数量: 7200（同比增长42%）
    - 员工数量: 145（同比增长15%）
    - 研发支出: 210万美元（同比增长45%）
    
    挑战：
    - 供应链中断影响了15%的订单
    - 核心市场竞争加剧
    - 人才获取成本增加30%
    
    机遇：
    - 计划第四季度推出新产品
    - 获批进入欧洲市场扩张
    - 与TechCorp签署战略合作伙伴关系
    """
    
    print("📄 文档分析服务示例")
    print("=" * 50)
    
    # 综合分析
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
        
        print("📊 关键洞察:")
        for insight in result.get('key_insights', []):
            print(f"  • {insight}")
        
        print(f"\n📈 性能指标:")
        for metric in result.get('extracted_metrics', []):
            print(f"  • {metric}")
        
        print(f"\n⚠️ 风险因素:")
        for risk in result.get('risk_factors', []):
            print(f"  • {risk}")
        
        print(f"\n💡 建议:")
        for rec in result.get('recommendations', []):
            print(f"  • {rec}")
        
        print(f"\n⏱️ 分析在{analysis_response.processing_time:.2f}秒内完成")
        print(f"🤖 使用的模型: {analysis_response.model_used}")
    
    else:
        print(f"❌ 分析失败: {analysis_response.error}")

asyncio.run(document_analysis_example())
```

### 数据洞察服务

```python
async def data_insights_example():
    """数据分析和洞察生成示例"""
    
    service_manager = ApplicationServiceManager()
    
    # 样本销售数据
    sales_data = """
    月度销售数据（2024年）：
    一月: 85万元（650个客户）
    二月: 92万元（720个客户）
    三月: 110万元（850个客户）
    四月: 98万元（780个客户）
    五月: 120万元（920个客户）
    六月: 135万元（1050个客户）
    七月: 110万元（880个客户）
    八月: 145万元（1100个客户）
    九月: 160万元（1200个客户）
    
    产品类别：
    - 软件许可: 收入的45%
    - 专业服务: 收入的30%
    - 支持与维护: 收入的20%
    - 培训: 收入的5%
    
    地理分布：
    - 北美: 60%
    - 欧洲: 25%
    - 亚太: 12%
    - 其他: 3%
    """
    
    print("📊 数据洞察服务示例")
    print("=" * 50)
    
    # 趋势分析
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
        
        print("📈 趋势分析:")
        print(f"  总体趋势: {result.get('overall_trend')}")
        print(f"  增长率: {result.get('growth_rate')}")
        print(f"  季节性: {result.get('seasonality_pattern')}")
        
        print(f"\n🔮 预测:")
        for period, forecast in result.get('forecasts', {}).items():
            print(f"  {period}: {forecast}")
        
        print(f"\n🔗 关键相关性:")
        for correlation in result.get('correlations', []):
            print(f"  • {correlation}")
        
        print(f"\n💡 战略洞察:")
        for insight in result.get('strategic_insights', []):
            print(f"  • {insight}")
    
    # 比较分析
    print(f"\n{'='*50}")
    print("🔍 比较分析")
    
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
        
        print("📊 季度比较:")
        for quarter, metrics in comp_result.get('quarterly_breakdown', {}).items():
            print(f"  {quarter}: {metrics}")
        
        print(f"\n🎯 性能基准:")
        for benchmark in comp_result.get('benchmarks', []):
            print(f"  • {benchmark}")

asyncio.run(data_insights_example())
```

### 决策支持系统

```python
async def decision_support_example():
    """战略决策支持示例"""
    
    service_manager = ApplicationServiceManager()
    
    # 商业场景
    decision_scenario = """
    战略决策：市场扩张
    
    背景：
    我们的SaaS公司（当前收入：1500万元/年）正在考虑进入欧洲市场。
    我们在北美有强大的产品市场匹配，客户满意度85%，月流失率15%。
    
    扩张选项：
    1. 直销方式：在英国、德国、法国雇佣本地销售团队
       - 预估投资：第一年250万元
       - 预计收入：第二年底300-500万元
       - 风险级别：中高
    
    2. 合作伙伴渠道策略：与本地系统集成商合作
       - 预估投资：第一年80万元
       - 预计收入：第二年底150-300万元
       - 风险级别：中等
    
    3. 收购策略：收购欧洲竞争对手
       - 预估投资：800-1200万元
       - 预计收入：400-600万元立即+增长
       - 风险级别：高
    
    约束条件：
    - 可用资本：1000万元
    - 时间线：必须在18个月内显示结果
    - 当前团队：45名员工（其中8名销售）
    - 监管：需要GDPR合规
    
    市场情报：
    - 欧洲SaaS市场年增长22%
    - 已有3个主要竞争对手
    - 平均客户获取成本：1200欧元
    - 平均交易规模：年费35000欧元
    """
    
    print("🎯 决策支持系统示例")
    print("=" * 50)
    
    # 综合决策分析
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
        
        print("⚖️ 决策分析:")
        print(f"推荐选项: {result.get('recommended_option')}")
        print(f"置信度: {result.get('confidence_score', 0)*100:.1f}%")
        
        print(f"\n📊 选项比较:")
        for option, analysis in result.get('option_analysis', {}).items():
            print(f"\n{option}:")
            print(f"  优点: {', '.join(analysis.get('pros', []))}")
            print(f"  缺点: {', '.join(analysis.get('cons', []))}")
            print(f"  风险分数: {analysis.get('risk_score', 'N/A')}/10")
            print(f"  预期ROI: {analysis.get('expected_roi', 'N/A')}")
        
        print(f"\n🎯 关键成功因素:")
        for factor in result.get('success_factors', []):
            print(f"  • {factor}")
        
        print(f"\n⚠️ 关键风险:")
        for risk in result.get('critical_risks', []):
            print(f"  • {risk['description']} (影响: {risk['impact']}, 概率: {risk['probability']})")
        
        print(f"\n📋 实施路线图:")
        for phase in result.get('implementation_plan', []):
            print(f"  阶段{phase['phase']}: {phase['description']}")
            print(f"    时间线: {phase['timeline']}")
            print(f"    资源: {phase['resources']}")
        
        print(f"\n📈 财务预测:")
        for projection in result.get('financial_projections', []):
            print(f"  • {projection}")
    
    else:
        print(f"❌ 决策分析失败: {decision_response.error}")

asyncio.run(decision_support_example())
```

## 高级配置示例

### 动态模型管理

```python
from llm_cooperation.tools import APIConfigManager

async def dynamic_model_management():
    """动态模型配置和管理示例"""
    
    config_manager = APIConfigManager()
    
    print("⚙️ 动态模型管理示例")
    print("=" * 50)
    
    # 添加多个API提供商
    providers = [
        {
            "name": "aigc_best",
            "base_url": "https://api2.aigcbest.top/v1",
            "api_key": "您的aigc密钥",
            "models": ["Qwen/Qwen3-32B", "Qwen/Qwen3-8B", "DeepSeek-V3"]
        },
        {
            "name": "openai", 
            "base_url": "https://api.openai.com/v1",
            "api_key": "您的openai密钥",
            "models": ["gpt-4", "gpt-3.5-turbo"]
        },
        {
            "name": "custom_provider",
            "base_url": "https://api.custom.com/v1", 
            "api_key": "您的自定义密钥",
            "models": ["custom-model-1", "custom-model-2"]
        }
    ]
    
    # 配置提供商
    for provider in providers:
        print(f"\n🔧 配置 {provider['name']}...")
        
        config_manager.add_endpoint_config(
            name=provider['name'],
            base_url=provider['base_url'],
            api_key=provider['api_key']
        )
        
        # 为此提供商添加模型
        for model in provider['models']:
            model_name = f"{provider['name']}_{model.split('/')[-1].lower().replace('-', '_')}"
            config_manager.add_model_config(
                model_name=model_name,
                model_path=model,
                provider=provider['name'],
                supported_tasks=["text", "chat", "completion"]
            )
            print(f"  ✅ 添加模型: {model_name} -> {model}")
    
    # 测试连接
    print(f"\n🔍 测试模型连接...")
    connectivity_results = await config_manager.test_model_connectivity()
    
    for model, result in connectivity_results.items():
        status = "✅" if result['accessible'] else "❌"
        latency = f"{result['latency']:.0f}ms" if result.get('latency') else "N/A"
        print(f"  {status} {model}: {latency}")
    
    # 性能基准测试
    print(f"\n📊 性能基准测试...")
    benchmark_query = "法国的首都是什么？"
    
    for model_name in ["aigc_best_qwen3_8b", "openai_gpt_3_5_turbo"]:
        if connectivity_results.get(model_name, {}).get('accessible'):
            start_time = time.time()
            try:
                # 模拟推理（实际场景中会使用真实引擎）
                await asyncio.sleep(0.1)  # 模拟响应时间
                end_time = time.time()
                print(f"  📈 {model_name}: {(end_time-start_time)*1000:.0f}ms")
            except Exception as e:
                print(f"  ❌ {model_name}: 错误 - {e}")
    
    # 动态负载均衡配置
    print(f"\n⚖️ 设置负载均衡...")
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
    print("  ✅ 负载均衡已配置")
    
    # 模型健康监控
    print(f"\n💓 健康监控设置...")
    health_config = {
        "check_interval": 60,  # 秒
        "failure_threshold": 3,
        "recovery_threshold": 2,
        "alert_webhooks": ["https://alerts.company.com/webhook"]
    }
    
    config_manager.setup_health_monitoring(health_config)
    print("  ✅ 健康监控已激活")

import time
asyncio.run(dynamic_model_management())
```

### 性能监控

```python
from llm_cooperation.monitoring import PerformanceMonitor
import matplotlib.pyplot as plt
import pandas as pd

async def performance_monitoring_example():
    """综合性能监控示例"""
    
    monitor = PerformanceMonitor()
    
    print("📊 性能监控示例")
    print("=" * 50)
    
    # 模拟各种请求进行监控
    test_scenarios = [
        {"type": "simple", "model": "qwen3_8b", "tokens": 100},
        {"type": "complex", "model": "qwen3_32b", "tokens": 500},
        {"type": "batch", "model": "qwen3_8b", "tokens": 200},
        {"type": "cooperation", "model": "multi", "tokens": 800},
    ]
    
    # 收集性能数据
    performance_data = []
    
    for scenario in test_scenarios * 10:  # 每种场景运行10次
        start_time = time.time()
        
        # 模拟请求处理
        processing_time = {
            "simple": 0.5 + random.uniform(-0.2, 0.3),
            "complex": 2.0 + random.uniform(-0.5, 1.0), 
            "batch": 1.2 + random.uniform(-0.3, 0.5),
            "cooperation": 3.5 + random.uniform(-1.0, 2.0)
        }[scenario["type"]]
        
        await asyncio.sleep(processing_time)
        end_time = time.time()
        
        # 记录指标
        metrics = {
            "timestamp": time.time(),
            "request_type": scenario["type"],
            "model": scenario["model"],
            "tokens": scenario["tokens"],
            "latency": end_time - start_time,
            "success": random.choice([True, True, True, False]),  # 75%成功率
            "memory_usage": random.uniform(100, 500),  # MB
            "cpu_usage": random.uniform(20, 80)  # %
        }
        
        performance_data.append(metrics)
        await monitor.record_metrics(metrics)
    
    # 生成性能报告
    print(f"\n📈 性能摘要:")
    df = pd.DataFrame(performance_data)
    
    # 总体统计
    print(f"总请求数: {len(df)}")
    print(f"成功率: {(df['success'].sum() / len(df) * 100):.1f}%")
    print(f"平均延迟: {df['latency'].mean():.2f}秒")
    print(f"95%延迟: {df['latency'].quantile(0.95):.2f}秒")
    
    # 按请求类型分组
    print(f"\n📊 按请求类型:")
    type_stats = df.groupby('request_type').agg({
        'latency': ['mean', 'std', 'count'],
        'success': 'mean',
        'tokens': 'mean'
    }).round(2)
    
    for req_type in type_stats.index:
        stats = type_stats.loc[req_type]
        print(f"  {req_type.upper()}:")
        print(f"    平均延迟: {stats[('latency', 'mean')]:.2f}秒 ± {stats[('latency', 'std')]:.2f}秒")
        print(f"    成功率: {stats[('success', 'mean')]*100:.1f}%")
        print(f"    请求数: {stats[('latency', 'count')]}")
    
    # 性能告警
    print(f"\n🚨 性能告警:")
    alerts = await monitor.check_performance_thresholds({
        "max_latency": 3.0,
        "min_success_rate": 0.90,
        "max_error_rate": 0.10
    })
    
    for alert in alerts:
        print(f"  ⚠️ {alert['severity'].upper()}: {alert['message']}")
    
    # 资源利用率
    print(f"\n💾 资源利用率:")
    print(f"平均内存使用: {df['memory_usage'].mean():.1f} MB")
    print(f"峰值内存使用: {df['memory_usage'].max():.1f} MB")
    print(f"平均CPU使用: {df['cpu_usage'].mean():.1f}%")
    print(f"峰值CPU使用: {df['cpu_usage'].max():.1f}%")
    
    # 趋势分析
    print(f"\n📈 趋势分析:")
    recent_data = df.tail(20)  # 最近20个请求
    older_data = df.head(20)   # 前20个请求
    
    latency_trend = recent_data['latency'].mean() - older_data['latency'].mean()
    success_trend = recent_data['success'].mean() - older_data['success'].mean()
    
    print(f"延迟趋势: {'↗️ +' if latency_trend > 0 else '↘️ '}{latency_trend:.2f}秒")
    print(f"成功率趋势: {'↗️ +' if success_trend > 0 else '↘️ '}{success_trend*100:.1f}%")

import random
import time
asyncio.run(performance_monitoring_example())
```

## 错误处理和容错示例

### 综合错误处理

```python
from llm_cooperation.exceptions import *
import logging

async def error_handling_example():
    """综合错误处理和容错示例"""
    
    print("🛡️ 错误处理和容错示例")
    print("=" * 50)
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = SystemConfig()
    engine = OpenAIEngine(config)
    
    # 可能失败的测试场景
    error_scenarios = [
        {
            "name": "无效模型",
            "request": InferenceRequest(
                prompt="测试", 
                model_name="nonexistent_model"
            ),
            "expected_error": ModelNotFoundError
        },
        {
            "name": "空提示", 
            "request": InferenceRequest(
                prompt="",
                model_name="qwen3_8b"
            ),
            "expected_error": ValidationError
        },
        {
            "name": "过度令牌请求",
            "request": InferenceRequest(
                prompt="测试" * 1000,
                model_name="qwen3_8b", 
                max_tokens=50000  # 不现实的数字
            ),
            "expected_error": APILimitError
        }
    ]
    
    try:
        await engine.initialize()
        
        for scenario in error_scenarios:
            print(f"\n🧪 测试: {scenario['name']}")
            print("-" * 30)
            
            try:
                response = await engine.inference(scenario['request'])
                
                if not response.success:
                    print(f"✅ 预期错误已优雅处理: {response.error}")
                else:
                    print(f"⚠️ 意外成功: {response.text[:50]}...")
                    
            except scenario['expected_error'] as e:
                print(f"✅ 捕获预期错误: {type(e).__name__}: {e}")
            except Exception as e:
                print(f"❌ 意外错误: {type(e).__name__}: {e}")
                logger.error(f"场景{scenario['name']}中的意外错误", exc_info=True)
        
        # 使用重试机制测试容错
        print(f"\n🔄 测试重试机制")
        print("-" * 30)
        
        from llm_cooperation.resilience import RetryHandler
        
        retry_handler = RetryHandler(
            max_retries=3,
            backoff_strategy="exponential",
            base_delay=1.0,
            max_delay=10.0
        )
        
        async def flaky_function():
            """模拟随机失败的函数"""
            if random.random() < 0.7:  # 70%失败率
                raise APIConnectionError("模拟网络错误")
            return "成功!"
        
        try:
            result = await retry_handler.execute_with_retry(flaky_function)
            print(f"✅ 重试机制成功: {result}")
        except Exception as e:
            print(f"❌ 重试机制耗尽: {e}")
        
        # 测试断路器
        print(f"\n⚡ 测试断路器")
        print("-" * 30)
        
        from llm_cooperation.resilience import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            expected_exception=APIConnectionError
        )
        
        async def unreliable_service():
            """模拟不可靠的外部服务"""
            if random.random() < 0.8:  # 80%失败率
                raise APIConnectionError("服务不可用")
            return "服务响应"
        
        # 测试断路器行为
        for i in range(10):
            try:
                result = await circuit_breaker.call(unreliable_service)
                print(f"  请求 {i+1}: ✅ {result}")
            except CircuitBreakerOpenError:
                print(f"  请求 {i+1}: 🚫 断路器开启")
            except APIConnectionError as e:
                print(f"  请求 {i+1}: ❌ {e}")
            
            await asyncio.sleep(0.5)  # 请求间短暂延迟
        
        # 测试优雅降级
        print(f"\n🎭 测试优雅降级")
        print("-" * 30)
        
        from llm_cooperation.resilience import GracefulDegradation
        
        degradation_handler = GracefulDegradation(
            fallback_models=["qwen3_8b", "backup_model"],
            cache_enabled=True,
            cache_ttl=300  # 5分钟
        )
        
        # 主模型失败，应该回退
        primary_request = InferenceRequest(
            prompt="什么是机器学习？",
            model_name="primary_model_that_fails"
        )
        
        try:
            response = await degradation_handler.handle_request(primary_request)
            print(f"✅ 优雅降级成功: 使用了 {response.model_name}")
            print(f"   响应: {response.text[:100]}...")
        except Exception as e:
            print(f"❌ 优雅降级失败: {e}")
    
    finally:
        await engine.shutdown()

asyncio.run(error_handling_example())
```

这个综合示例文档提供了LLM协作系统的实际、真实世界使用模式，涵盖从基础推理到高级企业应用的所有内容，包括错误处理和性能监控。

---

**语言选择**: [English](/) | [中文](/zh/)