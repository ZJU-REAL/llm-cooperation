"""
Intelligent Routing Layer
Analyzes user requests and determines optimal processing strategies
"""
import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

from llm_cooperation.config import SystemConfig
from vllm_engine import InferenceRequest, InferenceResponse
from llm_cooperation.managers.model_manager import model_manager
from llm_cooperation.schedulers.cooperation_scheduler import cooperation_scheduler, CooperationMode

class TaskType(Enum):
    SIMPLE_QA = "simple_qa"
    COMPLEX_REASONING = "complex_reasoning"
    MATH_CALCULATION = "math_calculation"
    CODE_ANALYSIS = "code_analysis"
    MULTIMODAL = "multimodal"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"

class ComplexityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4

@dataclass
class RequestAnalysis:
    """Analysis of user request"""
    task_type: TaskType
    complexity_level: ComplexityLevel
    requires_multimodal: bool
    estimated_tokens: int
    keywords: List[str]
    confidence: float
    suggested_models: List[str]
    cooperation_recommended: bool
    cooperation_mode: Optional[CooperationMode] = None

@dataclass
class RoutingDecision:
    """Final routing decision"""
    strategy: str  # single_model, sequential, parallel, voting, pipeline
    models: List[str]
    cooperation_config: Optional[Dict[str, Any]] = None
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    confidence: float = 0.0

class RequestAnalyzer:
    """Analyzes user requests to determine task characteristics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern matching for different task types
        self.patterns = {
            TaskType.MATH_CALCULATION: [
                r'\b(calculate|compute|solve|equation|formula|integral|derivative)\b',
                r'\b\d+[\+\-\*/]\d+\b',
                r'\b(sin|cos|tan|log|sqrt|factorial)\b',
                r'\b(prove|theorem|lemma)\b'
            ],
            TaskType.CODE_ANALYSIS: [
                r'\b(code|python|javascript|java|c\+\+|function|class|algorithm)\b',
                r'\b(debug|error|exception|syntax)\b',
                r'\b(implement|refactor|optimize)\b',
                r'```[\w]*\n.*```'
            ],
            TaskType.MULTIMODAL: [
                r'\b(image|picture|photo|chart|graph|diagram)\b',
                r'\b(visual|see|look|analyze.*image)\b',
                r'\b(OCR|text.*image|read.*image)\b'
            ],
            TaskType.CREATIVE_WRITING: [
                r'\b(write|story|poem|essay|creative|fiction)\b',
                r'\b(character|plot|narrative|dialogue)\b',
                r'\b(novel|screenplay|lyrics)\b'
            ],
            TaskType.TRANSLATION: [
                r'\b(translate|translation|language)\b',
                r'\b(chinese|english|spanish|french|german|japanese)\b',
                r'\b(from.*to|into.*language)\b'
            ],
            TaskType.SUMMARIZATION: [
                r'\b(summarize|summary|brief|overview)\b',
                r'\b(key points|main ideas|condensed)\b',
                r'\b(tl;dr|executive summary)\b'
            ],
            TaskType.COMPARISON: [
                r'\b(compare|comparison|versus|vs|difference)\b',
                r'\b(pros and cons|advantages|disadvantages)\b',
                r'\b(better|worse|prefer)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            ComplexityLevel.LOW: [
                r'\b(what|when|where|who|yes|no)\b',
                r'\b(simple|basic|quick)\b'
            ],
            ComplexityLevel.MEDIUM: [
                r'\b(how|why|explain|describe)\b',
                r'\b(analysis|process|method)\b'
            ],
            ComplexityLevel.HIGH: [
                r'\b(complex|detailed|comprehensive)\b',
                r'\b(multi-step|elaborate|thorough)\b',
                r'\b(research|investigation|deep)\b'
            ],
            ComplexityLevel.VERY_HIGH: [
                r'\b(advanced|expert|professional)\b',
                r'\b(multiple|various|several)\b.*\b(aspects|factors|considerations)\b',
                r'\b\d+.*steps?\b.*\b\d+.*stages?\b'
            ]
        }
    
    async def analyze_request(self, user_query: str) -> RequestAnalysis:
        """Analyze user request and determine characteristics"""
        query_lower = user_query.lower()
        
        # Detect task type
        task_type = self._detect_task_type(query_lower)
        
        # Assess complexity
        complexity_level = self._assess_complexity(query_lower, len(user_query.split()))
        
        # Check for multimodal requirements
        requires_multimodal = self._check_multimodal_requirements(query_lower)
        
        # Estimate token count
        estimated_tokens = len(user_query.split()) * 1.3  # Rough estimate
        
        # Extract keywords
        keywords = self._extract_keywords(user_query)
        
        # Determine model suggestions
        suggested_models = await self._suggest_models(task_type, complexity_level, requires_multimodal)
        
        # Determine if cooperation is recommended
        cooperation_recommended, cooperation_mode = self._assess_cooperation_need(
            task_type, complexity_level, len(user_query.split())
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(task_type, complexity_level)
        
        return RequestAnalysis(
            task_type=task_type,
            complexity_level=complexity_level,
            requires_multimodal=requires_multimodal,
            estimated_tokens=int(estimated_tokens),
            keywords=keywords,
            confidence=confidence,
            suggested_models=suggested_models,
            cooperation_recommended=cooperation_recommended,
            cooperation_mode=cooperation_mode
        )
    
    def _detect_task_type(self, query: str) -> TaskType:
        """Detect the primary task type from query"""
        scores = {}
        
        for task_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            scores[task_type] = score
        
        # Return task type with highest score, default to ANALYSIS
        if scores:
            best_task = max(scores, key=scores.get)
            if scores[best_task] > 0:
                return best_task
        
        return TaskType.ANALYSIS
    
    def _assess_complexity(self, query: str, word_count: int) -> ComplexityLevel:
        """Assess complexity level of the request"""
        # Start with word count based assessment
        if word_count < 10:
            base_complexity = ComplexityLevel.LOW
        elif word_count < 30:
            base_complexity = ComplexityLevel.MEDIUM
        elif word_count < 100:
            base_complexity = ComplexityLevel.HIGH
        else:
            base_complexity = ComplexityLevel.VERY_HIGH
        
        # Adjust based on complexity indicators
        complexity_scores = {}
        for level, patterns in self.complexity_indicators.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            complexity_scores[level] = score
        
        # If we have strong indicators, use them
        if complexity_scores:
            max_score = max(complexity_scores.values())
            if max_score > 0:
                indicated_complexity = max(complexity_scores, key=complexity_scores.get)
                # Take the higher of base and indicated complexity
                return ComplexityLevel(max(base_complexity.value, indicated_complexity.value))
        
        return base_complexity
    
    def _check_multimodal_requirements(self, query: str) -> bool:
        """Check if request requires multimodal processing"""
        multimodal_patterns = self.patterns.get(TaskType.MULTIMODAL, [])
        for pattern in multimodal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return top 10 most relevant keywords
        return keywords[:10]
    
    async def _suggest_models(self, task_type: TaskType, complexity_level: ComplexityLevel, requires_multimodal: bool) -> List[str]:
        """Suggest appropriate models for the task"""
        if requires_multimodal:
            return ["qwen2_5_vl_72b"]
        
        if task_type in [TaskType.MATH_CALCULATION, TaskType.CODE_ANALYSIS, TaskType.COMPLEX_REASONING]:
            return ["qwen3_32b_reasoning"]
        
        if complexity_level in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]:
            return ["qwen2_5_7b"]
        
        if complexity_level == ComplexityLevel.HIGH:
            return ["qwen3_32b_reasoning", "qwen2_5_7b"]
        
        # Very high complexity - recommend multiple models
        return ["qwen3_32b_reasoning", "qwen3_32b_router", "qwen2_5_7b"]
    
    def _assess_cooperation_need(self, task_type: TaskType, complexity_level: ComplexityLevel, word_count: int) -> Tuple[bool, Optional[CooperationMode]]:
        """Assess if cooperation between models is needed"""
        # High complexity tasks benefit from cooperation
        if complexity_level == ComplexityLevel.VERY_HIGH:
            return True, CooperationMode.SEQUENTIAL
        
        # Multi-faceted tasks benefit from parallel processing
        if task_type == TaskType.COMPARISON:
            return True, CooperationMode.PARALLEL
        
        # Long queries might benefit from pipeline processing
        if word_count > 100:
            return True, CooperationMode.PIPELINE
        
        # Complex reasoning tasks benefit from voting
        if task_type in [TaskType.MATH_CALCULATION, TaskType.COMPLEX_REASONING] and complexity_level == ComplexityLevel.HIGH:
            return True, CooperationMode.VOTING
        
        return False, None
    
    def _calculate_confidence(self, task_type: TaskType, complexity_level: ComplexityLevel) -> float:
        """Calculate confidence in the analysis"""
        # Base confidence
        confidence = 0.7
        
        # Higher confidence for clear task types
        if task_type in [TaskType.MATH_CALCULATION, TaskType.CODE_ANALYSIS, TaskType.MULTIMODAL]:
            confidence += 0.2
        
        # Adjust for complexity assessment confidence
        if complexity_level in [ComplexityLevel.LOW, ComplexityLevel.VERY_HIGH]:
            confidence += 0.1  # Extremes are easier to detect
        
        return min(confidence, 1.0)

class IntelligentRouter:
    """Main intelligent routing system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = RequestAnalyzer()
        self.routing_cache: Dict[str, RoutingDecision] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
    async def route_request(self, user_query: str, user_preferences: Dict[str, Any] = None) -> str:
        """Route user request to optimal processing strategy"""
        start_time = time.time()
        
        try:
            # Check cache first
            query_hash = hashlib.md5(user_query.encode()).hexdigest()
            if query_hash in self.routing_cache:
                self.logger.info("Using cached routing decision")
                decision = self.routing_cache[query_hash]
            else:
                # Analyze request
                analysis = await self.analyzer.analyze_request(user_query)
                self.logger.info(f"Request analysis: {analysis.task_type.value}, complexity: {analysis.complexity_level.value}")
                
                # Make routing decision
                decision = await self._make_routing_decision(analysis, user_preferences or {})
                
                # Cache decision
                self.routing_cache[query_hash] = decision
            
            # Execute routing strategy
            result = await self._execute_routing_strategy(user_query, decision)
            
            # Record performance
            self._record_performance(user_query, decision, time.time() - start_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            self._record_performance(user_query, None, time.time() - start_time, False)
            
            # Fallback to simple routing
            return await self._fallback_routing(user_query)
    
    async def _make_routing_decision(self, analysis: RequestAnalysis, user_preferences: Dict[str, Any]) -> RoutingDecision:
        """Make optimal routing decision based on analysis"""
        # Get user preferences
        preferred_strategy = user_preferences.get("strategy", "auto")
        max_latency = user_preferences.get("max_latency", 60.0)
        quality_priority = user_preferences.get("quality_priority", 0.7)  # 0-1 scale
        
        # Determine strategy
        if preferred_strategy != "auto":
            strategy = preferred_strategy
        elif analysis.cooperation_recommended:
            strategy = analysis.cooperation_mode.value
        else:
            strategy = "single_model"
        
        # Select models
        models = analysis.suggested_models
        if not models:
            models = ["qwen2_5_7b"]  # Fallback
        
        # Get model recommendations with performance scores
        if len(models) > 1 and strategy == "single_model":
            # Choose best model based on current performance
            recommendations = await model_manager.get_model_recommendations(analysis.task_type.value)
            if recommendations:
                best_model = recommendations[0][0]
                if best_model in models:
                    models = [best_model]
        
        # Configure cooperation if needed
        cooperation_config = None
        if strategy in ["sequential", "parallel", "voting", "pipeline"]:
            cooperation_config = {
                "integration_strategy": self._select_integration_strategy(analysis, quality_priority),
                "timeout": max_latency,
                "task_type": analysis.task_type.value
            }
        
        # Estimate cost and latency
        estimated_cost = self._estimate_cost(models, strategy, analysis.estimated_tokens)
        estimated_latency = await self._estimate_latency(models, strategy)
        
        return RoutingDecision(
            strategy=strategy,
            models=models,
            cooperation_config=cooperation_config,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            confidence=analysis.confidence
        )
    
    def _select_integration_strategy(self, analysis: RequestAnalysis, quality_priority: float) -> str:
        """Select appropriate integration strategy"""
        if quality_priority > 0.8:
            return "ensemble"  # Highest quality
        elif analysis.task_type in [TaskType.MATH_CALCULATION, TaskType.CODE_ANALYSIS]:
            return "voting"  # Need consensus for correctness
        else:
            return "concatenate"  # Simple and fast
    
    def _estimate_cost(self, models: List[str], strategy: str, estimated_tokens: int) -> float:
        """Estimate computational cost"""
        # Simple cost model (can be made more sophisticated)
        base_cost = {
            "qwen3_32b_router": 0.5,
            "qwen3_32b_reasoning": 1.0,
            "qwen2_5_vl_72b": 2.0,
            "qwen2_5_7b": 0.2
        }
        
        total_cost = 0.0
        for model in models:
            model_cost = base_cost.get(model, 0.5)
            total_cost += model_cost * (estimated_tokens / 1000)
        
        # Strategy overhead
        if strategy in ["sequential", "pipeline"]:
            total_cost *= 1.1  # 10% overhead
        elif strategy in ["parallel", "voting"]:
            total_cost *= len(models)  # Run multiple models
        
        return total_cost
    
    async def _estimate_latency(self, models: List[str], strategy: str) -> float:
        """Estimate processing latency"""
        model_latencies = {}
        
        # Get current model performance
        for model in models:
            metrics = model_manager.model_metrics.get(model)
            if metrics:
                model_latencies[model] = metrics.avg_latency
            else:
                # Default estimates
                defaults = {
                    "qwen3_32b_router": 2.0,
                    "qwen3_32b_reasoning": 8.0,
                    "qwen2_5_vl_72b": 15.0,
                    "qwen2_5_7b": 1.0
                }
                model_latencies[model] = defaults.get(model, 5.0)
        
        if strategy == "single_model":
            return model_latencies[models[0]]
        elif strategy == "sequential":
            return sum(model_latencies[model] for model in models)
        elif strategy in ["parallel", "voting"]:
            return max(model_latencies[model] for model in models) * 1.2  # Coordination overhead
        else:
            return sum(model_latencies[model] for model in models) * 1.1
    
    async def _execute_routing_strategy(self, user_query: str, decision: RoutingDecision) -> str:
        """Execute the routing strategy"""
        self.logger.info(f"Executing {decision.strategy} strategy with models: {decision.models}")
        
        if decision.strategy == "single_model":
            return await self._execute_single_model(user_query, decision.models[0])
        
        elif decision.strategy == "sequential":
            return await cooperation_scheduler.create_sequential_task(
                user_query, 
                decision.models,
                decision.cooperation_config.get("integration_strategy", "concatenate")
            )
        
        elif decision.strategy == "parallel":
            return await cooperation_scheduler.create_parallel_task(
                user_query,
                decision.models,
                decision.cooperation_config.get("integration_strategy", "ensemble")
            )
        
        elif decision.strategy == "voting":
            return await cooperation_scheduler.create_voting_task(
                user_query,
                decision.models
            )
        
        elif decision.strategy == "pipeline":
            # Create simple pipeline configuration
            pipeline_config = [
                {"model": model, "prompt_template": user_query if i == 0 else "{previous_result}"}
                for i, model in enumerate(decision.models)
            ]
            return await cooperation_scheduler.create_pipeline_task(user_query, pipeline_config)
        
        else:
            raise ValueError(f"Unknown strategy: {decision.strategy}")
    
    async def _execute_single_model(self, user_query: str, model_name: str) -> str:
        """Execute single model inference"""
        request = InferenceRequest(
            prompt=user_query,
            model_name=model_name,
            max_tokens=2048
        )
        
        response = await model_manager.execute_request(request)
        return response.text if response.success else f"Error: {response.error}"
    
    async def _fallback_routing(self, user_query: str) -> str:
        """Fallback routing strategy"""
        self.logger.info("Using fallback routing")
        return await self._execute_single_model(user_query, "qwen2_5_7b")
    
    def _record_performance(self, query: str, decision: Optional[RoutingDecision], 
                          latency: float, success: bool):
        """Record routing performance for learning"""
        record = {
            "timestamp": time.time(),
            "query_length": len(query.split()),
            "strategy": decision.strategy if decision else "fallback",
            "models": decision.models if decision else ["qwen2_5_7b"],
            "latency": latency,
            "success": success,
            "estimated_latency": decision.estimated_latency if decision else 0.0
        }
        
        self.performance_history.append(record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing system statistics"""
        if not self.performance_history:
            return {"total_requests": 0}
        
        total_requests = len(self.performance_history)
        successful_requests = sum(1 for r in self.performance_history if r["success"])
        
        avg_latency = sum(r["latency"] for r in self.performance_history) / total_requests
        
        strategy_usage = {}
        for record in self.performance_history:
            strategy = record["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / total_requests,
            "average_latency": avg_latency,
            "strategy_usage": strategy_usage,
            "cache_size": len(self.routing_cache)
        }

# Global router instance
intelligent_router = IntelligentRouter()