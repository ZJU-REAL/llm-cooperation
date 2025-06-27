"""
Application Service Layer
Provides high-level intelligent analysis services for industry applications
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

from llm_cooperation.routing.intelligent_router import intelligent_router
from llm_cooperation.config import SystemConfig

@dataclass
class ServiceRequest:
    """Service request structure"""
    request_id: str
    service_type: str
    content: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ServiceResponse:
    """Service response structure"""
    request_id: str
    service_type: str
    result: Any
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class BaseService(ABC):
    """Abstract base class for application services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
    
    @abstractmethod
    async def process(self, request: ServiceRequest) -> ServiceResponse:
        """Process service request"""
        pass
    
    def _create_response(self, request: ServiceRequest, result: Any, 
                        processing_time: float, metadata: Dict = None) -> ServiceResponse:
        """Create standardized service response"""
        return ServiceResponse(
            request_id=request.request_id,
            service_type=request.service_type,
            result=result,
            metadata=metadata or {},
            processing_time=processing_time,
            success=True
        )
    
    def _create_error_response(self, request: ServiceRequest, error: str, 
                             processing_time: float) -> ServiceResponse:
        """Create error response"""
        return ServiceResponse(
            request_id=request.request_id,
            service_type=request.service_type,
            result=None,
            metadata={},
            processing_time=processing_time,
            success=False,
            error=error
        )

class DocumentAnalysisService(BaseService):
    """Document analysis and understanding service"""
    
    def __init__(self):
        super().__init__("document_analysis")
    
    async def process(self, request: ServiceRequest) -> ServiceResponse:
        start_time = time.time()
        
        try:
            analysis_type = request.parameters.get("analysis_type", "comprehensive")
            document_type = request.parameters.get("document_type", "text")
            
            if document_type == "text":
                result = await self._analyze_text_document(request.content, analysis_type)
            elif document_type == "structured":
                result = await self._analyze_structured_document(request.content, analysis_type)
            else:
                raise ValueError(f"Unsupported document type: {document_type}")
            
            processing_time = time.time() - start_time
            return self._create_response(request, result, processing_time, 
                                       {"analysis_type": analysis_type, "document_type": document_type})
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(request, str(e), processing_time)
    
    async def _analyze_text_document(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze text document"""
        if analysis_type == "comprehensive":
            prompt = f"""
Please provide a comprehensive analysis of the following document:

{content}

Analysis should include:
1. Summary of main points
2. Key themes and topics
3. Document structure and organization
4. Important entities and relationships
5. Conclusions and insights
6. Potential questions or areas needing clarification

Please format your analysis in a structured manner.
"""
        elif analysis_type == "summary":
            prompt = f"""
Please provide a concise summary of the following document:

{content}

Focus on the main points, key conclusions, and essential information.
"""
        elif analysis_type == "entities":
            prompt = f"""
Please extract and categorize all important entities from the following document:

{content}

Categorize entities into: People, Organizations, Locations, Dates, Numbers, Concepts, etc.
"""
        else:
            prompt = f"Analyze the following document with focus on {analysis_type}:\n\n{content}"
        
        result = await intelligent_router.route_request(prompt, {"quality_priority": 0.8})
        
        return {
            "analysis": result,
            "document_length": len(content.split()),
            "analysis_type": analysis_type
        }
    
    async def _analyze_structured_document(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze structured document (JSON, CSV, etc.)"""
        prompt = f"""
Please analyze the following structured data:

{content}

Provide insights about:
1. Data structure and schema
2. Key patterns and trends
3. Statistical summaries
4. Data quality assessment
5. Business insights and recommendations

Format your analysis clearly and professionally.
"""
        
        result = await intelligent_router.route_request(prompt, {"quality_priority": 0.8})
        
        return {
            "analysis": result,
            "data_structure": "structured",
            "analysis_type": analysis_type
        }

class DataInsightService(BaseService):
    """Data analysis and insight generation service"""
    
    def __init__(self):
        super().__init__("data_insight")
    
    async def process(self, request: ServiceRequest) -> ServiceResponse:
        start_time = time.time()
        
        try:
            insight_type = request.parameters.get("insight_type", "trends")
            data_format = request.parameters.get("data_format", "text")
            
            if insight_type == "trends":
                result = await self._analyze_trends(request.content, data_format)
            elif insight_type == "patterns":
                result = await self._analyze_patterns(request.content, data_format)
            elif insight_type == "anomalies":
                result = await self._detect_anomalies(request.content, data_format)
            elif insight_type == "predictions":
                result = await self._generate_predictions(request.content, data_format)
            else:
                result = await self._general_analysis(request.content, insight_type)
            
            processing_time = time.time() - start_time
            return self._create_response(request, result, processing_time,
                                       {"insight_type": insight_type, "data_format": data_format})
            
        except Exception as e:
            self.logger.error(f"Data insight generation failed: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(request, str(e), processing_time)
    
    async def _analyze_trends(self, data: str, data_format: str) -> Dict[str, Any]:
        """Analyze trends in data"""
        prompt = f"""
Analyze the following data for trends and patterns:

{data}

Please provide:
1. Identified trends (increasing, decreasing, cyclical, etc.)
2. Statistical significance of trends
3. Time-based patterns if applicable
4. Potential causes or drivers
5. Future implications
6. Recommendations based on trends

Be specific and provide quantitative insights where possible.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "sequential"})
        
        return {
            "trend_analysis": result,
            "data_points": len(data.split('\n')),
            "analysis_method": "comprehensive_trend_analysis"
        }
    
    async def _analyze_patterns(self, data: str, data_format: str) -> Dict[str, Any]:
        """Analyze patterns in data"""
        prompt = f"""
Identify and analyze patterns in the following data:

{data}

Focus on:
1. Recurring patterns and cycles
2. Correlations between variables
3. Seasonal or temporal patterns
4. Structural patterns in the data
5. Hidden relationships
6. Pattern reliability and confidence

Provide detailed analysis with examples.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "parallel"})
        
        return {
            "pattern_analysis": result,
            "pattern_types": ["temporal", "structural", "correlational"],
            "confidence_level": "high"
        }
    
    async def _detect_anomalies(self, data: str, data_format: str) -> Dict[str, Any]:
        """Detect anomalies in data"""
        prompt = f"""
Analyze the following data to detect anomalies and outliers:

{data}

Identify:
1. Statistical outliers
2. Unexpected patterns or behaviors
3. Data quality issues
4. Potential errors or inconsistencies
5. Significance of each anomaly
6. Possible explanations for anomalies

Provide detailed findings with reasoning.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "voting"})
        
        return {
            "anomaly_detection": result,
            "detection_method": "statistical_and_pattern_based",
            "severity_levels": ["low", "medium", "high"]
        }
    
    async def _generate_predictions(self, data: str, data_format: str) -> Dict[str, Any]:
        """Generate predictions based on data"""
        prompt = f"""
Based on the following historical data, generate predictions and forecasts:

{data}

Provide:
1. Short-term predictions (next period)
2. Medium-term forecasts (next few periods)
3. Long-term trends and projections
4. Confidence intervals where applicable
5. Key assumptions in predictions
6. Risk factors and uncertainties
7. Scenario analysis (best/worst/likely cases)

Be specific about timeframes and confidence levels.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "ensemble", "quality_priority": 0.9})
        
        return {
            "predictions": result,
            "prediction_horizon": "multi_term",
            "methodology": "trend_and_pattern_based"
        }
    
    async def _general_analysis(self, data: str, insight_type: str) -> Dict[str, Any]:
        """General data analysis"""
        prompt = f"""
Perform {insight_type} analysis on the following data:

{data}

Provide comprehensive insights and actionable recommendations.
"""
        
        result = await intelligent_router.route_request(prompt)
        
        return {
            "analysis": result,
            "analysis_type": insight_type
        }

class DecisionSupportService(BaseService):
    """Decision support and recommendation service"""
    
    def __init__(self):
        super().__init__("decision_support")
    
    async def process(self, request: ServiceRequest) -> ServiceResponse:
        start_time = time.time()
        
        try:
            decision_type = request.parameters.get("decision_type", "recommendation")
            context = request.parameters.get("context", {})
            
            if decision_type == "recommendation":
                result = await self._generate_recommendations(request.content, context)
            elif decision_type == "comparison":
                result = await self._compare_options(request.content, context)
            elif decision_type == "risk_assessment":
                result = await self._assess_risks(request.content, context)
            elif decision_type == "strategy":
                result = await self._develop_strategy(request.content, context)
            else:
                result = await self._general_decision_support(request.content, decision_type, context)
            
            processing_time = time.time() - start_time
            return self._create_response(request, result, processing_time,
                                       {"decision_type": decision_type, "context_provided": bool(context)})
            
        except Exception as e:
            self.logger.error(f"Decision support failed: {e}")
            processing_time = time.time() - start_time
            return self._create_error_response(request, str(e), processing_time)
    
    async def _generate_recommendations(self, problem: str, context: Dict) -> Dict[str, Any]:
        """Generate recommendations for decision making"""
        context_str = json.dumps(context, indent=2) if context else "No additional context provided"
        
        prompt = f"""
Problem/Situation:
{problem}

Context:
{context_str}

Please provide comprehensive recommendations including:
1. Multiple viable options/solutions
2. Pros and cons of each option
3. Implementation considerations
4. Resource requirements
5. Timeline and milestones
6. Risk factors and mitigation strategies
7. Success metrics and KPIs
8. Final recommendation with reasoning

Structure your response clearly and provide actionable advice.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "sequential", "quality_priority": 0.9})
        
        return {
            "recommendations": result,
            "methodology": "comprehensive_analysis",
            "confidence": "high"
        }
    
    async def _compare_options(self, options_description: str, context: Dict) -> Dict[str, Any]:
        """Compare multiple options for decision making"""
        context_str = json.dumps(context, indent=2) if context else "No additional context provided"
        
        prompt = f"""
Options to Compare:
{options_description}

Context:
{context_str}

Please provide a detailed comparison including:
1. Feature-by-feature comparison
2. Advantages and disadvantages of each option
3. Cost-benefit analysis
4. Risk assessment for each option
5. Suitability for different scenarios
6. Implementation complexity
7. Long-term implications
8. Ranking and final recommendation

Use structured format with clear criteria and scoring.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "parallel", "quality_priority": 0.8})
        
        return {
            "comparison": result,
            "comparison_type": "multi_criteria",
            "evaluation_framework": "comprehensive"
        }
    
    async def _assess_risks(self, situation: str, context: Dict) -> Dict[str, Any]:
        """Assess risks in decision making"""
        context_str = json.dumps(context, indent=2) if context else "No additional context provided"
        
        prompt = f"""
Situation for Risk Assessment:
{situation}

Context:
{context_str}

Please conduct a comprehensive risk assessment including:
1. Identification of all potential risks
2. Risk categorization (operational, financial, strategic, etc.)
3. Probability assessment for each risk
4. Impact analysis (severity and scope)
5. Risk matrix (probability vs impact)
6. Mitigation strategies for each risk
7. Contingency planning
8. Risk monitoring and review recommendations

Provide both qualitative and quantitative analysis where possible.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "voting", "quality_priority": 0.9})
        
        return {
            "risk_assessment": result,
            "assessment_framework": "probability_impact_matrix",
            "risk_categories": ["operational", "financial", "strategic", "compliance"]
        }
    
    async def _develop_strategy(self, objective: str, context: Dict) -> Dict[str, Any]:
        """Develop strategic recommendations"""
        context_str = json.dumps(context, indent=2) if context else "No additional context provided"
        
        prompt = f"""
Strategic Objective:
{objective}

Context:
{context_str}

Please develop a comprehensive strategy including:
1. Situation analysis (current state, challenges, opportunities)
2. Strategic options and alternatives
3. Recommended strategic approach
4. Implementation roadmap with phases
5. Resource allocation requirements
6. Timeline and milestones
7. Success metrics and KPIs
8. Risk management plan
9. Monitoring and adjustment mechanisms

Provide a detailed, actionable strategic plan.
"""
        
        result = await intelligent_router.route_request(prompt, {"strategy": "ensemble", "quality_priority": 0.95})
        
        return {
            "strategy": result,
            "strategic_framework": "comprehensive_planning",
            "planning_horizon": "multi_phase"
        }
    
    async def _general_decision_support(self, problem: str, decision_type: str, context: Dict) -> Dict[str, Any]:
        """General decision support"""
        context_str = json.dumps(context, indent=2) if context else "No additional context provided"
        
        prompt = f"""
Decision Support Request ({decision_type}):
{problem}

Context:
{context_str}

Please provide comprehensive decision support with analysis, options, and recommendations.
"""
        
        result = await intelligent_router.route_request(prompt, {"quality_priority": 0.8})
        
        return {
            "decision_support": result,
            "support_type": decision_type
        }

class ApplicationServiceManager:
    """Main application service manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services = {
            "document_analysis": DocumentAnalysisService(),
            "data_insight": DataInsightService(),
            "decision_support": DecisionSupportService()
        }
        self.request_history: List[ServiceRequest] = []
        self.response_history: List[ServiceResponse] = []
    
    async def process_request(self, service_type: str, content: str, 
                            parameters: Dict[str, Any] = None,
                            user_id: str = None) -> ServiceResponse:
        """Process service request"""
        request_id = str(uuid.uuid4())
        
        request = ServiceRequest(
            request_id=request_id,
            service_type=service_type,
            content=content,
            parameters=parameters or {},
            user_id=user_id
        )
        
        self.request_history.append(request)
        
        try:
            service = self.services.get(service_type)
            if not service:
                raise ValueError(f"Unknown service type: {service_type}")
            
            self.logger.info(f"Processing {service_type} request {request_id}")
            response = await service.process(request)
            
            self.response_history.append(response)
            self.logger.info(f"Completed {service_type} request {request_id} in {response.processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Service request {request_id} failed: {e}")
            error_response = ServiceResponse(
                request_id=request_id,
                service_type=service_type,
                result=None,
                metadata={},
                processing_time=0.0,
                success=False,
                error=str(e)
            )
            self.response_history.append(error_response)
            return error_response
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        total_requests = len(self.request_history)
        successful_requests = sum(1 for r in self.response_history if r.success)
        
        if total_requests == 0:
            return {"total_requests": 0}
        
        service_usage = {}
        for request in self.request_history:
            service = request.service_type
            service_usage[service] = service_usage.get(service, 0) + 1
        
        avg_processing_time = sum(r.processing_time for r in self.response_history) / len(self.response_history)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests,
            "service_usage": service_usage,
            "average_processing_time": avg_processing_time,
            "available_services": list(self.services.keys())
        }
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent request history"""
        recent_requests = self.request_history[-limit:] if limit else self.request_history
        return [asdict(req) for req in recent_requests]
    
    def get_response_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent response history"""
        recent_responses = self.response_history[-limit:] if limit else self.response_history
        return [asdict(resp) for resp in recent_responses]

# Global service manager instance
service_manager = ApplicationServiceManager()