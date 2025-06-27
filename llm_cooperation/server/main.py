"""
Main LLM Cooperation Server
Unified entry point for the distributed inference system
"""
import asyncio
import logging
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime

from llm_cooperation.config import SystemConfig
from llm_cooperation.engines.openai_engine import engine
from llm_cooperation.managers.model_manager import model_manager
from llm_cooperation.schedulers.cooperation_scheduler import cooperation_scheduler
from llm_cooperation.routing.intelligent_router import intelligent_router
from llm_cooperation.services.application_service import service_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_cooperation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query to process")
    preferences: Optional[Dict[str, Any]] = Field(default=None, description="User preferences for routing")
    user_id: Optional[str] = Field(default=None, description="User identifier")

class QueryResponse(BaseModel):
    result: str = Field(..., description="Processing result")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    processing_time: float = Field(..., description="Total processing time in seconds")
    strategy_used: str = Field(..., description="Routing strategy used")

class ServiceRequest(BaseModel):
    service_type: str = Field(..., description="Type of service requested")
    content: str = Field(..., description="Content to process")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Service parameters")
    user_id: Optional[str] = Field(default=None, description="User identifier")

class ServiceResponse(BaseModel):
    request_id: str = Field(..., description="Request identifier")
    result: Any = Field(..., description="Service result")
    success: bool = Field(..., description="Whether request was successful")
    processing_time: float = Field(..., description="Processing time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class SystemStatus(BaseModel):
    system_health: str = Field(..., description="Overall system health")
    active_models: int = Field(..., description="Number of active models")
    total_requests: int = Field(..., description="Total requests processed")
    uptime: str = Field(..., description="System uptime")
    resource_usage: Dict[str, Any] = Field(..., description="System resource usage")

# System startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Cooperation System...")
    
    try:
        # Initialize components in order
        await engine.initialize()
        await model_manager.initialize()
        
        logger.info("LLM Cooperation System started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down LLM Cooperation System...")
        
        await model_manager.shutdown()
        await engine.shutdown()
        
        logger.info("LLM Cooperation System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="LLM Cooperation System",
    description="Intelligent multi-model cooperation system with vLLM inference engine",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Cooperation System",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query with intelligent routing"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Processing query from user {request.user_id}")
        
        # Route query through intelligent router
        result = await intelligent_router.route_request(
            request.query, 
            request.preferences or {}
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Get routing statistics for metadata
        routing_stats = intelligent_router.get_routing_stats()
        
        return QueryResponse(
            result=result,
            metadata={
                "query_length": len(request.query.split()),
                "routing_stats": routing_stats,
                "timestamp": datetime.now().isoformat()
            },
            processing_time=processing_time,
            strategy_used="intelligent_routing"
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/service", response_model=ServiceResponse)
async def process_service_request(request: ServiceRequest):
    """Process application service request"""
    try:
        logger.info(f"Processing {request.service_type} service request")
        
        response = await service_manager.process_request(
            service_type=request.service_type,
            content=request.content,
            parameters=request.parameters or {},
            user_id=request.user_id
        )
        
        return ServiceResponse(
            request_id=response.request_id,
            result=response.result,
            success=response.success,
            processing_time=response.processing_time,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Service request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get model status
        model_status = model_manager.get_system_status()
        ready_models = sum(1 for model_info in model_status["models"].values() 
                          if model_info["status"] == "ready")
        
        # Get system resources (for OpenAI API, this will be limited)
        resources = await engine.get_system_metrics()
        
        # Get routing statistics
        routing_stats = intelligent_router.get_routing_stats()
        
        # Get service statistics
        service_stats = service_manager.get_service_stats()
        
        # Determine system health
        model_health = ready_models / len(SystemConfig.MODELS) if SystemConfig.MODELS else 0
        
        if model_health >= 0.8:
            health_status = "healthy"
        elif model_health >= 0.5:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return SystemStatus(
            system_health=health_status,
            active_models=ready_models,
            total_requests=routing_stats.get("total_requests", 0) + service_stats.get("total_requests", 0),
            uptime=f"{asyncio.get_event_loop().time():.0f}s",
            resource_usage={
                "total_models": resources.get("total_models", 0),
                "available_models": resources.get("available_models", 0),
                "healthy_models": resources.get("healthy_models", 0),
                "health_percentage": resources.get("health_percentage", 0),
                "api_endpoints": resources.get("api_endpoints", [])
            }
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_model_status():
    """Get detailed model status"""
    try:
        return model_manager.get_system_status()
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routing/stats")
async def get_routing_stats():
    """Get routing system statistics"""
    try:
        return intelligent_router.get_routing_stats()
    except Exception as e:
        logger.error(f"Routing stats check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services/stats")
async def get_service_stats():
    """Get application service statistics"""
    try:
        return service_manager.get_service_stats()
    except Exception as e:
        logger.error(f"Service stats check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cooperation/stats")
async def get_cooperation_stats():
    """Get cooperation system statistics"""
    try:
        return cooperation_scheduler.get_system_stats()
    except Exception as e:
        logger.error(f"Cooperation stats check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cooperation/task")
async def create_cooperation_task(
    query: str,
    mode: str = "sequential",
    models: Optional[list] = None,
    integration_strategy: str = "concatenate"
):
    """Create a specific cooperation task"""
    try:
        if models is None:
            models = ["qwen3_32b_reasoning", "qwen2_5_7b"]
        
        if mode == "sequential":
            result = await cooperation_scheduler.create_sequential_task(
                query, models, integration_strategy
            )
        elif mode == "parallel":
            result = await cooperation_scheduler.create_parallel_task(
                query, models, integration_strategy
            )
        elif mode == "voting":
            result = await cooperation_scheduler.create_voting_task(query, models)
        else:
            raise ValueError(f"Unsupported cooperation mode: {mode}")
        
        return {"result": result, "mode": mode, "models": models}
        
    except Exception as e:
        logger.error(f"Cooperation task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

# Main server function
async def main():
    """Main server function"""
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())