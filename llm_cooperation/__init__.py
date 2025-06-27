"""
LLM Cooperation System

A comprehensive system for intelligent multi-model routing and coordination 
using OpenAI-compatible APIs with cooperative processing capabilities.
"""

__version__ = "1.0.0"
__author__ = "LLM Cooperation Team"
__email__ = "team@llmcooperation.com"
__description__ = "Intelligent multi-model LLM cooperation system"

# Core components
from .config import SystemConfig, ModelConfig, OpenAIAPIConfig
from .engines.openai_engine import OpenAIEngine, InferenceRequest, InferenceResponse
from .managers.model_manager import ModelResourceManager, ModelStatus, ModelMetrics
from .schedulers.cooperation_scheduler import CooperationScheduler, CooperationMode, CooperationTask
from .routing.intelligent_router import IntelligentRouter, TaskType, ComplexityLevel
from .services.application_service import ApplicationServiceManager

# Main classes for easy import
__all__ = [
    # Core config
    "SystemConfig",
    "ModelConfig", 
    "OpenAIAPIConfig",
    
    # Engine
    "OpenAIEngine",
    "InferenceRequest",
    "InferenceResponse",
    
    # Managers
    "ModelResourceManager",
    "ModelStatus",
    "ModelMetrics",
    
    # Schedulers
    "CooperationScheduler",
    "CooperationMode",
    "CooperationTask",
    
    # Routing
    "IntelligentRouter",
    "TaskType",
    "ComplexityLevel",
    
    # Services
    "ApplicationServiceManager",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# Package-level constants
DEFAULT_CONFIG_FILE = "llm_cooperation_config.json"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_API_TIMEOUT = 60.0

# Initialize logging for the package
import logging

def setup_logging(level: str = DEFAULT_LOG_LEVEL, 
                 format_string: str = None) -> logging.Logger:
    """Setup logging for the LLM Cooperation package"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string
    )
    
    logger = logging.getLogger("llm_cooperation")
    return logger

# Default logger
logger = setup_logging()

def get_version() -> str:
    """Get package version"""
    return __version__

def get_info() -> dict:
    """Get package information"""
    return {
        "name": "llm-cooperation",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__
    }