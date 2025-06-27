"""
Environment configuration loader
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

def load_environment_config(env_file: str = None) -> Dict[str, Any]:
    """Load configuration from environment variables"""
    
    # Try to load from different possible locations
    env_files = [
        env_file,
        "API_Key_DeepSeek.env",
        ".env",
        "llm_cooperation.env"
    ]
    
    for env_path in env_files:
        if env_path and Path(env_path).exists():
            load_dotenv(env_path)
            break
    
    # Load configuration from environment
    config = {
        "BASE_URL": os.getenv("BASE_URL", "https://api2.aigcbest.top/v1"),
        "API_KEY": os.getenv("API_KEY", ""),
        "MODEL": os.getenv("MODEL", "qwen3_8b"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "ENABLE_METRICS": os.getenv("ENABLE_METRICS", "true").lower() == "true",
        "METRICS_PORT": int(os.getenv("METRICS_PORT", "9090")),
        "SERVER_HOST": os.getenv("SERVER_HOST", "0.0.0.0"),
        "SERVER_PORT": int(os.getenv("SERVER_PORT", "8080")),
        "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
        "REQUEST_TIMEOUT": float(os.getenv("REQUEST_TIMEOUT", "60.0")),
    }
    
    return config

def save_environment_config(config: Dict[str, Any], env_file: str = "llm_cooperation.env"):
    """Save configuration to environment file"""
    with open(env_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

def validate_environment_config(config: Dict[str, Any]) -> bool:
    """Validate environment configuration"""
    required_keys = ["BASE_URL", "API_KEY"]
    
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required environment variable: {key}")
    
    return True