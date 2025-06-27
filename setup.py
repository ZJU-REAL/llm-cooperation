"""
Setup script for LLM Cooperation System
"""
from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "llm_cooperation" / "__init__.py"
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read long description from README
def get_long_description():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LLM Cooperation System for intelligent multi-model routing and coordination"

# Read requirements
def get_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="llm-cooperation",
    version=get_version(),
    author="LLM Cooperation Team", 
    author_email="team@llmcooperation.com",
    description="Intelligent multi-model LLM cooperation system with OpenAI API support",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/llm-cooperation/llm-cooperation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0"
        ],
        "server": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "prometheus-client>=0.19.0"
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
            "structlog>=23.2.0"
        ],
        "all": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0", 
            "prometheus-client>=0.19.0",
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
            "structlog>=23.2.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "llm-cooperation=llm_cooperation.cli:main",
            "llm-config=llm_cooperation.tools.config_cli:main",
            "llm-server=llm_cooperation.server.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_cooperation": [
            "config/*.json",
            "templates/*.yaml",
            "static/*"
        ],
    },
    zip_safe=False,
    keywords=[
        "llm", "cooperation", "ai", "openai", "api", "routing", 
        "multi-model", "inference", "nlp", "chatbot", "assistant"
    ],
    project_urls={
        "Bug Reports": "https://github.com/llm-cooperation/llm-cooperation/issues",
        "Source": "https://github.com/llm-cooperation/llm-cooperation",
        "Documentation": "https://llm-cooperation.readthedocs.io/",
    },
)