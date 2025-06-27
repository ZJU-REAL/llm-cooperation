#!/usr/bin/env python3
"""
System startup script for LLM Cooperation System
Handles initialization, health checks, and graceful startup
"""
import asyncio
import logging
import sys
import signal
import time
from pathlib import Path
import subprocess
import os
import argparse

from config import SystemConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages system startup and shutdown"""
    
    def __init__(self, development_mode: bool = False):
        self.development_mode = development_mode
        self.processes = []
        self.shutdown_event = asyncio.Event()
        
    async def start_system(self):
        """Start the entire LLM cooperation system"""
        logger.info("Starting LLM Cooperation System...")
        
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Create necessary directories
            self._create_directories()
            
            # Check system requirements
            await self._check_system_requirements()
            
            # Start in development or production mode
            if self.development_mode:
                await self._start_development_mode()
            else:
                await self._start_production_mode()
                
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            await self.shutdown_system()
            sys.exit(1)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "logs",
            "cache/huggingface",
            "cache/transformers",
            "config",
            "monitoring"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    async def _check_system_requirements(self):
        """Check system requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required")
        
        # Check GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            if gpu_available:
                logger.info(f"GPU support available: {gpu_count} GPUs detected")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("No GPU support detected - running in CPU mode")
                
        except ImportError:
            logger.warning("PyTorch not available - GPU check skipped")
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 50:  # Require at least 50GB free space
            raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB available")
        
        logger.info(f"Disk space: {free_gb:.1f}GB available")
    
    async def _start_development_mode(self):
        """Start system in development mode"""
        logger.info("Starting in development mode...")
        
        # Start main server directly
        from main_server import main
        
        # Run server in background task
        server_task = asyncio.create_task(main())
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
    
    async def _start_production_mode(self):
        """Start system in production mode with monitoring"""
        logger.info("Starting in production mode...")
        
        # Start monitoring services
        await self._start_monitoring()
        
        # Start main application
        main_process = await asyncio.create_subprocess_exec(
            sys.executable, "main_server.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.processes.append(main_process)
        
        logger.info(f"Main server started (PID: {main_process.pid})")
        
        # Monitor processes and handle shutdown
        await self._monitor_processes()
    
    async def _start_monitoring(self):
        """Start monitoring services"""
        logger.info("Starting monitoring services...")
        
        # Create Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-cooperation'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 10s
    metrics_path: '/metrics'
"""
        
        os.makedirs("monitoring", exist_ok=True)
        with open("monitoring/prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        logger.info("Monitoring configuration created")
    
    async def _monitor_processes(self):
        """Monitor running processes"""
        while not self.shutdown_event.is_set():
            # Check if processes are still running
            for process in self.processes[:]:  # Copy list to avoid modification during iteration
                if process.poll() is not None:
                    logger.error(f"Process {process.pid} has terminated")
                    self.processes.remove(process)
            
            # If no processes are running, shutdown
            if not self.processes:
                logger.warning("No processes running, shutting down")
                self.shutdown_event.set()
                break
            
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                continue
    
    async def shutdown_system(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down LLM Cooperation System...")
        
        # Terminate all processes
        for process in self.processes:
            try:
                logger.info(f"Terminating process {process.pid}")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing process {process.pid}")
                    process.kill()
                    await process.wait()
                    
            except Exception as e:
                logger.error(f"Error terminating process {process.pid}: {e}")
        
        logger.info("System shutdown complete")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLM Cooperation System")
    parser.add_argument("--dev", action="store_true", 
                       help="Run in development mode")
    parser.add_argument("--config", type=str, 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize system manager
    system_manager = SystemManager(development_mode=args.dev)
    
    try:
        await system_manager.start_system()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await system_manager.shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())