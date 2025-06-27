"""
Cooperation Scheduling Layer
Manages multi-model coordination, task decomposition, and result integration
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid

from llm_cooperation.config import SystemConfig
from vllm_engine import InferenceRequest, InferenceResponse
from llm_cooperation.managers.model_manager import model_manager

class CooperationMode(Enum):
    SEQUENTIAL = "sequential"  # Models work in sequence
    PARALLEL = "parallel"     # Models work in parallel
    VOTING = "voting"         # Multiple models vote on result
    PIPELINE = "pipeline"     # Output of one feeds into another
    ENSEMBLE = "ensemble"     # Combine multiple model outputs

@dataclass
class TaskStep:
    """Individual step in a cooperation task"""
    step_id: str
    model_name: str
    prompt: str
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[InferenceResponse] = None
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class CooperationTask:
    """Multi-model cooperation task"""
    task_id: str
    mode: CooperationMode
    steps: List[TaskStep]
    integration_strategy: str = "concatenate"
    timeout: float = 120.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    final_result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResultIntegrator(ABC):
    """Abstract base class for result integration strategies"""
    
    @abstractmethod
    async def integrate(self, results: List[InferenceResponse], task: CooperationTask) -> str:
        """Integrate multiple model results into final output"""
        pass

class ConcatenateIntegrator(ResultIntegrator):
    """Simple concatenation of results"""
    
    async def integrate(self, results: List[InferenceResponse], task: CooperationTask) -> str:
        successful_results = [r.text for r in results if r.success and r.text.strip()]
        return "\n\n".join(successful_results)

class VotingIntegrator(ResultIntegrator):
    """Select result by voting/consensus"""
    
    async def integrate(self, results: List[InferenceResponse], task: CooperationTask) -> str:
        if not results:
            return ""
        
        # For now, select the result with highest confidence (lowest latency as proxy)
        best_result = min(results, key=lambda r: r.latency if r.success else float('inf'))
        return best_result.text if best_result.success else ""

class EnsembleIntegrator(ResultIntegrator):
    """Ensemble integration using a meta-model"""
    
    async def integrate(self, results: List[InferenceResponse], task: CooperationTask) -> str:
        if not results:
            return ""
        
        # Create ensemble prompt
        result_texts = [f"Model {i+1} Response: {r.text}" 
                       for i, r in enumerate(results) if r.success]
        
        ensemble_prompt = f"""
Based on the following responses from different models, provide a comprehensive and accurate final answer:

{chr(10).join(result_texts)}

Please integrate the information above and provide the best possible response:
"""
        
        # Use the router model for ensemble integration
        ensemble_request = InferenceRequest(
            prompt=ensemble_prompt,
            model_name="qwen3_32b_router",
            max_tokens=2048
        )
        
        ensemble_response = await model_manager.execute_request(ensemble_request)
        return ensemble_response.text if ensemble_response.success else result_texts[0] if result_texts else ""

class CooperationScheduler:
    """Manages multi-model cooperation and task orchestration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_tasks: Dict[str, CooperationTask] = {}
        self.task_history: List[CooperationTask] = []
        self.integrators = {
            "concatenate": ConcatenateIntegrator(),
            "voting": VotingIntegrator(),
            "ensemble": EnsembleIntegrator()
        }
        
    async def create_sequential_task(self, 
                                   user_query: str, 
                                   model_sequence: List[str],
                                   integration_strategy: str = "concatenate") -> str:
        """Create a sequential cooperation task"""
        task_id = str(uuid.uuid4())
        
        steps = []
        for i, model_name in enumerate(model_sequence):
            step_id = f"{task_id}_step_{i}"
            
            # Create context-aware prompt for each step
            if i == 0:
                prompt = user_query
            else:
                prompt = f"""
Previous analysis: {{previous_result}}

Original query: {user_query}

Please provide additional analysis or refinement based on the previous work:
"""
            
            step = TaskStep(
                step_id=step_id,
                model_name=model_name,
                prompt=prompt,
                dependencies=[f"{task_id}_step_{i-1}"] if i > 0 else []
            )
            steps.append(step)
        
        task = CooperationTask(
            task_id=task_id,
            mode=CooperationMode.SEQUENTIAL,
            steps=steps,
            integration_strategy=integration_strategy
        )
        
        return await self._execute_task(task)
    
    async def create_parallel_task(self,
                                 user_query: str,
                                 model_list: List[str],
                                 integration_strategy: str = "ensemble") -> str:
        """Create a parallel cooperation task"""
        task_id = str(uuid.uuid4())
        
        steps = []
        for i, model_name in enumerate(model_list):
            step_id = f"{task_id}_parallel_{i}"
            step = TaskStep(
                step_id=step_id,
                model_name=model_name,
                prompt=user_query
            )
            steps.append(step)
        
        task = CooperationTask(
            task_id=task_id,
            mode=CooperationMode.PARALLEL,
            steps=steps,
            integration_strategy=integration_strategy
        )
        
        return await self._execute_task(task)
    
    async def create_voting_task(self,
                               user_query: str,
                               model_list: List[str],
                               voting_rounds: int = 1) -> str:
        """Create a voting-based cooperation task"""
        task_id = str(uuid.uuid4())
        
        steps = []
        # First round: all models provide initial responses
        for i, model_name in enumerate(model_list):
            step_id = f"{task_id}_vote_{i}"
            step = TaskStep(
                step_id=step_id,
                model_name=model_name,
                prompt=user_query
            )
            steps.append(step)
        
        task = CooperationTask(
            task_id=task_id,
            mode=CooperationMode.VOTING,
            steps=steps,
            integration_strategy="voting",
            metadata={"voting_rounds": voting_rounds}
        )
        
        return await self._execute_task(task)
    
    async def create_pipeline_task(self,
                                 user_query: str,
                                 pipeline_config: List[Dict[str, Any]]) -> str:
        """Create a pipeline cooperation task"""
        task_id = str(uuid.uuid4())
        
        steps = []
        for i, config in enumerate(pipeline_config):
            step_id = f"{task_id}_pipe_{i}"
            
            prompt = config.get("prompt_template", user_query)
            if i > 0:
                # Add placeholder for previous result
                prompt = prompt.replace("{previous_result}", "{{previous_result}}")
            
            step = TaskStep(
                step_id=step_id,
                model_name=config["model"],
                prompt=prompt,
                dependencies=[f"{task_id}_pipe_{i-1}"] if i > 0 else [],
                params=config.get("params", {})
            )
            steps.append(step)
        
        task = CooperationTask(
            task_id=task_id,
            mode=CooperationMode.PIPELINE,
            steps=steps,
            integration_strategy="concatenate"
        )
        
        return await self._execute_task(task)
    
    async def _execute_task(self, task: CooperationTask) -> str:
        """Execute a cooperation task"""
        self.logger.info(f"Starting cooperation task {task.task_id} in {task.mode.value} mode")
        
        self.active_tasks[task.task_id] = task
        task.started_at = time.time()
        
        try:
            if task.mode == CooperationMode.SEQUENTIAL:
                result = await self._execute_sequential(task)
            elif task.mode == CooperationMode.PARALLEL:
                result = await self._execute_parallel(task)
            elif task.mode == CooperationMode.VOTING:
                result = await self._execute_voting(task)
            elif task.mode == CooperationMode.PIPELINE:
                result = await self._execute_pipeline(task)
            else:
                raise ValueError(f"Unsupported cooperation mode: {task.mode}")
            
            task.final_result = result
            task.completed_at = time.time()
            
            self.logger.info(f"Cooperation task {task.task_id} completed in "
                           f"{task.completed_at - task.started_at:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cooperation task {task.task_id} failed: {e}")
            task.completed_at = time.time()
            return f"Task execution failed: {str(e)}"
        
        finally:
            # Move to history
            self.task_history.append(task)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _execute_sequential(self, task: CooperationTask) -> str:
        """Execute sequential cooperation"""
        results = []
        previous_result = ""
        
        for step in task.steps:
            # Substitute previous result if needed
            if "{previous_result}" in step.prompt:
                step.prompt = step.prompt.format(previous_result=previous_result)
            
            step.status = "running"
            
            request = InferenceRequest(
                prompt=step.prompt,
                model_name=step.model_name,
                **step.params
            )
            
            response = await model_manager.execute_request(request)
            step.result = response
            
            if response.success:
                step.status = "completed"
                results.append(response)
                previous_result = response.text
            else:
                step.status = "failed"
                self.logger.warning(f"Step {step.step_id} failed: {response.error}")
        
        return await self._integrate_results(results, task)
    
    async def _execute_parallel(self, task: CooperationTask) -> str:
        """Execute parallel cooperation"""
        # Create all requests
        requests = []
        for step in task.steps:
            step.status = "running"
            request = InferenceRequest(
                prompt=step.prompt,
                model_name=step.model_name,
                **step.params
            )
            requests.append((step, request))
        
        # Execute all requests in parallel
        tasks_list = [model_manager.execute_request(req) for _, req in requests]
        responses = await asyncio.gather(*tasks_list, return_exceptions=True)
        
        # Process results
        results = []
        for (step, _), response in zip(requests, responses):
            if isinstance(response, Exception):
                step.status = "failed"
                step.result = InferenceResponse(
                    text="", model_name=step.model_name, usage={}, latency=0,
                    success=False, error=str(response)
                )
            else:
                step.result = response
                if response.success:
                    step.status = "completed"
                    results.append(response)
                else:
                    step.status = "failed"
        
        return await self._integrate_results(results, task)
    
    async def _execute_voting(self, task: CooperationTask) -> str:
        """Execute voting-based cooperation"""
        # Similar to parallel execution for now
        return await self._execute_parallel(task)
    
    async def _execute_pipeline(self, task: CooperationTask) -> str:
        """Execute pipeline cooperation"""
        results = []
        context = {}
        
        for step in task.steps:
            # Wait for dependencies
            await self._wait_for_dependencies(step, task)
            
            # Build context from previous steps
            if step.dependencies:
                dep_results = [s.result.text for s in task.steps 
                             if s.step_id in step.dependencies and s.result and s.result.success]
                context["previous_result"] = "\n".join(dep_results)
            
            # Format prompt with context
            formatted_prompt = step.prompt.format(**context)
            
            step.status = "running"
            
            request = InferenceRequest(
                prompt=formatted_prompt,
                model_name=step.model_name,
                **step.params
            )
            
            response = await model_manager.execute_request(request)
            step.result = response
            
            if response.success:
                step.status = "completed"
                results.append(response)
                context[f"step_{step.step_id}"] = response.text
            else:
                step.status = "failed"
        
        return await self._integrate_results(results, task)
    
    async def _wait_for_dependencies(self, step: TaskStep, task: CooperationTask):
        """Wait for step dependencies to complete"""
        if not step.dependencies:
            return
        
        max_wait = 60  # Maximum wait time in seconds
        wait_interval = 0.5
        waited = 0
        
        while waited < max_wait:
            all_ready = True
            
            for dep_id in step.dependencies:
                dep_step = next((s for s in task.steps if s.step_id == dep_id), None)
                if not dep_step or dep_step.status not in ["completed", "failed"]:
                    all_ready = False
                    break
            
            if all_ready:
                return
            
            await asyncio.sleep(wait_interval)
            waited += wait_interval
        
        raise TimeoutError(f"Dependencies not ready for step {step.step_id}")
    
    async def _integrate_results(self, results: List[InferenceResponse], task: CooperationTask) -> str:
        """Integrate results using specified strategy"""
        if not results:
            return "No successful results to integrate"
        
        integrator = self.integrators.get(task.integration_strategy)
        if not integrator:
            # Fallback to concatenation
            integrator = self.integrators["concatenate"]
        
        return await integrator.integrate(results, task)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        task = self.active_tasks.get(task_id)
        if not task:
            # Check history
            task = next((t for t in self.task_history if t.task_id == task_id), None)
        
        if not task:
            return None
        
        step_status = []
        for step in task.steps:
            step_info = {
                "step_id": step.step_id,
                "model_name": step.model_name,
                "status": step.status,
                "dependencies": step.dependencies
            }
            if step.result:
                step_info["success"] = step.result.success
                step_info["latency"] = step.result.latency
            step_status.append(step_info)
        
        return {
            "task_id": task.task_id,
            "mode": task.mode.value,
            "status": "completed" if task.completed_at else "running",
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "steps": step_status
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get cooperation system statistics"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_history),
            "cooperation_modes_used": list(set(t.mode.value for t in self.task_history)),
            "average_task_duration": sum(
                (t.completed_at or time.time()) - t.started_at 
                for t in self.task_history if t.started_at
            ) / max(len(self.task_history), 1)
        }

# Global scheduler instance
cooperation_scheduler = CooperationScheduler()