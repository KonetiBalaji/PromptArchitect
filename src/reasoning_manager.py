"""
Reasoning Manager for Unified Reasoning Interface
Author: Balaji Koneti

Manages reasoning capabilities across all LLM providers, implementing
different reasoning strategies and providing a unified interface for
reasoning-based completions.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from .llm_manager import LLMManager
from .llm_providers.base import CompletionRequest, CompletionResponse, ProviderType

logger = structlog.get_logger(__name__)


class ReasoningStrategy(str, Enum):
    """Reasoning strategy types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    TREE_BASED = "tree_based"
    REFLECTIVE = "reflective"
    HYBRID = "hybrid"


class ReasoningConfig(BaseModel):
    """Configuration for reasoning operations"""
    strategy: ReasoningStrategy = Field(default=ReasoningStrategy.SEQUENTIAL, description="Reasoning strategy to use")
    max_steps: int = Field(default=5, description="Maximum reasoning steps")
    verification_enabled: bool = Field(default=True, description="Enable step verification")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")
    timeout: float = Field(default=60.0, description="Timeout for reasoning operations")
    parallel_workers: int = Field(default=3, description="Number of parallel workers for parallel strategy")


class ReasoningStep(BaseModel):
    """Individual reasoning step"""
    step_number: int = Field(..., description="Step number in the reasoning chain")
    description: str = Field(..., description="Description of the step")
    reasoning: str = Field(..., description="Detailed reasoning for this step")
    confidence: float = Field(..., description="Confidence score (0-1)")
    verification_result: Optional[Dict[str, Any]] = Field(default=None, description="Verification results")
    timestamp: float = Field(default_factory=time.time, description="Step timestamp")


class ReasoningResult(BaseModel):
    """Complete reasoning result"""
    strategy: ReasoningStrategy = Field(..., description="Strategy used")
    steps: List[ReasoningStep] = Field(..., description="Reasoning steps")
    final_answer: str = Field(..., description="Final answer")
    confidence: float = Field(..., description="Overall confidence score")
    total_time: float = Field(..., description="Total reasoning time")
    provider_used: ProviderType = Field(..., description="Provider used for reasoning")
    model_used: str = Field(..., description="Model used for reasoning")
    verification_passed: bool = Field(default=True, description="Whether verification passed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReasoningManager:
    """
    Unified reasoning manager for all LLM providers
    
    Features:
    - Multiple reasoning strategies
    - Step-by-step verification
    - Confidence scoring
    - Provider selection for reasoning
    - Result aggregation and analysis
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize reasoning manager
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.reasoning_history: List[ReasoningResult] = []
    
    async def reason(self, request: CompletionRequest, 
                    config: ReasoningConfig) -> ReasoningResult:
        """
        Perform reasoning using the specified strategy
        
        Args:
            request: Completion request
            config: Reasoning configuration
            
        Returns:
            Reasoning result with steps and final answer
        """
        start_time = time.time()
        
        try:
            # Select the best provider for reasoning
            provider_type = await self._select_reasoning_provider(request)
            
            # Execute reasoning based on strategy
            if config.strategy == ReasoningStrategy.SEQUENTIAL:
                result = await self._sequential_reasoning(request, config, provider_type)
            elif config.strategy == ReasoningStrategy.PARALLEL:
                result = await self._parallel_reasoning(request, config, provider_type)
            elif config.strategy == ReasoningStrategy.TREE_BASED:
                result = await self._tree_based_reasoning(request, config, provider_type)
            elif config.strategy == ReasoningStrategy.REFLECTIVE:
                result = await self._reflective_reasoning(request, config, provider_type)
            elif config.strategy == ReasoningStrategy.HYBRID:
                result = await self._hybrid_reasoning(request, config, provider_type)
            else:
                raise ValueError(f"Unsupported reasoning strategy: {config.strategy}")
            
            # Calculate total time
            result.total_time = time.time() - start_time
            
            # Store in history
            self.reasoning_history.append(result)
            
            # Keep only recent history (last 100 results)
            if len(self.reasoning_history) > 100:
                self.reasoning_history = self.reasoning_history[-100:]
            
            logger.info("Reasoning completed", 
                       strategy=config.strategy.value,
                       steps=len(result.steps),
                       confidence=result.confidence,
                       time=result.total_time)
            
            return result
            
        except Exception as e:
            logger.error("Reasoning failed", error=str(e), strategy=config.strategy.value)
            raise
    
    async def _select_reasoning_provider(self, request: CompletionRequest) -> ProviderType:
        """
        Select the best provider for reasoning
        
        Args:
            request: Completion request
            
        Returns:
            Best provider type for reasoning
        """
        # Get available providers
        available_providers = []
        for provider_type, provider_info in self.llm_manager.providers.items():
            if provider_info.status.value == "active" and provider_info.provider:
                # Check if provider supports reasoning
                model_info = await provider_info.provider.get_model_info(request.model)
                if model_info and model_info.supports_reasoning:
                    available_providers.append(provider_type)
        
        if not available_providers:
            # Fallback to any available provider
            available_providers = [
                ptype for ptype, pinfo in self.llm_manager.providers.items()
                if pinfo.status.value == "active" and pinfo.provider
            ]
        
        if not available_providers:
            raise RuntimeError("No available providers for reasoning")
        
        # Select based on priority (OpenAI o1/o3, Claude extended thinking, Gemini reasoning)
        priority_order = [
            ProviderType.OPENAI,  # o1/o3 models
            ProviderType.CLAUDE,  # Extended thinking
            ProviderType.GEMINI   # Reasoning models
        ]
        
        for provider_type in priority_order:
            if provider_type in available_providers:
                return provider_type
        
        # Fallback to first available
        return available_providers[0]
    
    async def _sequential_reasoning(self, request: CompletionRequest, 
                                  config: ReasoningConfig, 
                                  provider_type: ProviderType) -> ReasoningResult:
        """
        Perform sequential reasoning with step-by-step verification
        
        Args:
            request: Completion request
            config: Reasoning configuration
            provider_type: Provider to use
            
        Returns:
            Sequential reasoning result
        """
        steps = []
        current_prompt = request.prompt
        
        for step_num in range(1, config.max_steps + 1):
            # Create step-specific request
            step_request = CompletionRequest(
                prompt=current_prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                reasoning_effort=request.reasoning_effort,
                extended_thinking=request.extended_thinking,
                thinking_mode=request.thinking_mode
            )
            
            # Get completion for this step
            response = await self.llm_manager.generate_completion(
                step_request, 
                preferred_provider=provider_type
            )
            
            # Extract reasoning information
            reasoning_text = response.reasoning_trace or response.content
            confidence = response.confidence_score or 0.8
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=step_num,
                description=f"Step {step_num} reasoning",
                reasoning=reasoning_text,
                confidence=confidence
            )
            
            # Verify step if enabled
            if config.verification_enabled:
                verification_result = await self._verify_step(step, request)
                step.verification_result = verification_result
                
                if not verification_result.get("passed", True):
                    logger.warning("Step verification failed", step=step_num)
            
            steps.append(step)
            
            # Check if we have a final answer
            if self._is_final_answer(reasoning_text):
                break
            
            # Update prompt for next step
            current_prompt = f"{current_prompt}\n\nStep {step_num} result: {reasoning_text}\n\nContinue reasoning:"
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
        
        # Extract final answer
        final_answer = steps[-1].reasoning if steps else "No reasoning completed"
        
        return ReasoningResult(
            strategy=ReasoningStrategy.SEQUENTIAL,
            steps=steps,
            final_answer=final_answer,
            confidence=overall_confidence,
            total_time=0.0,  # Will be set by caller
            provider_used=provider_type,
            model_used=request.model,
            verification_passed=all(
                step.verification_result.get("passed", True) for step in steps
            ) if config.verification_enabled else True
        )
    
    async def _parallel_reasoning(self, request: CompletionRequest, 
                                config: ReasoningConfig, 
                                provider_type: ProviderType) -> ReasoningResult:
        """
        Perform parallel reasoning with multiple paths
        
        Args:
            request: Completion request
            config: Reasoning configuration
            provider_type: Provider to use
            
        Returns:
            Parallel reasoning result
        """
        # Create multiple reasoning paths
        path_prompts = [
            f"Approach 1 - Analytical: {request.prompt}",
            f"Approach 2 - Creative: {request.prompt}",
            f"Approach 3 - Systematic: {request.prompt}"
        ]
        
        # Execute parallel reasoning
        tasks = []
        for i, path_prompt in enumerate(path_prompts[:config.parallel_workers]):
            path_request = CompletionRequest(
                prompt=path_prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                reasoning_effort=request.reasoning_effort,
                extended_thinking=request.extended_thinking,
                thinking_mode=request.thinking_mode
            )
            
            task = self.llm_manager.generate_completion(
                path_request, 
                preferred_provider=provider_type
            )
            tasks.append(task)
        
        # Wait for all paths to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses into steps
        steps = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error("Parallel reasoning path failed", path=i, error=str(response))
                continue
            
            reasoning_text = response.reasoning_trace or response.content
            confidence = response.confidence_score or 0.8
            
            step = ReasoningStep(
                step_number=i + 1,
                description=f"Parallel path {i + 1}",
                reasoning=reasoning_text,
                confidence=confidence
            )
            
            steps.append(step)
        
        # Synthesize final answer from all paths
        final_answer = await self._synthesize_parallel_results(steps, request, provider_type)
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
        
        return ReasoningResult(
            strategy=ReasoningStrategy.PARALLEL,
            steps=steps,
            final_answer=final_answer,
            confidence=overall_confidence,
            total_time=0.0,  # Will be set by caller
            provider_used=provider_type,
            model_used=request.model,
            verification_passed=True  # Parallel reasoning doesn't use step verification
        )
    
    async def _tree_based_reasoning(self, request: CompletionRequest, 
                                  config: ReasoningConfig, 
                                  provider_type: ProviderType) -> ReasoningResult:
        """
        Perform tree-based reasoning with branch exploration
        
        Args:
            request: Completion request
            config: Reasoning configuration
            provider_type: Provider to use
            
        Returns:
            Tree-based reasoning result
        """
        # This is a simplified tree-based approach
        # In a full implementation, you would build a proper reasoning tree
        
        steps = []
        current_prompt = request.prompt
        branches_explored = 0
        max_branches = 3
        
        for step_num in range(1, config.max_steps + 1):
            # Explore different branches
            branch_prompts = [
                f"Branch A - Direct approach: {current_prompt}",
                f"Branch B - Alternative approach: {current_prompt}",
                f"Branch C - Creative approach: {current_prompt}"
            ]
            
            # Evaluate each branch
            branch_results = []
            for branch_prompt in branch_prompts[:max_branches]:
                branch_request = CompletionRequest(
                    prompt=branch_prompt,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens // 3,  # Shorter responses for branches
                    reasoning_effort=request.reasoning_effort,
                    extended_thinking=request.extended_thinking,
                    thinking_mode=request.thinking_mode
                )
                
                response = await self.llm_manager.generate_completion(
                    branch_request, 
                    preferred_provider=provider_type
                )
                
                branch_results.append({
                    "content": response.reasoning_trace or response.content,
                    "confidence": response.confidence_score or 0.8
                })
            
            # Select best branch
            best_branch = max(branch_results, key=lambda x: x["confidence"])
            
            step = ReasoningStep(
                step_number=step_num,
                description=f"Tree step {step_num} - Best branch selected",
                reasoning=best_branch["content"],
                confidence=best_branch["confidence"]
            )
            
            steps.append(step)
            branches_explored += len(branch_results)
            
            # Check if we have a final answer
            if self._is_final_answer(best_branch["content"]):
                break
            
            # Update prompt for next level
            current_prompt = f"Based on: {best_branch['content']}\n\nContinue reasoning:"
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
        
        # Extract final answer
        final_answer = steps[-1].reasoning if steps else "No reasoning completed"
        
        return ReasoningResult(
            strategy=ReasoningStrategy.TREE_BASED,
            steps=steps,
            final_answer=final_answer,
            confidence=overall_confidence,
            total_time=0.0,  # Will be set by caller
            provider_used=provider_type,
            model_used=request.model,
            verification_passed=True,
            metadata={"branches_explored": branches_explored}
        )
    
    async def _reflective_reasoning(self, request: CompletionRequest, 
                                  config: ReasoningConfig, 
                                  provider_type: ProviderType) -> ReasoningResult:
        """
        Perform reflective reasoning with self-evaluation
        
        Args:
            request: Completion request
            config: Reasoning configuration
            provider_type: Provider to use
            
        Returns:
            Reflective reasoning result
        """
        steps = []
        current_prompt = request.prompt
        
        for step_num in range(1, config.max_steps + 1):
            # Initial reasoning
            reasoning_request = CompletionRequest(
                prompt=current_prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                reasoning_effort=request.reasoning_effort,
                extended_thinking=request.extended_thinking,
                thinking_mode=request.thinking_mode
            )
            
            response = await self.llm_manager.generate_completion(
                reasoning_request, 
                preferred_provider=provider_type
            )
            
            reasoning_text = response.reasoning_trace or response.content
            confidence = response.confidence_score or 0.8
            
            # Self-reflection
            reflection_prompt = f"""
            Original question: {request.prompt}
            
            My reasoning: {reasoning_text}
            
            Please evaluate my reasoning:
            1. Is my logic sound?
            2. Are there any gaps or errors?
            3. How confident should I be in this answer?
            4. What improvements can I make?
            """
            
            reflection_request = CompletionRequest(
                prompt=reflection_prompt,
                model=request.model,
                temperature=0.3,  # Lower temperature for reflection
                max_tokens=request.max_tokens // 2
            )
            
            reflection_response = await self.llm_manager.generate_completion(
                reflection_request, 
                preferred_provider=provider_type
            )
            
            # Create step with reflection
            step = ReasoningStep(
                step_number=step_num,
                description=f"Reflective step {step_num}",
                reasoning=f"Reasoning: {reasoning_text}\n\nReflection: {reflection_response.content}",
                confidence=confidence
            )
            
            steps.append(step)
            
            # Check if reflection suggests we're done
            if "confident" in reflection_response.content.lower() and confidence > config.confidence_threshold:
                break
            
            # Update prompt based on reflection
            current_prompt = f"Based on my reflection: {reflection_response.content}\n\nLet me reconsider: {request.prompt}"
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
        
        # Extract final answer
        final_answer = steps[-1].reasoning if steps else "No reasoning completed"
        
        return ReasoningResult(
            strategy=ReasoningStrategy.REFLECTIVE,
            steps=steps,
            final_answer=final_answer,
            confidence=overall_confidence,
            total_time=0.0,  # Will be set by caller
            provider_used=provider_type,
            model_used=request.model,
            verification_passed=True
        )
    
    async def _hybrid_reasoning(self, request: CompletionRequest, 
                              config: ReasoningConfig, 
                              provider_type: ProviderType) -> ReasoningResult:
        """
        Perform hybrid reasoning combining multiple strategies
        
        Args:
            request: Completion request
            config: Reasoning configuration
            provider_type: Provider to use
            
        Returns:
            Hybrid reasoning result
        """
        # Start with sequential reasoning
        sequential_config = ReasoningConfig(
            strategy=ReasoningStrategy.SEQUENTIAL,
            max_steps=2,
            verification_enabled=config.verification_enabled,
            confidence_threshold=config.confidence_threshold
        )
        
        sequential_result = await self._sequential_reasoning(request, sequential_config, provider_type)
        
        # If confidence is low, try parallel reasoning
        if sequential_result.confidence < config.confidence_threshold:
            parallel_config = ReasoningConfig(
                strategy=ReasoningStrategy.PARALLEL,
                max_steps=2,
                verification_enabled=False,
                confidence_threshold=config.confidence_threshold,
                parallel_workers=2
            )
            
            parallel_result = await self._parallel_reasoning(request, parallel_config, provider_type)
            
            # Combine results
            combined_steps = sequential_result.steps + parallel_result.steps
            combined_confidence = (sequential_result.confidence + parallel_result.confidence) / 2
            
            # Synthesize final answer
            synthesis_prompt = f"""
            Sequential reasoning: {sequential_result.final_answer}
            Parallel reasoning: {parallel_result.final_answer}
            
            Please synthesize these into a single, coherent answer.
            """
            
            synthesis_request = CompletionRequest(
                prompt=synthesis_prompt,
                model=request.model,
                temperature=0.5,
                max_tokens=request.max_tokens
            )
            
            synthesis_response = await self.llm_manager.generate_completion(
                synthesis_request, 
                preferred_provider=provider_type
            )
            
            final_answer = synthesis_response.content
        else:
            combined_steps = sequential_result.steps
            combined_confidence = sequential_result.confidence
            final_answer = sequential_result.final_answer
        
        return ReasoningResult(
            strategy=ReasoningStrategy.HYBRID,
            steps=combined_steps,
            final_answer=final_answer,
            confidence=combined_confidence,
            total_time=0.0,  # Will be set by caller
            provider_used=provider_type,
            model_used=request.model,
            verification_passed=True
        )
    
    async def _verify_step(self, step: ReasoningStep, original_request: CompletionRequest) -> Dict[str, Any]:
        """
        Verify a reasoning step
        
        Args:
            step: Reasoning step to verify
            original_request: Original completion request
            
        Returns:
            Verification results
        """
        verification_prompt = f"""
        Original question: {original_request.prompt}
        
        Reasoning step: {step.reasoning}
        
        Please verify this reasoning step:
        1. Is the logic sound?
        2. Does it address the original question?
        3. Are there any obvious errors?
        4. Rate the confidence (0-1).
        
        Respond with JSON: {{"passed": true/false, "confidence": 0.0-1.0, "issues": ["list of issues"]}}
        """
        
        verification_request = CompletionRequest(
            prompt=verification_prompt,
            model="gpt-4o-mini",  # Use a fast model for verification
            temperature=0.1,
            max_tokens=200
        )
        
        try:
            response = await self.llm_manager.generate_completion(verification_request)
            
            # Parse JSON response
            import json
            verification_result = json.loads(response.content)
            
            return verification_result
            
        except Exception as e:
            logger.warning("Step verification failed", error=str(e))
            return {"passed": True, "confidence": step.confidence, "issues": ["Verification failed"]}
    
    async def _synthesize_parallel_results(self, steps: List[ReasoningStep], 
                                         original_request: CompletionRequest,
                                         provider_type: ProviderType) -> str:
        """
        Synthesize results from parallel reasoning paths
        
        Args:
            steps: List of reasoning steps from parallel paths
            original_request: Original completion request
            provider_type: Provider to use for synthesis
            
        Returns:
            Synthesized final answer
        """
        synthesis_prompt = f"""
        Original question: {original_request.prompt}
        
        Multiple reasoning approaches:
        """
        
        for i, step in enumerate(steps):
            synthesis_prompt += f"\nApproach {i+1}: {step.reasoning}"
        
        synthesis_prompt += """
        
        Please synthesize these approaches into a single, coherent answer that:
        1. Takes the best insights from each approach
        2. Resolves any conflicts between approaches
        3. Provides a clear, confident final answer
        """
        
        synthesis_request = CompletionRequest(
            prompt=synthesis_prompt,
            model=original_request.model,
            temperature=0.5,
            max_tokens=original_request.max_tokens
        )
        
        response = await self.llm_manager.generate_completion(
            synthesis_request, 
            preferred_provider=provider_type
        )
        
        return response.content
    
    def _is_final_answer(self, text: str) -> bool:
        """
        Check if the reasoning text contains a final answer
        
        Args:
            text: Reasoning text to check
            
        Returns:
            True if text appears to contain a final answer
        """
        final_indicators = [
            "final answer",
            "conclusion",
            "therefore",
            "in summary",
            "the answer is",
            "my answer is"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in final_indicators)
    
    def get_reasoning_history(self, limit: int = 10) -> List[ReasoningResult]:
        """
        Get recent reasoning history
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent reasoning results
        """
        return self.reasoning_history[-limit:] if self.reasoning_history else []
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """
        Get reasoning statistics
        
        Returns:
            Dictionary with reasoning statistics
        """
        if not self.reasoning_history:
            return {"total_reasoning_sessions": 0}
        
        strategies_used = {}
        avg_confidence = 0.0
        avg_time = 0.0
        total_sessions = len(self.reasoning_history)
        
        for result in self.reasoning_history:
            strategy = result.strategy.value
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
            avg_confidence += result.confidence
            avg_time += result.total_time
        
        avg_confidence /= total_sessions
        avg_time /= total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "strategies_used": strategies_used,
            "average_confidence": avg_confidence,
            "average_time": avg_time,
            "verification_pass_rate": sum(
                1 for r in self.reasoning_history if r.verification_passed
            ) / total_sessions
        }
