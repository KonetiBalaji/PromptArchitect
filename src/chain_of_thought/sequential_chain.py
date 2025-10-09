"""
Sequential Chain-of-Thought Implementation
Author: Balaji Koneti

Implements sequential reasoning with step-by-step breakdown, intermediate verification,
and reasoning trace visualization for complex problem solving.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from ..llm_manager import LLMManager
from ..llm_providers.base import CompletionRequest, CompletionResponse, ProviderType

logger = structlog.get_logger(__name__)


class StepType(str, Enum):
    """Types of reasoning steps"""
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"


class ReasoningStep(BaseModel):
    """Individual reasoning step in sequential chain"""
    step_number: int = Field(..., description="Step number in the chain")
    step_type: StepType = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Human-readable description")
    reasoning: str = Field(..., description="Detailed reasoning content")
    confidence: float = Field(..., description="Confidence score (0-1)")
    verification_result: Optional[Dict[str, Any]] = Field(default=None, description="Verification results")
    dependencies: List[int] = Field(default_factory=list, description="Dependencies on previous steps")
    timestamp: float = Field(default_factory=time.time, description="Step timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


class SequentialChainConfig(BaseModel):
    """Configuration for sequential chain-of-thought"""
    max_steps: int = Field(default=10, description="Maximum number of reasoning steps")
    verification_enabled: bool = Field(default=True, description="Enable step verification")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    timeout_per_step: float = Field(default=30.0, description="Timeout per step in seconds")
    enable_backtracking: bool = Field(default=True, description="Enable backtracking on low confidence")
    step_types: List[StepType] = Field(
        default_factory=lambda: [StepType.ANALYSIS, StepType.HYPOTHESIS, StepType.VERIFICATION, StepType.SYNTHESIS, StepType.CONCLUSION],
        description="Allowed step types"
    )


class SequentialChainResult(BaseModel):
    """Result of sequential chain-of-thought reasoning"""
    steps: List[ReasoningStep] = Field(..., description="All reasoning steps")
    final_answer: str = Field(..., description="Final synthesized answer")
    overall_confidence: float = Field(..., description="Overall confidence score")
    total_time: float = Field(..., description="Total reasoning time")
    verification_passed: bool = Field(..., description="Whether all verifications passed")
    backtracking_used: bool = Field(default=False, description="Whether backtracking was used")
    provider_used: ProviderType = Field(..., description="Provider used for reasoning")
    model_used: str = Field(..., description="Model used for reasoning")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SequentialChain:
    """
    Sequential chain-of-thought reasoning implementation
    
    Features:
    - Step-by-step reasoning breakdown
    - Intermediate verification
    - Backtracking on low confidence
    - Reasoning trace visualization
    - Dependency tracking between steps
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize sequential chain
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.reasoning_history: List[SequentialChainResult] = []
    
    async def reason(self, request: CompletionRequest, 
                    config: SequentialChainConfig) -> SequentialChainResult:
        """
        Perform sequential chain-of-thought reasoning
        
        Args:
            request: Completion request
            config: Sequential chain configuration
            
        Returns:
            Sequential chain reasoning result
        """
        start_time = time.time()
        
        try:
            logger.info("Starting sequential chain reasoning", 
                       prompt_length=len(request.prompt),
                       max_steps=config.max_steps)
            
            # Initialize reasoning state
            steps = []
            current_context = request.prompt
            backtracking_used = False
            
            # Select best provider for reasoning
            provider_type = await self._select_reasoning_provider(request)
            
            # Main reasoning loop
            for step_num in range(1, config.max_steps + 1):
                # Determine step type based on current progress
                step_type = self._determine_step_type(step_num, steps, config)
                
                # Generate reasoning step
                step = await self._generate_step(
                    current_context, step_num, step_type, request, config, provider_type
                )
                
                if not step:
                    logger.warning("Failed to generate step", step_number=step_num)
                    break
                
                # Verify step if enabled
                if config.verification_enabled:
                    verification_result = await self._verify_step(step, request, config)
                    step.verification_result = verification_result
                    
                    # Check if verification failed
                    if not verification_result.get("passed", True):
                        logger.warning("Step verification failed", 
                                     step_number=step_num,
                                     issues=verification_result.get("issues", []))
                        
                        # Backtrack if enabled
                        if config.enable_backtracking and step_num > 1:
                            logger.info("Backtracking due to verification failure")
                            backtracking_used = True
                            steps = steps[:-1]  # Remove last step
                            step_num -= 2  # Go back two steps (will be incremented)
                            continue
                
                steps.append(step)
                
                # Update context for next step
                current_context = self._update_context(current_context, step, steps)
                
                # Check if we have reached a conclusion
                if step_type == StepType.CONCLUSION or self._is_final_answer(step.reasoning):
                    logger.info("Reached conclusion", step_number=step_num)
                    break
                
                # Check confidence threshold
                if step.confidence < config.confidence_threshold:
                    logger.warning("Low confidence step", 
                                 step_number=step_num,
                                 confidence=step.confidence)
                    
                    if config.enable_backtracking and step_num > 1:
                        logger.info("Backtracking due to low confidence")
                        backtracking_used = True
                        steps = steps[:-1]
                        step_num -= 2
                        continue
            
            # Synthesize final answer
            final_answer = await self._synthesize_final_answer(steps, request, provider_type)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(steps)
            
            # Create result
            result = SequentialChainResult(
                steps=steps,
                final_answer=final_answer,
                overall_confidence=overall_confidence,
                total_time=time.time() - start_time,
                verification_passed=all(
                    step.verification_result.get("passed", True) for step in steps
                ) if config.verification_enabled else True,
                backtracking_used=backtracking_used,
                provider_used=provider_type,
                model_used=request.model,
                metadata={
                    "total_steps": len(steps),
                    "step_types_used": [step.step_type.value for step in steps],
                    "average_step_confidence": overall_confidence
                }
            )
            
            # Store in history
            self.reasoning_history.append(result)
            
            logger.info("Sequential chain reasoning completed",
                       total_steps=len(steps),
                       overall_confidence=overall_confidence,
                       total_time=result.total_time)
            
            return result
            
        except Exception as e:
            logger.error("Sequential chain reasoning failed", error=str(e))
            raise
    
    async def _select_reasoning_provider(self, request: CompletionRequest) -> ProviderType:
        """
        Select the best provider for reasoning
        
        Args:
            request: Completion request
            
        Returns:
            Best provider type for reasoning
        """
        # Get available providers with reasoning capabilities
        available_providers = []
        for provider_type, provider_info in self.llm_manager.providers.items():
            if provider_info.status.value == "active" and provider_info.provider:
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
        
        # Prefer reasoning models
        priority_order = [ProviderType.OPENAI, ProviderType.CLAUDE, ProviderType.GEMINI]
        
        for provider_type in priority_order:
            if provider_type in available_providers:
                return provider_type
        
        return available_providers[0]
    
    def _determine_step_type(self, step_num: int, existing_steps: List[ReasoningStep], 
                           config: SequentialChainConfig) -> StepType:
        """
        Determine the type of reasoning step based on current progress
        
        Args:
            step_num: Current step number
            existing_steps: Previously completed steps
            config: Chain configuration
            
        Returns:
            Step type for current step
        """
        # If this is the first step, start with analysis
        if step_num == 1:
            return StepType.ANALYSIS
        
        # If we have analysis, move to hypothesis
        if any(step.step_type == StepType.ANALYSIS for step in existing_steps):
            if not any(step.step_type == StepType.HYPOTHESIS for step in existing_steps):
                return StepType.HYPOTHESIS
        
        # If we have hypothesis, move to verification
        if any(step.step_type == StepType.HYPOTHESIS for step in existing_steps):
            if not any(step.step_type == StepType.VERIFICATION for step in existing_steps):
                return StepType.VERIFICATION
        
        # If we have verification, move to synthesis
        if any(step.step_type == StepType.VERIFICATION for step in existing_steps):
            if not any(step.step_type == StepType.SYNTHESIS for step in existing_steps):
                return StepType.SYNTHESIS
        
        # If we have synthesis, move to conclusion
        if any(step.step_type == StepType.SYNTHESIS for step in existing_steps):
            return StepType.CONCLUSION
        
        # Default to analysis for additional steps
        return StepType.ANALYSIS
    
    async def _generate_step(self, context: str, step_num: int, step_type: StepType,
                           original_request: CompletionRequest, config: SequentialChainConfig,
                           provider_type: ProviderType) -> Optional[ReasoningStep]:
        """
        Generate a single reasoning step
        
        Args:
            context: Current reasoning context
            step_num: Step number
            step_type: Type of step to generate
            original_request: Original completion request
            config: Chain configuration
            provider_type: Provider to use
            
        Returns:
            Generated reasoning step or None if failed
        """
        try:
            # Create step-specific prompt
            step_prompt = self._create_step_prompt(context, step_num, step_type, original_request)
            
            # Create completion request
            step_request = CompletionRequest(
                prompt=step_prompt,
                model=original_request.model,
                temperature=original_request.temperature,
                max_tokens=original_request.max_tokens // 2,  # Shorter for individual steps
                reasoning_effort=original_request.reasoning_effort,
                extended_thinking=original_request.extended_thinking,
                thinking_mode=original_request.thinking_mode
            )
            
            # Get completion with timeout
            response = await asyncio.wait_for(
                self.llm_manager.generate_completion(step_request, preferred_provider=provider_type),
                timeout=config.timeout_per_step
            )
            
            # Extract reasoning content
            reasoning_content = response.reasoning_trace or response.content
            confidence = response.confidence_score or 0.8
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=step_num,
                step_type=step_type,
                description=f"{step_type.value.title()} step {step_num}",
                reasoning=reasoning_content,
                confidence=confidence,
                dependencies=self._calculate_dependencies(step_num, step_type),
                metadata={
                    "provider": provider_type.value,
                    "model": original_request.model,
                    "response_time": response.response_time
                }
            )
            
            return step
            
        except asyncio.TimeoutError:
            logger.error("Step generation timeout", step_number=step_num)
            return None
        except Exception as e:
            logger.error("Step generation failed", step_number=step_num, error=str(e))
            return None
    
    def _create_step_prompt(self, context: str, step_num: int, step_type: StepType,
                          original_request: CompletionRequest) -> str:
        """
        Create a prompt for a specific reasoning step
        
        Args:
            context: Current reasoning context
            step_num: Step number
            step_type: Type of step
            original_request: Original completion request
            
        Returns:
            Step-specific prompt
        """
        base_prompt = f"""
        Original question: {original_request.prompt}
        
        Current context: {context}
        
        You are performing step {step_num} of a sequential reasoning process.
        Step type: {step_type.value.upper()}
        
        """
        
        if step_type == StepType.ANALYSIS:
            base_prompt += """
            Your task is to analyze the problem:
            1. Break down the problem into its key components
            2. Identify what information is given and what needs to be found
            3. Identify any assumptions or constraints
            4. Determine the approach needed to solve this problem
            
            Provide a clear analysis of the problem.
            """
        
        elif step_type == StepType.HYPOTHESIS:
            base_prompt += """
            Your task is to form hypotheses:
            1. Based on your analysis, what are the possible approaches?
            2. What are the key insights or patterns you notice?
            3. What assumptions can you make?
            4. What is your initial hypothesis for solving this problem?
            
            Provide your reasoning and initial hypothesis.
            """
        
        elif step_type == StepType.VERIFICATION:
            base_prompt += """
            Your task is to verify your hypothesis:
            1. Test your hypothesis against the given information
            2. Check for logical consistency
            3. Identify any potential issues or gaps
            4. Validate your reasoning steps
            
            Provide verification of your hypothesis and reasoning.
            """
        
        elif step_type == StepType.SYNTHESIS:
            base_prompt += """
            Your task is to synthesize your findings:
            1. Combine insights from your analysis and verification
            2. Refine your approach based on what you've learned
            3. Prepare a coherent solution approach
            4. Identify the key points that support your solution
            
            Provide a synthesis of your reasoning and findings.
            """
        
        elif step_type == StepType.CONCLUSION:
            base_prompt += """
            Your task is to provide a conclusion:
            1. State your final answer clearly
            2. Summarize the key reasoning that led to this conclusion
            3. Express your confidence level
            4. Note any limitations or uncertainties
            
            Provide your final conclusion and answer.
            """
        
        return base_prompt
    
    def _calculate_dependencies(self, step_num: int, step_type: StepType) -> List[int]:
        """
        Calculate dependencies for a reasoning step
        
        Args:
            step_num: Current step number
            step_type: Type of step
            
        Returns:
            List of step numbers this step depends on
        """
        dependencies = []
        
        # Each step depends on the previous step
        if step_num > 1:
            dependencies.append(step_num - 1)
        
        # Specific dependencies based on step type
        if step_type == StepType.VERIFICATION:
            # Verification depends on hypothesis
            dependencies.extend([i for i in range(1, step_num) if i != step_num - 1])
        elif step_type == StepType.SYNTHESIS:
            # Synthesis depends on analysis and verification
            dependencies.extend([i for i in range(1, step_num)])
        elif step_type == StepType.CONCLUSION:
            # Conclusion depends on all previous steps
            dependencies.extend(list(range(1, step_num)))
        
        return dependencies
    
    async def _verify_step(self, step: ReasoningStep, original_request: CompletionRequest,
                         config: SequentialChainConfig) -> Dict[str, Any]:
        """
        Verify a reasoning step
        
        Args:
            step: Reasoning step to verify
            original_request: Original completion request
            config: Chain configuration
            
        Returns:
            Verification results
        """
        verification_prompt = f"""
        Original question: {original_request.prompt}
        
        Reasoning step to verify:
        Step {step.step_number} ({step.step_type.value}): {step.reasoning}
        
        Please verify this reasoning step:
        1. Is the logic sound and consistent?
        2. Does it appropriately address the {step.step_type.value} task?
        3. Are there any logical errors or gaps?
        4. How confident are you in this step (0-1)?
        5. What issues, if any, do you see?
        
        Respond with JSON: {{
            "passed": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list of specific issues"],
            "suggestions": ["list of improvement suggestions"]
        }}
        """
        
        verification_request = CompletionRequest(
            prompt=verification_prompt,
            model="gpt-4o-mini",  # Use fast model for verification
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            response = await self.llm_manager.generate_completion(verification_request)
            
            # Parse JSON response
            import json
            verification_result = json.loads(response.content)
            
            return verification_result
            
        except Exception as e:
            logger.warning("Step verification failed", error=str(e))
            return {
                "passed": True,
                "confidence": step.confidence,
                "issues": ["Verification failed"],
                "suggestions": []
            }
    
    def _update_context(self, current_context: str, step: ReasoningStep, 
                       all_steps: List[ReasoningStep]) -> str:
        """
        Update context for next reasoning step
        
        Args:
            current_context: Current reasoning context
            step: Latest completed step
            all_steps: All completed steps
            
        Returns:
            Updated context for next step
        """
        # Add the latest step to context
        updated_context = f"{current_context}\n\nStep {step.step_number} ({step.step_type.value}): {step.reasoning}"
        
        # If we have many steps, summarize earlier ones to keep context manageable
        if len(all_steps) > 5:
            summary = self._summarize_early_steps(all_steps[:-2])  # Keep last 2 steps detailed
            updated_context = f"Previous reasoning summary: {summary}\n\n{updated_context}"
        
        return updated_context
    
    def _summarize_early_steps(self, steps: List[ReasoningStep]) -> str:
        """
        Summarize early reasoning steps to manage context length
        
        Args:
            steps: Steps to summarize
            
        Returns:
            Summary of early steps
        """
        if not steps:
            return ""
        
        summary_parts = []
        for step in steps:
            summary_parts.append(f"Step {step.step_number}: {step.reasoning[:100]}...")
        
        return " | ".join(summary_parts)
    
    async def _synthesize_final_answer(self, steps: List[ReasoningStep], 
                                     original_request: CompletionRequest,
                                     provider_type: ProviderType) -> str:
        """
        Synthesize final answer from all reasoning steps
        
        Args:
            steps: All reasoning steps
            original_request: Original completion request
            provider_type: Provider to use
            
        Returns:
            Synthesized final answer
        """
        if not steps:
            return "No reasoning completed"
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Original question: {original_request.prompt}
        
        Complete reasoning process:
        """
        
        for step in steps:
            synthesis_prompt += f"\nStep {step.step_number} ({step.step_type.value}): {step.reasoning}"
        
        synthesis_prompt += """
        
        Based on this complete reasoning process, provide a clear, concise final answer that:
        1. Directly addresses the original question
        2. Incorporates the key insights from the reasoning
        3. Is well-structured and easy to understand
        4. Expresses appropriate confidence level
        
        Provide your final answer:
        """
        
        synthesis_request = CompletionRequest(
            prompt=synthesis_prompt,
            model=original_request.model,
            temperature=0.3,  # Lower temperature for synthesis
            max_tokens=original_request.max_tokens
        )
        
        try:
            response = await self.llm_manager.generate_completion(
                synthesis_request, 
                preferred_provider=provider_type
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Final answer synthesis failed", error=str(e))
            # Fallback to last step's reasoning
            return steps[-1].reasoning if steps else "Synthesis failed"
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate overall confidence from all steps
        
        Args:
            steps: All reasoning steps
            
        Returns:
            Overall confidence score
        """
        if not steps:
            return 0.0
        
        # Weight later steps more heavily
        weights = [i + 1 for i in range(len(steps))]
        weighted_confidence = sum(step.confidence * weight for step, weight in zip(steps, weights))
        total_weight = sum(weights)
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _is_final_answer(self, reasoning: str) -> bool:
        """
        Check if reasoning contains a final answer
        
        Args:
            reasoning: Reasoning text to check
            
        Returns:
            True if text appears to contain a final answer
        """
        final_indicators = [
            "final answer",
            "conclusion",
            "therefore",
            "in summary",
            "the answer is",
            "my answer is",
            "based on this analysis"
        ]
        
        reasoning_lower = reasoning.lower()
        return any(indicator in reasoning_lower for indicator in final_indicators)
    
    def get_reasoning_history(self, limit: int = 10) -> List[SequentialChainResult]:
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
        
        total_sessions = len(self.reasoning_history)
        avg_steps = sum(len(result.steps) for result in self.reasoning_history) / total_sessions
        avg_confidence = sum(result.overall_confidence for result in self.reasoning_history) / total_sessions
        avg_time = sum(result.total_time for result in self.reasoning_history) / total_sessions
        backtracking_usage = sum(1 for result in self.reasoning_history if result.backtracking_used) / total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "average_steps_per_session": avg_steps,
            "average_confidence": avg_confidence,
            "average_time": avg_time,
            "backtracking_usage_rate": backtracking_usage,
            "verification_pass_rate": sum(
                1 for result in self.reasoning_history if result.verification_passed
            ) / total_sessions
        }
