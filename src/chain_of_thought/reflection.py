"""
Reflection and Self-Correction Implementation
Author: Balaji Koneti

Implements reflective reasoning with self-evaluation, error detection,
and iterative refinement for improved reasoning quality.
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


class ReflectionType(str, Enum):
    """Types of reflection"""
    SELF_EVALUATION = "self_evaluation"
    ERROR_DETECTION = "error_detection"
    GAP_ANALYSIS = "gap_analysis"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    IMPROVEMENT_SUGGESTION = "improvement_suggestion"


class ReflectionStep(BaseModel):
    """Individual reflection step"""
    step_number: int = Field(..., description="Step number in reflection process")
    reflection_type: ReflectionType = Field(..., description="Type of reflection")
    original_reasoning: str = Field(..., description="Original reasoning being reflected upon")
    reflection_content: str = Field(..., description="Reflection analysis")
    confidence_before: float = Field(..., description="Confidence before reflection")
    confidence_after: float = Field(..., description="Confidence after reflection")
    issues_identified: List[str] = Field(default_factory=list, description="Issues identified")
    improvements_suggested: List[str] = Field(default_factory=list, description="Improvements suggested")
    provider_used: ProviderType = Field(..., description="Provider used for reflection")
    model_used: str = Field(..., description="Model used for reflection")
    response_time: float = Field(..., description="Response time for reflection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Reflection metadata")


class RefinementResult(BaseModel):
    """Result of reasoning refinement"""
    original_reasoning: str = Field(..., description="Original reasoning")
    refined_reasoning: str = Field(..., description="Refined reasoning")
    improvements_made: List[str] = Field(..., description="List of improvements made")
    confidence_improvement: float = Field(..., description="Confidence improvement")
    refinement_iterations: int = Field(..., description="Number of refinement iterations")
    final_confidence: float = Field(..., description="Final confidence score")


class ReflectionConfig(BaseModel):
    """Configuration for reflection-based reasoning"""
    max_reflection_steps: int = Field(default=3, description="Maximum reflection steps")
    confidence_threshold: float = Field(default=0.8, description="Target confidence threshold")
    improvement_threshold: float = Field(default=0.1, description="Minimum improvement threshold")
    enable_error_detection: bool = Field(default=True, description="Enable error detection")
    enable_gap_analysis: bool = Field(default=True, description="Enable gap analysis")
    enable_confidence_assessment: bool = Field(default=True, description="Enable confidence assessment")
    max_refinement_iterations: int = Field(default=2, description="Maximum refinement iterations")
    timeout_per_reflection: float = Field(default=30.0, description="Timeout per reflection step")


class ReflectionResult(BaseModel):
    """Result of reflection-based reasoning"""
    original_reasoning: str = Field(..., description="Original reasoning")
    reflection_steps: List[ReflectionStep] = Field(..., description="All reflection steps")
    refinement_result: RefinementResult = Field(..., description="Refinement result")
    final_answer: str = Field(..., description="Final refined answer")
    total_time: float = Field(..., description="Total reflection time")
    confidence_improvement: float = Field(..., description="Overall confidence improvement")
    provider_used: ProviderType = Field(..., description="Provider used")
    model_used: str = Field(..., description="Model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Reflection:
    """
    Reflection and self-correction implementation
    
    Features:
    - Self-evaluation of reasoning quality
    - Error detection and correction
    - Gap analysis and improvement suggestions
    - Confidence assessment and calibration
    - Iterative refinement process
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize reflection system
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.reasoning_history: List[ReflectionResult] = []
    
    async def reflect_and_refine(self, request: CompletionRequest, 
                               config: ReflectionConfig) -> ReflectionResult:
        """
        Perform reflection and refinement on reasoning
        
        Args:
            request: Completion request
            config: Reflection configuration
            
        Returns:
            Reflection and refinement result
        """
        start_time = time.time()
        
        try:
            logger.info("Starting reflection and refinement", 
                       prompt_length=len(request.prompt),
                       max_steps=config.max_reflection_steps)
            
            # Generate initial reasoning
            initial_reasoning = await self._generate_initial_reasoning(request)
            initial_confidence = 0.7  # Default initial confidence
            
            # Perform reflection steps
            reflection_steps = []
            current_reasoning = initial_reasoning
            current_confidence = initial_confidence
            
            for step_num in range(1, config.max_reflection_steps + 1):
                # Determine reflection type
                reflection_type = self._determine_reflection_type(step_num, config)
                
                # Perform reflection
                reflection_step = await self._perform_reflection(
                    current_reasoning, reflection_type, request, config, step_num
                )
                
                if not reflection_step:
                    logger.warning("Reflection step failed", step_number=step_num)
                    break
                
                reflection_steps.append(reflection_step)
                
                # Update confidence
                current_confidence = reflection_step.confidence_after
                
                # Check if we've reached target confidence
                if current_confidence >= config.confidence_threshold:
                    logger.info("Target confidence reached", 
                              step=step_num, 
                              confidence=current_confidence)
                    break
                
                # Check if improvement is significant
                confidence_improvement = current_confidence - reflection_step.confidence_before
                if confidence_improvement < config.improvement_threshold:
                    logger.info("Minimal improvement, stopping reflection",
                              improvement=confidence_improvement)
                    break
            
            # Perform refinement based on reflections
            refinement_result = await self._refine_reasoning(
                current_reasoning, reflection_steps, request, config
            )
            
            # Generate final answer
            final_answer = await self._generate_final_answer(
                refinement_result.refined_reasoning, request
            )
            
            # Calculate overall improvement
            confidence_improvement = refinement_result.final_confidence - initial_confidence
            
            # Create result
            result = ReflectionResult(
                original_reasoning=initial_reasoning,
                reflection_steps=reflection_steps,
                refinement_result=refinement_result,
                final_answer=final_answer,
                total_time=time.time() - start_time,
                confidence_improvement=confidence_improvement,
                provider_used=reflection_steps[0].provider_used if reflection_steps else ProviderType.OPENAI,
                model_used=request.model,
                metadata={
                    "reflection_steps_completed": len(reflection_steps),
                    "target_confidence_achieved": refinement_result.final_confidence >= config.confidence_threshold,
                    "significant_improvement": confidence_improvement >= config.improvement_threshold
                }
            )
            
            # Store in history
            self.reasoning_history.append(result)
            
            logger.info("Reflection and refinement completed",
                       steps_completed=len(reflection_steps),
                       confidence_improvement=confidence_improvement,
                       total_time=result.total_time)
            
            return result
            
        except Exception as e:
            logger.error("Reflection and refinement failed", error=str(e))
            raise
    
    async def _generate_initial_reasoning(self, request: CompletionRequest) -> str:
        """
        Generate initial reasoning for reflection
        
        Args:
            request: Completion request
            
        Returns:
            Initial reasoning text
        """
        initial_prompt = f"""
        Problem: {request.prompt}
        
        Provide your initial reasoning and analysis for this problem.
        Think step by step and explain your reasoning process.
        Be thorough but don't worry about perfection - this is just the starting point.
        """
        
        initial_request = CompletionRequest(
            prompt=initial_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
            extended_thinking=request.extended_thinking,
            thinking_mode=request.thinking_mode
        )
        
        # Select provider
        provider_type = await self._select_provider(request)
        
        try:
            response = await self.llm_manager.generate_completion(
                initial_request, preferred_provider=provider_type
            )
            
            return response.reasoning_trace or response.content
            
        except Exception as e:
            logger.error("Initial reasoning generation failed", error=str(e))
            return "Initial reasoning generation failed"
    
    def _determine_reflection_type(self, step_num: int, config: ReflectionConfig) -> ReflectionType:
        """
        Determine the type of reflection for this step
        
        Args:
            step_num: Current step number
            config: Reflection configuration
            
        Returns:
            Reflection type for this step
        """
        # Cycle through different reflection types
        reflection_types = []
        
        if config.enable_error_detection:
            reflection_types.append(ReflectionType.ERROR_DETECTION)
        
        if config.enable_gap_analysis:
            reflection_types.append(ReflectionType.GAP_ANALYSIS)
        
        if config.enable_confidence_assessment:
            reflection_types.append(ReflectionType.CONFIDENCE_ASSESSMENT)
        
        reflection_types.extend([
            ReflectionType.SELF_EVALUATION,
            ReflectionType.IMPROVEMENT_SUGGESTION
        ])
        
        # Select type based on step number
        type_index = (step_num - 1) % len(reflection_types)
        return reflection_types[type_index]
    
    async def _perform_reflection(self, reasoning: str, reflection_type: ReflectionType,
                                original_request: CompletionRequest, config: ReflectionConfig,
                                step_num: int) -> Optional[ReflectionStep]:
        """
        Perform a single reflection step
        
        Args:
            reasoning: Reasoning to reflect upon
            reflection_type: Type of reflection to perform
            original_request: Original completion request
            config: Reflection configuration
            step_num: Step number
            
        Returns:
            Reflection step result or None if failed
        """
        try:
            # Create reflection prompt
            reflection_prompt = self._create_reflection_prompt(
                reasoning, reflection_type, original_request, step_num
            )
            
            # Create completion request
            reflection_request = CompletionRequest(
                prompt=reflection_prompt,
                model=original_request.model,
                temperature=0.3,  # Lower temperature for reflection
                max_tokens=original_request.max_tokens // 2,
                reasoning_effort=original_request.reasoning_effort,
                extended_thinking=original_request.extended_thinking,
                thinking_mode=original_request.thinking_mode
            )
            
            # Select provider
            provider_type = await self._select_provider(original_request)
            
            # Get reflection with timeout
            start_time = time.time()
            response = await asyncio.wait_for(
                self.llm_manager.generate_completion(reflection_request, preferred_provider=provider_type),
                timeout=config.timeout_per_reflection
            )
            response_time = time.time() - start_time
            
            # Parse reflection content
            reflection_content = response.reasoning_trace or response.content
            
            # Extract structured information from reflection
            issues, improvements, confidence = self._parse_reflection_content(
                reflection_content, reflection_type
            )
            
            # Create reflection step
            reflection_step = ReflectionStep(
                step_number=step_num,
                reflection_type=reflection_type,
                original_reasoning=reasoning,
                reflection_content=reflection_content,
                confidence_before=0.7,  # Will be updated
                confidence_after=confidence,
                issues_identified=issues,
                improvements_suggested=improvements,
                provider_used=provider_type,
                model_used=original_request.model,
                response_time=response_time,
                metadata={
                    "reflection_type": reflection_type.value,
                    "tokens_used": response.usage.get("total_tokens", 0)
                }
            )
            
            return reflection_step
            
        except asyncio.TimeoutError:
            logger.warning("Reflection timeout", step_number=step_num)
            return None
        except Exception as e:
            logger.error("Reflection failed", step_number=step_num, error=str(e))
            return None
    
    def _create_reflection_prompt(self, reasoning: str, reflection_type: ReflectionType,
                                original_request: CompletionRequest, step_num: int) -> str:
        """
        Create a prompt for specific reflection type
        
        Args:
            reasoning: Reasoning to reflect upon
            reflection_type: Type of reflection
            original_request: Original completion request
            step_num: Step number
            
        Returns:
            Reflection prompt
        """
        base_prompt = f"""
        Original question: {original_request.prompt}
        
        Current reasoning to reflect upon:
        {reasoning}
        
        This is reflection step {step_num}. Please perform a {reflection_type.value} analysis.
        
        """
        
        if reflection_type == ReflectionType.ERROR_DETECTION:
            base_prompt += """
            Focus on error detection:
            1. Identify any logical errors or inconsistencies
            2. Look for factual inaccuracies or unsupported claims
            3. Check for reasoning gaps or missing steps
            4. Identify any contradictions or conflicts
            5. Rate your confidence in the reasoning (0-1)
            
            Provide specific issues found and rate confidence.
            """
        
        elif reflection_type == ReflectionType.GAP_ANALYSIS:
            base_prompt += """
            Focus on gap analysis:
            1. Identify what information is missing
            2. Look for unexplored aspects of the problem
            3. Identify assumptions that need validation
            4. Find areas where reasoning could be deeper
            5. Rate your confidence in the reasoning (0-1)
            
            Provide specific gaps identified and rate confidence.
            """
        
        elif reflection_type == ReflectionType.CONFIDENCE_ASSESSMENT:
            base_prompt += """
            Focus on confidence assessment:
            1. Evaluate how certain you are about each part of the reasoning
            2. Identify which parts are most/least confident
            3. Consider what additional evidence would increase confidence
            4. Assess the overall reliability of the reasoning
            5. Provide an overall confidence score (0-1)
            
            Provide detailed confidence assessment and overall score.
            """
        
        elif reflection_type == ReflectionType.SELF_EVALUATION:
            base_prompt += """
            Focus on self-evaluation:
            1. Critically evaluate the quality of your reasoning
            2. Assess the clarity and organization of your thoughts
            3. Evaluate the completeness of your analysis
            4. Consider alternative approaches or perspectives
            5. Rate your confidence in the reasoning (0-1)
            
            Provide honest self-evaluation and rate confidence.
            """
        
        elif reflection_type == ReflectionType.IMPROVEMENT_SUGGESTION:
            base_prompt += """
            Focus on improvement suggestions:
            1. Suggest specific ways to improve the reasoning
            2. Identify areas that need more detail or explanation
            3. Suggest alternative approaches or methods
            4. Recommend additional considerations or factors
            5. Rate your confidence in the reasoning (0-1)
            
            Provide specific improvement suggestions and rate confidence.
            """
        
        base_prompt += """
        
        Format your response as:
        Issues: [list of specific issues]
        Improvements: [list of specific improvements]
        Confidence: [0.0-1.0]
        Analysis: [detailed reflection analysis]
        """
        
        return base_prompt
    
    def _parse_reflection_content(self, content: str, reflection_type: ReflectionType) -> Tuple[List[str], List[str], float]:
        """
        Parse structured information from reflection content
        
        Args:
            content: Reflection content
            reflection_type: Type of reflection
            
        Returns:
            Tuple of (issues, improvements, confidence)
        """
        issues = []
        improvements = []
        confidence = 0.7  # Default confidence
        
        try:
            # Try to extract structured information
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('Issues:'):
                    current_section = 'issues'
                    continue
                elif line.startswith('Improvements:'):
                    current_section = 'improvements'
                    continue
                elif line.startswith('Confidence:'):
                    # Extract confidence score
                    try:
                        confidence_str = line.split(':', 1)[1].strip()
                        confidence = float(confidence_str)
                    except (ValueError, IndexError):
                        pass
                    current_section = None
                    continue
                
                # Add content to appropriate section
                if current_section == 'issues' and line:
                    issues.append(line)
                elif current_section == 'improvements' and line:
                    improvements.append(line)
            
            # If no structured format found, try to extract from general text
            if not issues and not improvements:
                # Look for confidence indicators
                if 'confidence' in content.lower():
                    import re
                    confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', content.lower())
                    if confidence_match:
                        try:
                            confidence = float(confidence_match.group(1))
                        except ValueError:
                            pass
                
                # Extract general issues and improvements
                if 'error' in content.lower() or 'problem' in content.lower():
                    issues.append("Potential issues identified in reflection")
                
                if 'improve' in content.lower() or 'better' in content.lower():
                    improvements.append("Improvements suggested in reflection")
        
        except Exception as e:
            logger.warning("Failed to parse reflection content", error=str(e))
        
        return issues, improvements, confidence
    
    async def _refine_reasoning(self, original_reasoning: str, reflection_steps: List[ReflectionStep],
                              original_request: CompletionRequest, config: ReflectionConfig) -> RefinementResult:
        """
        Refine reasoning based on reflection steps
        
        Args:
            original_reasoning: Original reasoning
            reflection_steps: All reflection steps
            original_request: Original completion request
            config: Reflection configuration
            
        Returns:
            Refinement result
        """
        if not reflection_steps:
            return RefinementResult(
                original_reasoning=original_reasoning,
                refined_reasoning=original_reasoning,
                improvements_made=[],
                confidence_improvement=0.0,
                refinement_iterations=0,
                final_confidence=0.7
            )
        
        # Collect all issues and improvements
        all_issues = []
        all_improvements = []
        
        for step in reflection_steps:
            all_issues.extend(step.issues_identified)
            all_improvements.extend(step.improvements_suggested)
        
        # Create refinement prompt
        refinement_prompt = f"""
        Original question: {original_request.prompt}
        
        Original reasoning:
        {original_reasoning}
        
        Issues identified through reflection:
        {chr(10).join(f"- {issue}" for issue in all_issues)}
        
        Improvements suggested:
        {chr(10).join(f"- {improvement}" for improvement in all_improvements)}
        
        Please refine the original reasoning by:
        1. Addressing the identified issues
        2. Incorporating the suggested improvements
        3. Maintaining the core insights while improving quality
        4. Ensuring the reasoning is clear and well-structured
        5. Providing a confidence assessment (0-1)
        
        Provide the refined reasoning and confidence score.
        """
        
        refinement_request = CompletionRequest(
            prompt=refinement_prompt,
            model=original_request.model,
            temperature=0.2,  # Low temperature for refinement
            max_tokens=original_request.max_tokens
        )
        
        try:
            # Use provider from reflection steps
            provider_type = reflection_steps[0].provider_used
            response = await self.llm_manager.generate_completion(
                refinement_request, preferred_provider=provider_type
            )
            
            refined_reasoning = response.reasoning_trace or response.content
            
            # Extract confidence from refined reasoning
            final_confidence = self._extract_confidence_from_text(refined_reasoning)
            
            # Calculate improvement
            original_confidence = reflection_steps[0].confidence_before if reflection_steps else 0.7
            confidence_improvement = final_confidence - original_confidence
            
            return RefinementResult(
                original_reasoning=original_reasoning,
                refined_reasoning=refined_reasoning,
                improvements_made=all_improvements,
                confidence_improvement=confidence_improvement,
                refinement_iterations=1,
                final_confidence=final_confidence
            )
            
        except Exception as e:
            logger.error("Reasoning refinement failed", error=str(e))
            return RefinementResult(
                original_reasoning=original_reasoning,
                refined_reasoning=original_reasoning,
                improvements_made=[],
                confidence_improvement=0.0,
                refinement_iterations=0,
                final_confidence=0.7
            )
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """
        Extract confidence score from text
        
        Args:
            text: Text to extract confidence from
            
        Returns:
            Confidence score (0-1)
        """
        try:
            import re
            # Look for confidence patterns
            confidence_patterns = [
                r'confidence[:\s]*([0-9]*\.?[0-9]+)',
                r'confident[:\s]*([0-9]*\.?[0-9]+)',
                r'certainty[:\s]*([0-9]*\.?[0-9]+)'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    confidence = float(match.group(1))
                    # Normalize to 0-1 range if needed
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    return max(0.0, min(1.0, confidence))
        
        except (ValueError, AttributeError):
            pass
        
        return 0.7  # Default confidence
    
    async def _generate_final_answer(self, refined_reasoning: str, 
                                   original_request: CompletionRequest) -> str:
        """
        Generate final answer from refined reasoning
        
        Args:
            refined_reasoning: Refined reasoning
            original_request: Original completion request
            
        Returns:
            Final answer
        """
        final_prompt = f"""
        Original question: {original_request.prompt}
        
        Refined reasoning:
        {refined_reasoning}
        
        Based on this refined reasoning, provide a clear, final answer that:
        1. Directly addresses the original question
        2. Incorporates the key insights from the reasoning
        3. Is well-structured and easy to understand
        4. Reflects the confidence level from the reasoning
        
        Provide your final answer:
        """
        
        final_request = CompletionRequest(
            prompt=final_prompt,
            model=original_request.model,
            temperature=0.3,
            max_tokens=original_request.max_tokens
        )
        
        try:
            provider_type = await self._select_provider(original_request)
            response = await self.llm_manager.generate_completion(
                final_request, preferred_provider=provider_type
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Final answer generation failed", error=str(e))
            return refined_reasoning
    
    async def _select_provider(self, request: CompletionRequest) -> ProviderType:
        """
        Select provider for reflection
        
        Args:
            request: Completion request
            
        Returns:
            Provider type
        """
        # Get available providers
        available_providers = [
            ptype for ptype, pinfo in self.llm_manager.providers.items()
            if pinfo.status.value == "active" and pinfo.provider
        ]
        
        if not available_providers:
            raise RuntimeError("No available providers")
        
        # Prefer reasoning models for reflection
        priority_order = [ProviderType.OPENAI, ProviderType.CLAUDE, ProviderType.GEMINI]
        
        for provider_type in priority_order:
            if provider_type in available_providers:
                return provider_type
        
        return available_providers[0]
    
    def get_reasoning_history(self, limit: int = 10) -> List[ReflectionResult]:
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
        avg_steps = sum(len(result.reflection_steps) for result in self.reasoning_history) / total_sessions
        avg_improvement = sum(result.confidence_improvement for result in self.reasoning_history) / total_sessions
        avg_time = sum(result.total_time for result in self.reasoning_history) / total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "average_reflection_steps": avg_steps,
            "average_confidence_improvement": avg_improvement,
            "average_time": avg_time,
            "target_confidence_achievement_rate": sum(
                1 for result in self.reasoning_history 
                if result.refinement_result.final_confidence >= 0.8
            ) / total_sessions,
            "significant_improvement_rate": sum(
                1 for result in self.reasoning_history 
                if result.confidence_improvement >= 0.1
            ) / total_sessions
        }
