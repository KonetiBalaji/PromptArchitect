"""
Parallel Chain-of-Thought Implementation
Author: Balaji Koneti

Implements parallel reasoning with simultaneous exploration of multiple reasoning paths,
consensus building, and conflict resolution for comprehensive problem solving.
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


class ReasoningPath(BaseModel):
    """Individual reasoning path in parallel chain"""
    path_id: int = Field(..., description="Unique path identifier")
    approach: str = Field(..., description="Approach description")
    reasoning: str = Field(..., description="Complete reasoning for this path")
    confidence: float = Field(..., description="Confidence score (0-1)")
    provider_used: ProviderType = Field(..., description="Provider used for this path")
    model_used: str = Field(..., description="Model used for this path")
    response_time: float = Field(..., description="Response time for this path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Path metadata")


class ConsensusResult(BaseModel):
    """Result of consensus building from parallel paths"""
    consensus_answer: str = Field(..., description="Consensus answer")
    agreement_level: float = Field(..., description="Level of agreement (0-1)")
    conflicting_paths: List[int] = Field(default_factory=list, description="Paths with conflicts")
    supporting_paths: List[int] = Field(default_factory=list, description="Paths supporting consensus")
    confidence: float = Field(..., description="Overall confidence in consensus")
    conflicts_resolved: List[Dict[str, Any]] = Field(default_factory=list, description="Resolved conflicts")


class ParallelChainConfig(BaseModel):
    """Configuration for parallel chain-of-thought"""
    num_paths: int = Field(default=3, description="Number of parallel reasoning paths")
    approaches: List[str] = Field(
        default_factory=lambda: ["analytical", "creative", "systematic"],
        description="Reasoning approaches to use"
    )
    consensus_threshold: float = Field(default=0.7, description="Minimum consensus threshold")
    conflict_resolution_enabled: bool = Field(default=True, description="Enable conflict resolution")
    timeout_per_path: float = Field(default=30.0, description="Timeout per path in seconds")
    max_retries: int = Field(default=2, description="Maximum retries for failed paths")
    enable_path_verification: bool = Field(default=True, description="Enable path verification")


class ParallelChainResult(BaseModel):
    """Result of parallel chain-of-thought reasoning"""
    paths: List[ReasoningPath] = Field(..., description="All reasoning paths")
    consensus: ConsensusResult = Field(..., description="Consensus building result")
    final_answer: str = Field(..., description="Final synthesized answer")
    total_time: float = Field(..., description="Total reasoning time")
    parallel_efficiency: float = Field(..., description="Parallel execution efficiency")
    provider_diversity: Dict[str, int] = Field(..., description="Provider usage distribution")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ParallelChain:
    """
    Parallel chain-of-thought reasoning implementation
    
    Features:
    - Simultaneous exploration of multiple reasoning paths
    - Consensus building from multiple approaches
    - Conflict resolution mechanisms
    - Provider diversity for robustness
    - Parallel execution efficiency
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize parallel chain
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.reasoning_history: List[ParallelChainResult] = []
    
    async def reason(self, request: CompletionRequest, 
                    config: ParallelChainConfig) -> ParallelChainResult:
        """
        Perform parallel chain-of-thought reasoning
        
        Args:
            request: Completion request
            config: Parallel chain configuration
            
        Returns:
            Parallel chain reasoning result
        """
        start_time = time.time()
        
        try:
            logger.info("Starting parallel chain reasoning", 
                       prompt_length=len(request.prompt),
                       num_paths=config.num_paths)
            
            # Generate parallel reasoning paths
            paths = await self._generate_parallel_paths(request, config)
            
            if not paths:
                raise RuntimeError("Failed to generate any reasoning paths")
            
            # Build consensus from paths
            consensus = await self._build_consensus(paths, request, config)
            
            # Synthesize final answer
            final_answer = await self._synthesize_final_answer(paths, consensus, request)
            
            # Calculate metrics
            total_time = time.time() - start_time
            parallel_efficiency = self._calculate_parallel_efficiency(paths, total_time)
            provider_diversity = self._calculate_provider_diversity(paths)
            
            # Create result
            result = ParallelChainResult(
                paths=paths,
                consensus=consensus,
                final_answer=final_answer,
                total_time=total_time,
                parallel_efficiency=parallel_efficiency,
                provider_diversity=provider_diversity,
                metadata={
                    "successful_paths": len(paths),
                    "failed_paths": config.num_paths - len(paths),
                    "consensus_achieved": consensus.agreement_level >= config.consensus_threshold
                }
            )
            
            # Store in history
            self.reasoning_history.append(result)
            
            logger.info("Parallel chain reasoning completed",
                       successful_paths=len(paths),
                       consensus_level=consensus.agreement_level,
                       total_time=total_time)
            
            return result
            
        except Exception as e:
            logger.error("Parallel chain reasoning failed", error=str(e))
            raise
    
    async def _generate_parallel_paths(self, request: CompletionRequest, 
                                     config: ParallelChainConfig) -> List[ReasoningPath]:
        """
        Generate parallel reasoning paths
        
        Args:
            request: Completion request
            config: Parallel chain configuration
            
        Returns:
            List of successful reasoning paths
        """
        # Create tasks for parallel execution
        tasks = []
        for i in range(config.num_paths):
            approach = config.approaches[i % len(config.approaches)]
            task = self._generate_single_path(request, i + 1, approach, config)
            tasks.append(task)
        
        # Execute all paths in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_paths = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Path generation failed", path_id=i + 1, error=str(result))
                continue
            
            if result:
                successful_paths.append(result)
        
        return successful_paths
    
    async def _generate_single_path(self, request: CompletionRequest, path_id: int, 
                                  approach: str, config: ParallelChainConfig) -> Optional[ReasoningPath]:
        """
        Generate a single reasoning path
        
        Args:
            request: Completion request
            path_id: Path identifier
            approach: Reasoning approach
            config: Parallel chain configuration
            
        Returns:
            Generated reasoning path or None if failed
        """
        for attempt in range(config.max_retries + 1):
            try:
                # Create approach-specific prompt
                path_prompt = self._create_path_prompt(request.prompt, approach, path_id)
                
                # Select provider for this path
                provider_type = await self._select_path_provider(request, path_id)
                
                # Create completion request
                path_request = CompletionRequest(
                    prompt=path_prompt,
                    model=request.model,
                    temperature=request.temperature + (path_id * 0.1),  # Vary temperature slightly
                    max_tokens=request.max_tokens,
                    reasoning_effort=request.reasoning_effort,
                    extended_thinking=request.extended_thinking,
                    thinking_mode=request.thinking_mode
                )
                
                # Get completion with timeout
                start_time = time.time()
                response = await asyncio.wait_for(
                    self.llm_manager.generate_completion(path_request, preferred_provider=provider_type),
                    timeout=config.timeout_per_path
                )
                response_time = time.time() - start_time
                
                # Extract reasoning content
                reasoning_content = response.reasoning_trace or response.content
                confidence = response.confidence_score or 0.8
                
                # Create reasoning path
                path = ReasoningPath(
                    path_id=path_id,
                    approach=approach,
                    reasoning=reasoning_content,
                    confidence=confidence,
                    provider_used=provider_type,
                    model_used=request.model,
                    response_time=response_time,
                    metadata={
                        "attempt": attempt + 1,
                        "temperature_used": path_request.temperature,
                        "tokens_used": response.usage.get("total_tokens", 0)
                    }
                )
                
                # Verify path if enabled
                if config.enable_path_verification:
                    verification_result = await self._verify_path(path, request)
                    if not verification_result.get("passed", True):
                        logger.warning("Path verification failed", path_id=path_id)
                        if attempt < config.max_retries:
                            continue
                
                return path
                
            except asyncio.TimeoutError:
                logger.warning("Path generation timeout", path_id=path_id, attempt=attempt + 1)
                if attempt < config.max_retries:
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
            except Exception as e:
                logger.warning("Path generation failed", path_id=path_id, attempt=attempt + 1, error=str(e))
                if attempt < config.max_retries:
                    await asyncio.sleep(1)
                    continue
        
        return None
    
    def _create_path_prompt(self, original_prompt: str, approach: str, path_id: int) -> str:
        """
        Create a prompt for a specific reasoning approach
        
        Args:
            original_prompt: Original user prompt
            approach: Reasoning approach
            path_id: Path identifier
            
        Returns:
            Approach-specific prompt
        """
        base_prompt = f"""
        Original question: {original_prompt}
        
        You are exploring this problem using a {approach} approach (Path {path_id}).
        
        """
        
        if approach == "analytical":
            base_prompt += """
            Use an analytical approach:
            1. Break down the problem into logical components
            2. Apply systematic analysis and reasoning
            3. Use data-driven insights and evidence
            4. Focus on accuracy and precision
            5. Provide step-by-step logical reasoning
            
            Think analytically and provide a thorough analysis.
            """
        
        elif approach == "creative":
            base_prompt += """
            Use a creative approach:
            1. Think outside the box and explore novel solutions
            2. Consider unconventional perspectives and analogies
            3. Use imagination and lateral thinking
            4. Look for innovative connections and insights
            5. Challenge assumptions and explore alternatives
            
            Think creatively and provide an innovative perspective.
            """
        
        elif approach == "systematic":
            base_prompt += """
            Use a systematic approach:
            1. Follow a structured methodology
            2. Organize information systematically
            3. Use frameworks and established processes
            4. Ensure comprehensive coverage of all aspects
            5. Provide clear, organized reasoning
            
            Think systematically and provide a well-structured analysis.
            """
        
        elif approach == "intuitive":
            base_prompt += """
            Use an intuitive approach:
            1. Trust your first impressions and gut feelings
            2. Look for patterns and connections
            3. Use experience-based insights
            4. Focus on the essence and core of the problem
            5. Provide insights based on intuition and experience
            
            Think intuitively and provide insightful analysis.
            """
        
        elif approach == "critical":
            base_prompt += """
            Use a critical approach:
            1. Question assumptions and challenge ideas
            2. Look for potential flaws and weaknesses
            3. Evaluate evidence and arguments carefully
            4. Consider alternative explanations
            5. Provide critical analysis and evaluation
            
            Think critically and provide a thorough evaluation.
            """
        
        else:
            base_prompt += f"""
            Use a {approach} approach:
            1. Apply your best reasoning for this type of problem
            2. Consider multiple perspectives
            3. Provide clear and logical reasoning
            4. Support your conclusions with evidence
            5. Be thorough and comprehensive
            
            Think carefully and provide your best analysis.
            """
        
        return base_prompt
    
    async def _select_path_provider(self, request: CompletionRequest, path_id: int) -> ProviderType:
        """
        Select provider for a reasoning path
        
        Args:
            request: Completion request
            path_id: Path identifier
            
        Returns:
            Provider type for this path
        """
        # Get available providers
        available_providers = [
            ptype for ptype, pinfo in self.llm_manager.providers.items()
            if pinfo.status.value == "active" and pinfo.provider
        ]
        
        if not available_providers:
            raise RuntimeError("No available providers")
        
        # Distribute paths across providers for diversity
        provider_index = (path_id - 1) % len(available_providers)
        return available_providers[provider_index]
    
    async def _verify_path(self, path: ReasoningPath, original_request: CompletionRequest) -> Dict[str, Any]:
        """
        Verify a reasoning path
        
        Args:
            path: Reasoning path to verify
            original_request: Original completion request
            
        Returns:
            Verification results
        """
        verification_prompt = f"""
        Original question: {original_request.prompt}
        
        Reasoning path to verify:
        Approach: {path.approach}
        Reasoning: {path.reasoning}
        
        Please verify this reasoning path:
        1. Is the {path.approach} approach appropriate for this problem?
        2. Is the reasoning sound and well-structured?
        3. Does it address the original question effectively?
        4. Are there any logical gaps or errors?
        5. How confident are you in this path (0-1)?
        
        Respond with JSON: {{
            "passed": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list of specific issues"],
            "strengths": ["list of strengths"]
        }}
        """
        
        verification_request = CompletionRequest(
            prompt=verification_prompt,
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            response = await self.llm_manager.generate_completion(verification_request)
            
            import json
            verification_result = json.loads(response.content)
            
            return verification_result
            
        except Exception as e:
            logger.warning("Path verification failed", error=str(e))
            return {
                "passed": True,
                "confidence": path.confidence,
                "issues": ["Verification failed"],
                "strengths": []
            }
    
    async def _build_consensus(self, paths: List[ReasoningPath], 
                             original_request: CompletionRequest,
                             config: ParallelChainConfig) -> ConsensusResult:
        """
        Build consensus from parallel reasoning paths
        
        Args:
            paths: All reasoning paths
            original_request: Original completion request
            config: Parallel chain configuration
            
        Returns:
            Consensus building result
        """
        if len(paths) == 1:
            # Single path - no consensus needed
            return ConsensusResult(
                consensus_answer=paths[0].reasoning,
                agreement_level=1.0,
                supporting_paths=[paths[0].path_id],
                confidence=paths[0].confidence
            )
        
        # Create consensus building prompt
        consensus_prompt = f"""
        Original question: {original_request.prompt}
        
        Multiple reasoning approaches have been used to address this question:
        """
        
        for path in paths:
            consensus_prompt += f"""
        
        Approach {path.path_id} ({path.approach}):
        {path.reasoning}
        """
        
        consensus_prompt += """
        
        Please analyze these different approaches and build a consensus:
        1. Identify areas of agreement between the approaches
        2. Identify any conflicts or contradictions
        3. Synthesize the best insights from each approach
        4. Resolve any conflicts by finding common ground or choosing the best option
        5. Provide a consensus answer that incorporates the strengths of each approach
        6. Rate the level of agreement (0-1) between the approaches
        
        Respond with JSON: {
            "consensus_answer": "your consensus answer",
            "agreement_level": 0.0-1.0,
            "supporting_approaches": [list of approach numbers that support the consensus],
            "conflicting_approaches": [list of approach numbers with conflicts],
            "conflicts_resolved": [list of conflicts that were resolved],
            "confidence": 0.0-1.0
        }
        """
        
        consensus_request = CompletionRequest(
            prompt=consensus_prompt,
            model=original_request.model,
            temperature=0.3,
            max_tokens=original_request.max_tokens
        )
        
        try:
            response = await self.llm_manager.generate_completion(consensus_request)
            
            import json
            consensus_data = json.loads(response.content)
            
            return ConsensusResult(
                consensus_answer=consensus_data["consensus_answer"],
                agreement_level=consensus_data["agreement_level"],
                supporting_paths=consensus_data["supporting_approaches"],
                conflicting_paths=consensus_data["conflicting_approaches"],
                confidence=consensus_data["confidence"],
                conflicts_resolved=consensus_data.get("conflicts_resolved", [])
            )
            
        except Exception as e:
            logger.error("Consensus building failed", error=str(e))
            # Fallback: use highest confidence path
            best_path = max(paths, key=lambda p: p.confidence)
            return ConsensusResult(
                consensus_answer=best_path.reasoning,
                agreement_level=0.5,  # Unknown agreement level
                supporting_paths=[best_path.path_id],
                confidence=best_path.confidence
            )
    
    async def _synthesize_final_answer(self, paths: List[ReasoningPath], 
                                     consensus: ConsensusResult,
                                     original_request: CompletionRequest) -> str:
        """
        Synthesize final answer from paths and consensus
        
        Args:
            paths: All reasoning paths
            consensus: Consensus building result
            original_request: Original completion request
            
        Returns:
            Final synthesized answer
        """
        synthesis_prompt = f"""
        Original question: {original_request.prompt}
        
        Consensus analysis: {consensus.consensus_answer}
        Agreement level: {consensus.agreement_level}
        Supporting approaches: {consensus.supporting_paths}
        
        Based on the consensus analysis, provide a clear, final answer that:
        1. Directly addresses the original question
        2. Incorporates the consensus insights
        3. Is well-structured and easy to understand
        4. Reflects the confidence level from the consensus
        5. Acknowledges any remaining uncertainties
        
        Provide your final answer:
        """
        
        synthesis_request = CompletionRequest(
            prompt=synthesis_prompt,
            model=original_request.model,
            temperature=0.2,
            max_tokens=original_request.max_tokens
        )
        
        try:
            response = await self.llm_manager.generate_completion(synthesis_request)
            return response.content
            
        except Exception as e:
            logger.error("Final answer synthesis failed", error=str(e))
            return consensus.consensus_answer
    
    def _calculate_parallel_efficiency(self, paths: List[ReasoningPath], total_time: float) -> float:
        """
        Calculate parallel execution efficiency
        
        Args:
            paths: All reasoning paths
            total_time: Total execution time
            
        Returns:
            Parallel efficiency (0-1)
        """
        if not paths:
            return 0.0
        
        # Calculate sequential time (sum of all path times)
        sequential_time = sum(path.response_time for path in paths)
        
        # Efficiency = sequential_time / (parallel_time * num_paths)
        if total_time > 0:
            efficiency = sequential_time / (total_time * len(paths))
            return min(efficiency, 1.0)  # Cap at 100%
        
        return 0.0
    
    def _calculate_provider_diversity(self, paths: List[ReasoningPath]) -> Dict[str, int]:
        """
        Calculate provider usage distribution
        
        Args:
            paths: All reasoning paths
            
        Returns:
            Dictionary mapping provider types to usage counts
        """
        diversity = {}
        for path in paths:
            provider = path.provider_used.value
            diversity[provider] = diversity.get(provider, 0) + 1
        
        return diversity
    
    def get_reasoning_history(self, limit: int = 10) -> List[ParallelChainResult]:
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
        avg_paths = sum(len(result.paths) for result in self.reasoning_history) / total_sessions
        avg_consensus = sum(result.consensus.agreement_level for result in self.reasoning_history) / total_sessions
        avg_efficiency = sum(result.parallel_efficiency for result in self.reasoning_history) / total_sessions
        avg_time = sum(result.total_time for result in self.reasoning_history) / total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "average_paths_per_session": avg_paths,
            "average_consensus_level": avg_consensus,
            "average_parallel_efficiency": avg_efficiency,
            "average_time": avg_time,
            "consensus_achievement_rate": sum(
                1 for result in self.reasoning_history 
                if result.consensus.agreement_level >= 0.7
            ) / total_sessions
        }
