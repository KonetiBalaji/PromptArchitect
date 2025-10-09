"""
Tree-Based Reasoning Implementation
Author: Balaji Koneti

Implements tree-based reasoning with branch exploration, pruning strategies,
and best-path selection for complex multi-step problem solving.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from ..llm_manager import LLMManager
from ..llm_providers.base import CompletionRequest, CompletionResponse, ProviderType

logger = structlog.get_logger(__name__)


class NodeType(str, Enum):
    """Types of reasoning tree nodes"""
    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"
    PRUNED = "pruned"


class ReasoningNode(BaseModel):
    """Individual node in the reasoning tree"""
    node_id: int = Field(..., description="Unique node identifier")
    parent_id: Optional[int] = Field(default=None, description="Parent node ID")
    node_type: NodeType = Field(..., description="Type of node")
    depth: int = Field(..., description="Depth in the tree")
    reasoning: str = Field(..., description="Reasoning content for this node")
    confidence: float = Field(..., description="Confidence score (0-1)")
    children: List[int] = Field(default_factory=list, description="Child node IDs")
    is_expanded: bool = Field(default=False, description="Whether node has been expanded")
    is_pruned: bool = Field(default=False, description="Whether node has been pruned")
    provider_used: ProviderType = Field(..., description="Provider used for this node")
    model_used: str = Field(..., description="Model used for this node")
    response_time: float = Field(..., description="Response time for this node")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")


class PruningStrategy(str, Enum):
    """Tree pruning strategies"""
    CONFIDENCE_BASED = "confidence_based"
    DEPTH_LIMITED = "depth_limited"
    BEAM_SEARCH = "beam_search"
    ADAPTIVE = "adaptive"


class TreeReasoningConfig(BaseModel):
    """Configuration for tree-based reasoning"""
    max_depth: int = Field(default=4, description="Maximum tree depth")
    max_branches_per_node: int = Field(default=3, description="Maximum branches per node")
    pruning_strategy: PruningStrategy = Field(default=PruningStrategy.ADAPTIVE, description="Pruning strategy")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence threshold")
    beam_width: int = Field(default=2, description="Beam width for beam search")
    timeout_per_node: float = Field(default=20.0, description="Timeout per node in seconds")
    enable_node_verification: bool = Field(default=True, description="Enable node verification")
    max_total_nodes: int = Field(default=50, description="Maximum total nodes in tree")


class TreeReasoningResult(BaseModel):
    """Result of tree-based reasoning"""
    tree: Dict[int, ReasoningNode] = Field(..., description="Complete reasoning tree")
    best_path: List[int] = Field(..., description="Best path through the tree")
    final_answer: str = Field(..., description="Final answer from best path")
    tree_metrics: Dict[str, Any] = Field(..., description="Tree structure metrics")
    total_time: float = Field(..., description="Total reasoning time")
    nodes_explored: int = Field(..., description="Total nodes explored")
    nodes_pruned: int = Field(..., description="Total nodes pruned")
    provider_used: ProviderType = Field(..., description="Primary provider used")
    model_used: str = Field(..., description="Model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TreeReasoning:
    """
    Tree-based reasoning implementation
    
    Features:
    - Hierarchical reasoning tree construction
    - Multiple pruning strategies
    - Best path selection algorithms
    - Adaptive exploration
    - Branch evaluation and scoring
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize tree reasoning
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.reasoning_history: List[TreeReasoningResult] = []
        self._node_counter = 0
    
    async def reason(self, request: CompletionRequest, 
                    config: TreeReasoningConfig) -> TreeReasoningResult:
        """
        Perform tree-based reasoning
        
        Args:
            request: Completion request
            config: Tree reasoning configuration
            
        Returns:
            Tree reasoning result
        """
        start_time = time.time()
        
        try:
            logger.info("Starting tree-based reasoning", 
                       prompt_length=len(request.prompt),
                       max_depth=config.max_depth)
            
            # Initialize tree
            tree = {}
            self._node_counter = 0
            
            # Create root node
            root_node = await self._create_root_node(request, config)
            tree[root_node.node_id] = root_node
            
            # Expand tree iteratively
            nodes_to_expand = [root_node.node_id]
            nodes_explored = 1
            nodes_pruned = 0
            
            while nodes_to_expand and len(tree) < config.max_total_nodes:
                # Select nodes to expand based on pruning strategy
                nodes_to_expand = self._select_nodes_to_expand(
                    tree, nodes_to_expand, config
                )
                
                if not nodes_to_expand:
                    break
                
                # Expand selected nodes
                new_nodes = await self._expand_nodes(
                    tree, nodes_to_expand, request, config
                )
                
                # Add new nodes to tree
                for node in new_nodes:
                    tree[node.node_id] = node
                    nodes_explored += 1
                
                # Update expansion queue
                nodes_to_expand = [
                    node_id for node_id in nodes_to_expand
                    if not tree[node_id].is_expanded and not tree[node_id].is_pruned
                ]
                
                # Apply pruning
                pruned_count = self._apply_pruning(tree, config)
                nodes_pruned += pruned_count
                
                # Update queue to remove pruned nodes
                nodes_to_expand = [
                    node_id for node_id in nodes_to_expand
                    if not tree[node_id].is_pruned
                ]
            
            # Find best path through the tree
            best_path = self._find_best_path(tree, config)
            
            # Generate final answer from best path
            final_answer = await self._generate_final_answer(tree, best_path, request)
            
            # Calculate tree metrics
            tree_metrics = self._calculate_tree_metrics(tree, best_path)
            
            # Create result
            result = TreeReasoningResult(
                tree=tree,
                best_path=best_path,
                final_answer=final_answer,
                tree_metrics=tree_metrics,
                total_time=time.time() - start_time,
                nodes_explored=nodes_explored,
                nodes_pruned=nodes_pruned,
                provider_used=root_node.provider_used,
                model_used=request.model,
                metadata={
                    "max_depth_reached": max(node.depth for node in tree.values()) if tree else 0,
                    "total_branches": sum(len(node.children) for node in tree.values()),
                    "pruning_strategy_used": config.pruning_strategy.value
                }
            )
            
            # Store in history
            self.reasoning_history.append(result)
            
            logger.info("Tree-based reasoning completed",
                       nodes_explored=nodes_explored,
                       nodes_pruned=nodes_pruned,
                       best_path_length=len(best_path),
                       total_time=result.total_time)
            
            return result
            
        except Exception as e:
            logger.error("Tree-based reasoning failed", error=str(e))
            raise
    
    async def _create_root_node(self, request: CompletionRequest, 
                              config: TreeReasoningConfig) -> ReasoningNode:
        """
        Create the root node of the reasoning tree
        
        Args:
            request: Completion request
            config: Tree reasoning configuration
            
        Returns:
            Root reasoning node
        """
        self._node_counter += 1
        
        # Create root prompt
        root_prompt = f"""
        Problem to solve: {request.prompt}
        
        This is the root of a reasoning tree. Your task is to:
        1. Analyze the problem and identify the main question
        2. Break down the problem into key components
        3. Identify the main approaches or strategies to solve this problem
        4. Provide initial reasoning and analysis
        
        Think step by step and provide a comprehensive initial analysis.
        """
        
        # Select provider
        provider_type = await self._select_provider(request)
        
        # Create completion request
        root_request = CompletionRequest(
            prompt=root_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens // 2,
            reasoning_effort=request.reasoning_effort,
            extended_thinking=request.extended_thinking,
            thinking_mode=request.thinking_mode
        )
        
        # Get completion
        start_time = time.time()
        response = await self.llm_manager.generate_completion(
            root_request, preferred_provider=provider_type
        )
        response_time = time.time() - start_time
        
        # Create root node
        root_node = ReasoningNode(
            node_id=self._node_counter,
            parent_id=None,
            node_type=NodeType.ROOT,
            depth=0,
            reasoning=response.reasoning_trace or response.content,
            confidence=response.confidence_score or 0.8,
            provider_used=provider_type,
            model_used=request.model,
            response_time=response_time,
            metadata={
                "is_root": True,
                "tokens_used": response.usage.get("total_tokens", 0)
            }
        )
        
        return root_node
    
    async def _expand_nodes(self, tree: Dict[int, ReasoningNode], 
                          node_ids: List[int], request: CompletionRequest,
                          config: TreeReasoningConfig) -> List[ReasoningNode]:
        """
        Expand multiple nodes in parallel
        
        Args:
            tree: Current reasoning tree
            node_ids: Node IDs to expand
            request: Original completion request
            config: Tree reasoning configuration
            
        Returns:
            List of new nodes created
        """
        # Create expansion tasks
        tasks = []
        for node_id in node_ids:
            if node_id in tree and not tree[node_id].is_expanded:
                task = self._expand_single_node(tree, node_id, request, config)
                tasks.append(task)
        
        # Execute expansions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        new_nodes = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Node expansion failed", error=str(result))
                continue
            
            if result:
                new_nodes.extend(result)
        
        return new_nodes
    
    async def _expand_single_node(self, tree: Dict[int, ReasoningNode], 
                                node_id: int, request: CompletionRequest,
                                config: TreeReasoningConfig) -> List[ReasoningNode]:
        """
        Expand a single node by creating child branches
        
        Args:
            tree: Current reasoning tree
            node_id: Node ID to expand
            request: Original completion request
            config: Tree reasoning configuration
            
        Returns:
            List of new child nodes
        """
        parent_node = tree[node_id]
        if parent_node.is_expanded or parent_node.depth >= config.max_depth:
            return []
        
        # Generate branches for this node
        branches = await self._generate_branches(parent_node, request, config)
        
        # Create child nodes
        child_nodes = []
        for i, branch_reasoning in enumerate(branches[:config.max_branches_per_node]):
            self._node_counter += 1
            
            child_node = ReasoningNode(
                node_id=self._node_counter,
                parent_id=node_id,
                node_type=NodeType.BRANCH if parent_node.depth < config.max_depth - 1 else NodeType.LEAF,
                depth=parent_node.depth + 1,
                reasoning=branch_reasoning,
                confidence=0.8,  # Will be updated by verification
                provider_used=parent_node.provider_used,
                model_used=request.model,
                response_time=0.0,  # Will be updated
                metadata={
                    "branch_index": i,
                    "parent_reasoning": parent_node.reasoning[:100] + "..."
                }
            )
            
            child_nodes.append(child_node)
        
        # Update parent node
        parent_node.children = [node.node_id for node in child_nodes]
        parent_node.is_expanded = True
        
        return child_nodes
    
    async def _generate_branches(self, parent_node: ReasoningNode, 
                               request: CompletionRequest,
                               config: TreeReasoningConfig) -> List[str]:
        """
        Generate reasoning branches for a node
        
        Args:
            parent_node: Parent node to branch from
            request: Original completion request
            config: Tree reasoning configuration
            
        Returns:
            List of branch reasoning texts
        """
        branch_prompt = f"""
        Original problem: {request.prompt}
        
        Current reasoning path (depth {parent_node.depth}):
        {parent_node.reasoning}
        
        Generate {config.max_branches_per_node} different branches to continue this reasoning:
        1. Each branch should explore a different aspect or approach
        2. Each branch should build logically on the current reasoning
        3. Each branch should be a complete thought that could lead to a solution
        4. Make the branches distinct and complementary
        
        Provide each branch as a separate, complete reasoning step.
        Format: Branch 1: [reasoning] | Branch 2: [reasoning] | Branch 3: [reasoning]
        """
        
        branch_request = CompletionRequest(
            prompt=branch_prompt,
            model=request.model,
            temperature=request.temperature + 0.2,  # Higher temperature for diversity
            max_tokens=request.max_tokens // 2,
            reasoning_effort=request.reasoning_effort,
            extended_thinking=request.extended_thinking,
            thinking_mode=request.thinking_mode
        )
        
        try:
            response = await self.llm_manager.generate_completion(
                branch_request, preferred_provider=parent_node.provider_used
            )
            
            # Parse branches from response
            branch_text = response.reasoning_trace or response.content
            branches = self._parse_branches(branch_text, config.max_branches_per_node)
            
            return branches
            
        except Exception as e:
            logger.warning("Branch generation failed", error=str(e))
            return []
    
    def _parse_branches(self, branch_text: str, max_branches: int) -> List[str]:
        """
        Parse branch reasoning from response text
        
        Args:
            branch_text: Response text containing branches
            max_branches: Maximum number of branches to extract
            
        Returns:
            List of parsed branch reasoning texts
        """
        # Try to split by branch markers
        if "|" in branch_text:
            branches = [branch.strip() for branch in branch_text.split("|")]
        elif "Branch" in branch_text:
            # Extract branches by looking for "Branch X:" patterns
            import re
            branch_pattern = r"Branch \d+:\s*(.*?)(?=Branch \d+:|$)"
            matches = re.findall(branch_pattern, branch_text, re.DOTALL)
            branches = [match.strip() for match in matches]
        else:
            # Fallback: split by sentences and group
            sentences = branch_text.split(".")
            branches = []
            current_branch = ""
            
            for sentence in sentences:
                current_branch += sentence.strip() + ". "
                if len(current_branch) > 100:  # Reasonable branch length
                    branches.append(current_branch.strip())
                    current_branch = ""
            
            if current_branch.strip():
                branches.append(current_branch.strip())
        
        # Clean and limit branches
        cleaned_branches = []
        for branch in branches[:max_branches]:
            branch = branch.strip()
            if branch and len(branch) > 20:  # Minimum meaningful length
                cleaned_branches.append(branch)
        
        return cleaned_branches
    
    def _select_nodes_to_expand(self, tree: Dict[int, ReasoningNode], 
                              candidate_nodes: List[int],
                              config: TreeReasoningConfig) -> List[int]:
        """
        Select which nodes to expand based on pruning strategy
        
        Args:
            tree: Current reasoning tree
            candidate_nodes: Candidate nodes for expansion
            config: Tree reasoning configuration
            
        Returns:
            List of node IDs to expand
        """
        if not candidate_nodes:
            return []
        
        # Filter out already expanded or pruned nodes
        expandable_nodes = [
            node_id for node_id in candidate_nodes
            if node_id in tree and not tree[node_id].is_expanded and not tree[node_id].is_pruned
        ]
        
        if config.pruning_strategy == PruningStrategy.BEAM_SEARCH:
            # Select top-k nodes by confidence
            node_confidences = [
                (node_id, tree[node_id].confidence) for node_id in expandable_nodes
            ]
            node_confidences.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in node_confidences[:config.beam_width]]
        
        elif config.pruning_strategy == PruningStrategy.CONFIDENCE_BASED:
            # Select nodes above confidence threshold
            return [
                node_id for node_id in expandable_nodes
                if tree[node_id].confidence >= config.confidence_threshold
            ]
        
        elif config.pruning_strategy == PruningStrategy.DEPTH_LIMITED:
            # Select nodes below max depth
            return [
                node_id for node_id in expandable_nodes
                if tree[node_id].depth < config.max_depth
            ]
        
        else:  # ADAPTIVE
            # Combine multiple criteria
            selected_nodes = []
            for node_id in expandable_nodes:
                node = tree[node_id]
                if (node.confidence >= config.confidence_threshold and 
                    node.depth < config.max_depth):
                    selected_nodes.append(node_id)
            
            # Limit by beam width
            if len(selected_nodes) > config.beam_width:
                node_confidences = [
                    (node_id, tree[node_id].confidence) for node_id in selected_nodes
                ]
                node_confidences.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = [node_id for node_id, _ in node_confidences[:config.beam_width]]
            
            return selected_nodes
    
    def _apply_pruning(self, tree: Dict[int, ReasoningNode], 
                      config: TreeReasoningConfig) -> int:
        """
        Apply pruning strategy to the tree
        
        Args:
            tree: Current reasoning tree
            config: Tree reasoning configuration
            
        Returns:
            Number of nodes pruned
        """
        pruned_count = 0
        
        for node_id, node in tree.items():
            if node.is_pruned:
                continue
            
            should_prune = False
            
            if config.pruning_strategy == PruningStrategy.CONFIDENCE_BASED:
                should_prune = node.confidence < config.confidence_threshold
            
            elif config.pruning_strategy == PruningStrategy.DEPTH_LIMITED:
                should_prune = node.depth >= config.max_depth
            
            elif config.pruning_strategy == PruningStrategy.BEAM_SEARCH:
                # Prune nodes that are not in the top-k at their depth
                if node.parent_id is not None:
                    parent = tree[node.parent_id]
                    siblings = [tree[child_id] for child_id in parent.children]
                    siblings.sort(key=lambda x: x.confidence, reverse=True)
                    top_siblings = siblings[:config.beam_width]
                    should_prune = node not in top_siblings
            
            else:  # ADAPTIVE
                # Prune based on multiple criteria
                should_prune = (
                    node.confidence < config.confidence_threshold or
                    node.depth >= config.max_depth or
                    (node.parent_id is not None and 
                     len(tree[node.parent_id].children) > config.max_branches_per_node)
                )
            
            if should_prune:
                node.is_pruned = True
                node.node_type = NodeType.PRUNED
                pruned_count += 1
        
        return pruned_count
    
    def _find_best_path(self, tree: Dict[int, ReasoningNode], 
                       config: TreeReasoningConfig) -> List[int]:
        """
        Find the best path through the reasoning tree
        
        Args:
            tree: Complete reasoning tree
            config: Tree reasoning configuration
            
        Returns:
            List of node IDs representing the best path
        """
        if not tree:
            return []
        
        # Find all leaf nodes
        leaf_nodes = [
            node_id for node_id, node in tree.items()
            if node.node_type == NodeType.LEAF and not node.is_pruned
        ]
        
        if not leaf_nodes:
            # No leaves, find deepest nodes
            max_depth = max(node.depth for node in tree.values())
            leaf_nodes = [
                node_id for node_id, node in tree.items()
                if node.depth == max_depth and not node.is_pruned
            ]
        
        if not leaf_nodes:
            return []
        
        # Find best leaf node
        best_leaf = max(leaf_nodes, key=lambda node_id: tree[node_id].confidence)
        
        # Trace path back to root
        path = []
        current_node_id = best_leaf
        
        while current_node_id is not None:
            path.append(current_node_id)
            current_node_id = tree[current_node_id].parent_id
        
        path.reverse()  # Reverse to get root-to-leaf path
        return path
    
    async def _generate_final_answer(self, tree: Dict[int, ReasoningNode], 
                                   best_path: List[int],
                                   original_request: CompletionRequest) -> str:
        """
        Generate final answer from the best path
        
        Args:
            tree: Complete reasoning tree
            best_path: Best path through the tree
            original_request: Original completion request
            
        Returns:
            Final synthesized answer
        """
        if not best_path:
            return "No reasoning path found"
        
        # Collect reasoning from best path
        path_reasoning = []
        for node_id in best_path:
            if node_id in tree:
                path_reasoning.append(f"Step {tree[node_id].depth}: {tree[node_id].reasoning}")
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Original question: {original_request.prompt}
        
        Best reasoning path through the tree:
        {chr(10).join(path_reasoning)}
        
        Based on this reasoning path, provide a clear, final answer that:
        1. Directly addresses the original question
        2. Synthesizes the key insights from the reasoning path
        3. Is well-structured and easy to understand
        4. Reflects the confidence level from the reasoning
        
        Provide your final answer:
        """
        
        synthesis_request = CompletionRequest(
            prompt=synthesis_prompt,
            model=original_request.model,
            temperature=0.3,
            max_tokens=original_request.max_tokens
        )
        
        try:
            # Use the provider from the root node
            root_provider = tree[best_path[0]].provider_used if best_path else ProviderType.OPENAI
            response = await self.llm_manager.generate_completion(
                synthesis_request, preferred_provider=root_provider
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Final answer generation failed", error=str(e))
            # Fallback to last node in path
            if best_path:
                return tree[best_path[-1]].reasoning
            return "Answer generation failed"
    
    def _calculate_tree_metrics(self, tree: Dict[int, ReasoningNode], 
                              best_path: List[int]) -> Dict[str, Any]:
        """
        Calculate metrics for the reasoning tree
        
        Args:
            tree: Complete reasoning tree
            best_path: Best path through the tree
            
        Returns:
            Dictionary with tree metrics
        """
        if not tree:
            return {}
        
        # Basic metrics
        total_nodes = len(tree)
        max_depth = max(node.depth for node in tree.values())
        avg_confidence = sum(node.confidence for node in tree.values()) / total_nodes
        
        # Branching metrics
        branching_factors = [len(node.children) for node in tree.values() if node.children]
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        
        # Path metrics
        path_length = len(best_path)
        path_confidence = sum(tree[node_id].confidence for node_id in best_path) / path_length if best_path else 0
        
        # Pruning metrics
        pruned_nodes = sum(1 for node in tree.values() if node.is_pruned)
        pruning_rate = pruned_nodes / total_nodes if total_nodes > 0 else 0
        
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "average_confidence": avg_confidence,
            "average_branching_factor": avg_branching,
            "best_path_length": path_length,
            "best_path_confidence": path_confidence,
            "pruned_nodes": pruned_nodes,
            "pruning_rate": pruning_rate,
            "tree_density": total_nodes / (2 ** max_depth) if max_depth > 0 else 0
        }
    
    async def _select_provider(self, request: CompletionRequest) -> ProviderType:
        """
        Select provider for reasoning
        
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
        
        # Prefer reasoning models
        priority_order = [ProviderType.OPENAI, ProviderType.CLAUDE, ProviderType.GEMINI]
        
        for provider_type in priority_order:
            if provider_type in available_providers:
                return provider_type
        
        return available_providers[0]
    
    def get_reasoning_history(self, limit: int = 10) -> List[TreeReasoningResult]:
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
        avg_nodes = sum(result.nodes_explored for result in self.reasoning_history) / total_sessions
        avg_depth = sum(result.tree_metrics.get("max_depth", 0) for result in self.reasoning_history) / total_sessions
        avg_time = sum(result.total_time for result in self.reasoning_history) / total_sessions
        avg_pruning = sum(result.nodes_pruned for result in self.reasoning_history) / total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "average_nodes_explored": avg_nodes,
            "average_max_depth": avg_depth,
            "average_time": avg_time,
            "average_nodes_pruned": avg_pruning,
            "average_tree_density": sum(
                result.tree_metrics.get("tree_density", 0) for result in self.reasoning_history
            ) / total_sessions
        }
