"""
Chain-of-Thought Reasoning Module
Author: Balaji Koneti

This module provides advanced chain-of-thought reasoning implementations
including sequential, parallel, tree-based, and reflective reasoning strategies.
"""

from .sequential_chain import SequentialChain, SequentialChainConfig, SequentialChainResult
from .parallel_chain import ParallelChain, ParallelChainConfig, ParallelChainResult
from .tree_reasoning import TreeReasoning, TreeReasoningConfig, TreeReasoningResult
from .reflection import Reflection, ReflectionConfig, ReflectionResult

__all__ = [
    # Sequential Chain
    "SequentialChain",
    "SequentialChainConfig", 
    "SequentialChainResult",
    
    # Parallel Chain
    "ParallelChain",
    "ParallelChainConfig",
    "ParallelChainResult",
    
    # Tree Reasoning
    "TreeReasoning",
    "TreeReasoningConfig",
    "TreeReasoningResult",
    
    # Reflection
    "Reflection",
    "ReflectionConfig",
    "ReflectionResult",
]
