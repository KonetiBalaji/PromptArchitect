"""
Base LLM Provider Interface
Author: Balaji Koneti

Abstract base class defining the interface for all LLM providers.
This ensures consistent behavior across different LLM services (OpenAI, Claude, Gemini).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from datetime import datetime


class ProviderType(str, Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ModelInfo(BaseModel):
    """Model information container"""
    name: str = Field(..., description="Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet')")
    provider: ProviderType = Field(..., description="Provider type")
    max_tokens: int = Field(..., description="Maximum tokens supported")
    cost_per_1k_tokens: float = Field(..., description="Cost per 1000 tokens")
    context_window: int = Field(..., description="Context window size")
    supports_streaming: bool = Field(default=True, description="Whether model supports streaming")
    supports_functions: bool = Field(default=False, description="Whether model supports function calling")
    
    # Reasoning model capabilities
    supports_reasoning: bool = Field(default=False, description="Whether model supports reasoning")
    supports_extended_thinking: bool = Field(default=False, description="Whether model supports extended thinking")
    supports_grounding: bool = Field(default=False, description="Whether model supports grounding and search")
    reasoning_effort_levels: List[str] = Field(default_factory=list, description="Available reasoning effort levels")
    max_reasoning_tokens: Optional[int] = Field(default=None, description="Maximum reasoning tokens")
    reasoning_cost_multiplier: float = Field(default=1.0, description="Cost multiplier for reasoning")


class CompletionRequest(BaseModel):
    """Standardized request format for all providers"""
    prompt: str = Field(..., description="Input prompt text")
    model: str = Field(..., description="Model name to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream response")
    system_message: Optional[str] = Field(default=None, description="System message for chat models")
    
    # Reasoning-specific parameters
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort level (low, medium, high, max)")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum completion tokens for reasoning models")
    extended_thinking: bool = Field(default=False, description="Enable extended thinking mode")
    thinking_mode: bool = Field(default=False, description="Enable thinking mode for reasoning")
    grounding: bool = Field(default=False, description="Enable grounding and search capabilities")
    search_web: bool = Field(default=False, description="Enable web search for grounding")


class CompletionResponse(BaseModel):
    """Standardized response format from all providers"""
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model used for generation")
    provider: ProviderType = Field(..., description="Provider that generated the response")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Reason for completion")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Reasoning-specific response fields
    reasoning_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="Step-by-step reasoning process")
    thinking_blocks: Optional[List[str]] = Field(default=None, description="Thinking blocks from extended thinking")
    reasoning_trace: Optional[str] = Field(default=None, description="Full reasoning trace")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score for the response")
    verification_results: Optional[Dict[str, Any]] = Field(default=None, description="Reasoning verification results")
    grounded_sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sources used for grounding")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers
    
    All LLM providers must implement this interface to ensure
    consistent behavior across different services.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the provider with API credentials
        
        Args:
            api_key: API key for the provider
            base_url: Optional custom base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        self._models_cache: Dict[str, ModelInfo] = {}
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider client
        This method should set up the API client and validate credentials
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available models for this provider
        
        Returns:
            List of ModelInfo objects describing available models
        """
        pass
    
    @abstractmethod
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using the specified model
        
        Args:
            request: CompletionRequest with all necessary parameters
            
        Returns:
            CompletionResponse with generated content and metadata
            
        Raises:
            ProviderError: If the request fails
        """
        pass
    
    @abstractmethod
    async def get_token_count(self, text: str, model: str) -> int:
        """
        Get the token count for a given text and model
        
        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization
            
        Returns:
            Number of tokens in the text
        """
        pass
    
    @abstractmethod
    async def validate_response(self, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a response against given criteria
        
        Args:
            response: Response text to validate
            criteria: Validation criteria dictionary
            
        Returns:
            Dictionary with validation results
        """
        pass
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if model not found
        """
        if not self._models_cache:
            await self.get_available_models()
        return self._models_cache.get(model_name)
    
    async def estimate_cost(self, request: CompletionRequest) -> float:
        """
        Estimate the cost for a completion request
        
        Args:
            request: CompletionRequest to estimate cost for
            
        Returns:
            Estimated cost in USD
        """
        model_info = await self.get_model_info(request.model)
        if not model_info:
            return 0.0
        
        # Estimate input tokens (rough approximation)
        input_tokens = await self.get_token_count(request.prompt, request.model)
        
        # Estimate output tokens based on model type
        if model_info.supports_reasoning:
            # For reasoning models, use max_completion_tokens or model default
            output_tokens = request.max_completion_tokens or model_info.max_reasoning_tokens or 1000
        else:
            # For standard models, use max_tokens or model default
            output_tokens = request.max_tokens or min(1000, model_info.max_tokens)
        
        # Calculate base cost
        input_cost = (input_tokens / 1000) * model_info.cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * model_info.cost_per_1k_tokens
        
        # Apply reasoning cost multiplier if applicable
        total_cost = (input_cost + output_cost) * model_info.reasoning_cost_multiplier
        
        return total_cost
    
    def __str__(self) -> str:
        """String representation of the provider"""
        return f"{self.provider_type.value.title()}Provider"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(provider_type={self.provider_type})"


class ProviderError(Exception):
    """Base exception for provider-related errors"""
    
    def __init__(self, message: str, provider: Optional[BaseLLMProvider] = None, 
                 original_error: Optional[Exception] = None):
        """
        Initialize provider error
        
        Args:
            message: Error message
            provider: Provider that caused the error
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
        self.provider_type = provider.provider_type if provider else None


class RateLimitError(ProviderError):
    """Exception raised when rate limits are exceeded"""
    pass


class AuthenticationError(ProviderError):
    """Exception raised when authentication fails"""
    pass


class ModelNotFoundError(ProviderError):
    """Exception raised when requested model is not found"""
    pass


class QuotaExceededError(ProviderError):
    """Exception raised when quota is exceeded"""
    pass
