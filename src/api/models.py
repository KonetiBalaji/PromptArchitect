"""
API Models
Author: Balaji Koneti

Pydantic models for API request/response validation and serialization.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# Enums
class ProviderType(str, Enum):
    """LLM Provider types"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ReasoningStrategy(str, Enum):
    """Reasoning strategy types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    TREE = "tree"
    REFLECTION = "reflection"


class TemplateCategory(str, Enum):
    """Template categories"""
    RESEARCH = "research"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    BUSINESS = "business"
    MARKETING = "marketing"
    CUSTOM = "custom"


class TemplateComplexity(str, Enum):
    """Template complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


# Request Models
class CompletionRequestModel(BaseModel):
    """Completion request model"""
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
    preferred_provider: Optional[ProviderType] = Field(default=None, description="Preferred LLM provider")
    
    # Reasoning-specific parameters
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort level")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum completion tokens for reasoning models")
    extended_thinking: bool = Field(default=False, description="Enable extended thinking mode")
    thinking_mode: bool = Field(default=False, description="Enable thinking mode for reasoning")
    grounding: bool = Field(default=False, description="Enable grounding and search capabilities")
    search_web: bool = Field(default=False, description="Enable web search for grounding")


class ReasoningRequestModel(BaseModel):
    """Reasoning request model"""
    prompt: str = Field(..., description="Input prompt for reasoning")
    model: str = Field(..., description="Model name to use")
    strategy: ReasoningStrategy = Field(default=ReasoningStrategy.SEQUENTIAL, description="Reasoning strategy")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    max_steps: int = Field(default=5, ge=1, le=20, description="Maximum reasoning steps")
    verification_enabled: bool = Field(default=True, description="Enable step verification")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold")
    timeout: float = Field(default=60.0, ge=10.0, le=300.0, description="Timeout in seconds")
    parallel_workers: Optional[int] = Field(default=None, ge=1, le=10, description="Number of parallel workers")
    
    # Reasoning-specific parameters
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort level")
    extended_thinking: bool = Field(default=False, description="Enable extended thinking mode")
    thinking_mode: bool = Field(default=True, description="Enable thinking mode for reasoning")


class TemplateGenerationRequestModel(BaseModel):
    """Template generation request model"""
    category: TemplateCategory = Field(..., description="Template category")
    complexity: TemplateComplexity = Field(default=TemplateComplexity.MODERATE, description="Template complexity")
    use_case: str = Field(..., description="Specific use case description")
    requirements: Optional[List[str]] = Field(default=None, description="Specific requirements")
    constraints: Optional[List[str]] = Field(default=None, description="Constraints to consider")
    target_audience: Optional[str] = Field(default=None, description="Target audience")
    language: str = Field(default="en", description="Template language")
    reasoning_enabled: bool = Field(default=True, description="Enable reasoning integration")
    custom_instructions: Optional[str] = Field(default=None, description="Custom instructions")


class TemplateSelectionRequestModel(BaseModel):
    """Template selection request model"""
    user_input: str = Field(..., description="User input to analyze")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    preferred_category: Optional[TemplateCategory] = Field(default=None, description="Preferred template category")
    complexity_preference: Optional[TemplateComplexity] = Field(default=None, description="Preferred complexity")


# Response Models
class CompletionResponseModel(BaseModel):
    """Completion response model"""
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model used for generation")
    provider: ProviderType = Field(..., description="Provider that generated the response")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Reason for completion")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: datetime = Field(..., description="Response timestamp")
    
    # Reasoning-specific response fields
    reasoning_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="Step-by-step reasoning process")
    thinking_blocks: Optional[List[str]] = Field(default=None, description="Thinking blocks from extended thinking")
    reasoning_trace: Optional[str] = Field(default=None, description="Full reasoning trace")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score for the response")
    verification_results: Optional[Dict[str, Any]] = Field(default=None, description="Reasoning verification results")
    grounded_sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sources used for grounding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReasoningResponseModel(BaseModel):
    """Reasoning response model"""
    strategy: str = Field(..., description="Reasoning strategy used")
    steps: List[Dict[str, Any]] = Field(..., description="Reasoning steps")
    final_answer: str = Field(..., description="Final reasoning result")
    confidence: float = Field(..., description="Overall confidence score")
    total_time: float = Field(..., description="Total reasoning time")
    provider_used: str = Field(..., description="Provider used for reasoning")
    model_used: str = Field(..., description="Model used for reasoning")
    verification_passed: bool = Field(..., description="Whether verification passed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TemplateResponseModel(BaseModel):
    """Template response model"""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    complexity: TemplateComplexity = Field(..., description="Template complexity")
    template_content: str = Field(..., description="Template content")
    variables: List[str] = Field(..., description="Template variables")
    reasoning_integration: bool = Field(..., description="Whether reasoning is integrated")
    created_at: datetime = Field(..., description="Creation timestamp")
    version: str = Field(..., description="Template version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TemplateSelectionResponseModel(BaseModel):
    """Template selection response model"""
    template_id: str = Field(..., description="Selected template ID")
    confidence: float = Field(..., description="Selection confidence")
    detected_intent: str = Field(..., description="Detected user intent")
    reasoning: str = Field(..., description="Selection reasoning")
    context_analysis: Dict[str, Any] = Field(..., description="Context analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AnalyticsResponseModel(BaseModel):
    """Analytics response model"""
    reasoning_stats: Dict[str, Any] = Field(..., description="Reasoning statistics")
    template_stats: Dict[str, Any] = Field(..., description="Template statistics")
    registry_analytics: Dict[str, Any] = Field(..., description="Template registry analytics")
    provider_status: Dict[str, Any] = Field(..., description="Provider status information")
    timestamp: datetime = Field(..., description="Analytics timestamp")


class HealthResponseModel(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, bool] = Field(..., description="Component status")


# Error Models
class ErrorResponseModel(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


# Utility Models
class PaginationModel(BaseModel):
    """Pagination model"""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    total: Optional[int] = Field(default=None, description="Total number of items")


class PaginatedResponseModel(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(..., description="List of items")
    pagination: PaginationModel = Field(..., description="Pagination information")


# Configuration Models
class APIConfigModel(BaseModel):
    """API configuration model"""
    title: str = Field(default="Visionary Prompt Architect API")
    description: str = Field(default="Advanced AI-powered prompt engineering")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["*"])
    rate_limit_enabled: bool = Field(default=True)
    auth_enabled: bool = Field(default=True)
    docs_enabled: bool = Field(default=True)
