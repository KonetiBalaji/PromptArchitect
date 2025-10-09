"""
OpenAI Provider Implementation
Author: Balaji Koneti

Implementation of the BaseLLMProvider interface for OpenAI's GPT models.
Supports GPT-4, GPT-4o, GPT-3.5-turbo and other OpenAI models.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .base import (
    BaseLLMProvider, ProviderType, ModelInfo, CompletionRequest, 
    CompletionResponse, ProviderError, RateLimitError, AuthenticationError,
    ModelNotFoundError, QuotaExceededError
)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation for GPT models
    
    Supports all OpenAI chat completion models including GPT-4, GPT-4o,
    and GPT-3.5-turbo with full feature support.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL (for Azure OpenAI or other endpoints)
        """
        super().__init__(api_key, base_url)
        self._client: Optional[AsyncOpenAI] = None
    
    @property
    def provider_type(self) -> ProviderType:
        """Return OpenAI as the provider type"""
        return ProviderType.OPENAI
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI client and validate credentials
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        try:
            # Initialize OpenAI client with API key and optional base URL
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Validate credentials by making a test call
            await self._client.models.list()
            
            # Cache available models
            await self.get_available_models()
            
        except Exception as e:
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(
                    f"OpenAI authentication failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"Failed to initialize OpenAI provider: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available OpenAI models
        
        Returns:
            List of ModelInfo objects for available models
        """
        if not self._client:
            await self.initialize()
        
        try:
            # Get models from OpenAI API
            models_response = await self._client.models.list()
            
            # Define model information (costs and capabilities)
            model_specs = {
                # Latest Reasoning Models
                "o1-preview": ModelInfo(
                    name="o1-preview",
                    provider=ProviderType.OPENAI,
                    max_tokens=32768,
                    cost_per_1k_tokens=0.015,  # $15 per 1M tokens
                    context_window=128000,
                    supports_streaming=False,  # o1 models don't support streaming
                    supports_functions=False,
                    supports_reasoning=True,
                    supports_extended_thinking=False,
                    supports_grounding=False,
                    reasoning_effort_levels=["low", "medium", "high", "max"],
                    max_reasoning_tokens=32768,
                    reasoning_cost_multiplier=1.0
                ),
                "o1-mini": ModelInfo(
                    name="o1-mini",
                    provider=ProviderType.OPENAI,
                    max_tokens=16384,
                    cost_per_1k_tokens=0.003,  # $3 per 1M tokens
                    context_window=128000,
                    supports_streaming=False,
                    supports_functions=False,
                    supports_reasoning=True,
                    supports_extended_thinking=False,
                    supports_grounding=False,
                    reasoning_effort_levels=["low", "medium", "high", "max"],
                    max_reasoning_tokens=16384,
                    reasoning_cost_multiplier=1.0
                ),
                "o3-mini": ModelInfo(
                    name="o3-mini",
                    provider=ProviderType.OPENAI,
                    max_tokens=65536,
                    cost_per_1k_tokens=0.006,  # $6 per 1M tokens
                    context_window=200000,
                    supports_streaming=False,
                    supports_functions=False,
                    supports_reasoning=True,
                    supports_extended_thinking=False,
                    supports_grounding=False,
                    reasoning_effort_levels=["low", "medium", "high", "max"],
                    max_reasoning_tokens=65536,
                    reasoning_cost_multiplier=1.0
                ),
                # Standard Models
                "gpt-4o": ModelInfo(
                    name="gpt-4o",
                    provider=ProviderType.OPENAI,
                    max_tokens=128000,
                    cost_per_1k_tokens=0.005,  # $5 per 1M tokens
                    context_window=128000,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gpt-4o-mini": ModelInfo(
                    name="gpt-4o-mini",
                    provider=ProviderType.OPENAI,
                    max_tokens=128000,
                    cost_per_1k_tokens=0.00015,  # $0.15 per 1M tokens
                    context_window=128000,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gpt-4-turbo": ModelInfo(
                    name="gpt-4-turbo",
                    provider=ProviderType.OPENAI,
                    max_tokens=128000,
                    cost_per_1k_tokens=0.01,  # $10 per 1M tokens
                    context_window=128000,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gpt-4": ModelInfo(
                    name="gpt-4",
                    provider=ProviderType.OPENAI,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.03,  # $30 per 1M tokens
                    context_window=8192,
                    supports_streaming=True,
                    supports_functions=True,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gpt-3.5-turbo": ModelInfo(
                    name="gpt-3.5-turbo",
                    provider=ProviderType.OPENAI,
                    max_tokens=4096,
                    cost_per_1k_tokens=0.0015,  # $1.5 per 1M tokens
                    context_window=4096,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                )
            }
            
            # Filter to only include models that are actually available
            available_models = []
            for model in models_response.data:
                if model.id in model_specs:
                    available_models.append(model_specs[model.id])
                    self._models_cache[model.id] = model_specs[model.id]
            
            return available_models
            
        except Exception as e:
            raise ProviderError(
                f"Failed to get OpenAI models: {str(e)}", 
                provider=self, 
                original_error=e
            )
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using OpenAI's chat completion API
        
        Args:
            request: CompletionRequest with all necessary parameters
            
        Returns:
            CompletionResponse with generated content and metadata
            
        Raises:
            ProviderError: If the request fails
            RateLimitError: If rate limits are exceeded
            ModelNotFoundError: If the model is not found
        """
        if not self._client:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare messages for chat completion
            messages = []
            
            # Add system message if provided
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            
            # Add user message
            messages.append({"role": "user", "content": request.prompt})
            
            # Prepare completion parameters
            completion_params = {
                "model": request.model,
                "messages": messages,
                "stream": request.stream
            }
            
            # Get model info to check if it's a reasoning model
            model_info = await self.get_model_info(request.model)
            is_reasoning_model = model_info and model_info.supports_reasoning
            
            # Add parameters based on model type
            if is_reasoning_model:
                # Reasoning models (o1, o3) have different parameters
                if request.reasoning_effort:
                    completion_params["reasoning_effort"] = request.reasoning_effort
                if request.max_completion_tokens:
                    completion_params["max_completion_tokens"] = request.max_completion_tokens
                # Reasoning models don't support temperature, top_p, etc.
            else:
                # Standard models support all parameters
                completion_params.update({
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty
                })
                
                # Add optional parameters for standard models
                if request.max_tokens:
                    completion_params["max_tokens"] = request.max_tokens
                if request.stop:
                    completion_params["stop"] = request.stop
            
            # Make the API call
            response: ChatCompletion = await self._client.chat.completions.create(
                **completion_params
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract content and metadata
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Extract reasoning-specific information if available
            reasoning_steps = None
            thinking_blocks = None
            reasoning_trace = None
            confidence_score = None
            
            if is_reasoning_model:
                # For reasoning models, extract reasoning information
                # Note: OpenAI o1/o3 models include reasoning in the response
                # We'll parse this from the content or metadata
                if hasattr(response.choices[0], 'reasoning') and response.choices[0].reasoning:
                    reasoning_trace = response.choices[0].reasoning
                
                # Extract reasoning steps from content if available
                if "Let me think" in content or "Step" in content:
                    reasoning_steps = self._extract_reasoning_steps(content)
                
                # For o1/o3 models, the reasoning is typically embedded in the response
                # We'll extract it for better analysis
                if content:
                    reasoning_trace = content
                    confidence_score = 0.9  # High confidence for reasoning models
            
            # Create standardized response
            return CompletionResponse(
                content=content,
                model=request.model,
                provider=ProviderType.OPENAI,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                reasoning_steps=reasoning_steps,
                thinking_blocks=thinking_blocks,
                reasoning_trace=reasoning_trace,
                confidence_score=confidence_score,
                metadata={
                    "openai_response_id": response.id,
                    "openai_model": response.model,
                    "openai_created": response.created,
                    "is_reasoning_model": is_reasoning_model
                }
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg:
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "model" in error_msg and "not found" in error_msg:
                raise ModelNotFoundError(
                    f"OpenAI model not found: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "quota" in error_msg or "billing" in error_msg:
                raise QuotaExceededError(
                    f"OpenAI quota exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"OpenAI completion failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_token_count(self, text: str, model: str) -> int:
        """
        Get token count for text using OpenAI's tiktoken library
        
        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization
            
        Returns:
            Number of tokens in the text
        """
        try:
            import tiktoken
            
            # Get encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            
            # Count tokens
            token_count = len(encoding.encode(text))
            
            return token_count
            
        except Exception as e:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    async def validate_response(self, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate OpenAI response against given criteria
        
        Args:
            response: Response text to validate
            criteria: Validation criteria dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "score": 1.0
        }
        
        try:
            # Check response length
            if "max_length" in criteria:
                max_length = criteria["max_length"]
                if len(response) > max_length:
                    validation_results["errors"].append(f"Response too long: {len(response)} > {max_length}")
                    validation_results["is_valid"] = False
            
            # Check for required keywords
            if "required_keywords" in criteria:
                required_keywords = criteria["required_keywords"]
                missing_keywords = []
                for keyword in required_keywords:
                    if keyword.lower() not in response.lower():
                        missing_keywords.append(keyword)
                
                if missing_keywords:
                    validation_results["warnings"].append(f"Missing keywords: {missing_keywords}")
                    validation_results["score"] -= 0.1 * len(missing_keywords)
            
            # Check format compliance
            if "format" in criteria:
                expected_format = criteria["format"]
                if expected_format == "json":
                    try:
                        import json
                        json.loads(response)
                    except json.JSONDecodeError:
                        validation_results["errors"].append("Response is not valid JSON")
                        validation_results["is_valid"] = False
                elif expected_format == "markdown":
                    if not any(marker in response for marker in ["#", "**", "*", "```"]):
                        validation_results["warnings"].append("Response doesn't appear to be markdown")
                        validation_results["score"] -= 0.1
            
            # Calculate final score
            validation_results["score"] = max(0.0, validation_results["score"])
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def _extract_reasoning_steps(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract reasoning steps from content
        
        Args:
            content: Response content that may contain reasoning steps
            
        Returns:
            List of reasoning step dictionaries
        """
        reasoning_steps = []
        
        try:
            # Simple extraction of reasoning steps
            # This is a basic implementation - could be enhanced with more sophisticated parsing
            lines = content.split('\n')
            current_step = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for step indicators
                if line.startswith(('Step', 'Let me', 'First', 'Next', 'Then', 'Finally')):
                    if current_step:
                        reasoning_steps.append(current_step)
                    
                    current_step = {
                        "step_number": len(reasoning_steps) + 1,
                        "description": line,
                        "reasoning": line,
                        "confidence": 0.8
                    }
                elif current_step and line:
                    # Add to current step
                    current_step["reasoning"] += " " + line
            
            # Add the last step
            if current_step:
                reasoning_steps.append(current_step)
            
        except Exception as e:
            # If extraction fails, create a single step with the full content
            reasoning_steps = [{
                "step_number": 1,
                "description": "Complete reasoning process",
                "reasoning": content,
                "confidence": 0.7
            }]
        
        return reasoning_steps
