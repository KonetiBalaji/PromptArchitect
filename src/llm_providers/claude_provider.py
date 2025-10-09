"""
Anthropic Claude Provider Implementation
Author: Balaji Koneti

Implementation of the BaseLLMProvider interface for Anthropic's Claude models.
Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other Claude models.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import anthropic
from anthropic.types import Message, MessageParam

from .base import (
    BaseLLMProvider, ProviderType, ModelInfo, CompletionRequest, 
    CompletionResponse, ProviderError, RateLimitError, AuthenticationError,
    ModelNotFoundError, QuotaExceededError
)


class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation
    
    Supports all Claude models including Claude 3.5 Sonnet, Claude 3 Opus,
    and Claude 3 Haiku with full feature support.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Claude provider
        
        Args:
            api_key: Anthropic API key
            base_url: Optional custom base URL (not typically used for Claude)
        """
        super().__init__(api_key, base_url)
        self._client: Optional[anthropic.AsyncAnthropic] = None
    
    @property
    def provider_type(self) -> ProviderType:
        """Return Claude as the provider type"""
        return ProviderType.CLAUDE
    
    async def initialize(self) -> None:
        """
        Initialize the Claude client and validate credentials
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        try:
            # Initialize Claude client with API key
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Validate credentials by making a test call
            await self._client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            
            # Cache available models
            await self.get_available_models()
            
        except Exception as e:
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(
                    f"Claude authentication failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"Failed to initialize Claude provider: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available Claude models
        
        Returns:
            List of ModelInfo objects for available models
        """
        if not self._client:
            await self.initialize()
        
        try:
            # Define Claude model information (costs and capabilities)
            model_specs = {
                # Latest Claude 3.7 Sonnet with Extended Thinking
                "claude-3-7-sonnet-20241218": ModelInfo(
                    name="claude-3-7-sonnet-20241218",
                    provider=ProviderType.CLAUDE,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.003,  # $3 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=True,
                    supports_extended_thinking=True,
                    supports_grounding=False,
                    reasoning_effort_levels=["low", "medium", "high"],
                    max_reasoning_tokens=8192,
                    reasoning_cost_multiplier=1.0
                ),
                # Standard Claude Models
                "claude-3-5-sonnet-20241022": ModelInfo(
                    name="claude-3-5-sonnet-20241022",
                    provider=ProviderType.CLAUDE,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.003,  # $3 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "claude-3-5-sonnet-20240620": ModelInfo(
                    name="claude-3-5-sonnet-20240620",
                    provider=ProviderType.CLAUDE,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.003,  # $3 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "claude-3-opus-20240229": ModelInfo(
                    name="claude-3-opus-20240229",
                    provider=ProviderType.CLAUDE,
                    max_tokens=4096,
                    cost_per_1k_tokens=0.015,  # $15 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "claude-3-sonnet-20240229": ModelInfo(
                    name="claude-3-sonnet-20240229",
                    provider=ProviderType.CLAUDE,
                    max_tokens=4096,
                    cost_per_1k_tokens=0.003,  # $3 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "claude-3-haiku-20240307": ModelInfo(
                    name="claude-3-haiku-20240307",
                    provider=ProviderType.CLAUDE,
                    max_tokens=4096,
                    cost_per_1k_tokens=0.00025,  # $0.25 per 1M tokens
                    context_window=200000,
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                )
            }
            
            # For Claude, we'll return all defined models as they're generally available
            # In a real implementation, you might want to test each model
            available_models = list(model_specs.values())
            
            # Cache the models
            for model in available_models:
                self._models_cache[model.name] = model
            
            return available_models
            
        except Exception as e:
            raise ProviderError(
                f"Failed to get Claude models: {str(e)}", 
                provider=self, 
                original_error=e
            )
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Claude's messages API
        
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
            # Prepare messages for Claude API
            messages: List[MessageParam] = []
            
            # Add system message if provided (Claude supports system messages)
            system_message = request.system_message
            
            # Add user message
            messages.append({"role": "user", "content": request.prompt})
            
            # Get model info to check capabilities
            model_info = await self.get_model_info(request.model)
            supports_extended_thinking = model_info and model_info.supports_extended_thinking
            
            # Prepare completion parameters
            completion_params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1024,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            # Add system message if provided
            if system_message:
                completion_params["system"] = system_message
            
            # Add stop sequences if provided
            if request.stop:
                completion_params["stop_sequences"] = request.stop
            
            # Add extended thinking if supported and requested
            if supports_extended_thinking and request.extended_thinking:
                completion_params["extended_thinking"] = True
            
            # Make the API call
            response = await self._client.messages.create(**completion_params)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract content and metadata
            content = response.content[0].text if response.content else ""
            finish_reason = response.stop_reason or "unknown"
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Extract thinking blocks if extended thinking was used
            thinking_blocks = None
            reasoning_trace = None
            confidence_score = None
            
            if supports_extended_thinking and request.extended_thinking:
                # Extract thinking blocks from response
                if hasattr(response, 'thinking') and response.thinking:
                    thinking_blocks = [response.thinking]
                    reasoning_trace = response.thinking
                    confidence_score = 0.9  # High confidence for extended thinking
                elif hasattr(response, 'content') and len(response.content) > 1:
                    # Sometimes thinking is in separate content blocks
                    thinking_blocks = []
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == 'thinking':
                            thinking_blocks.append(block.text)
                    
                    if thinking_blocks:
                        reasoning_trace = "\n".join(thinking_blocks)
                        confidence_score = 0.9
            
            # Create standardized response
            return CompletionResponse(
                content=content,
                model=request.model,
                provider=ProviderType.CLAUDE,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                thinking_blocks=thinking_blocks,
                reasoning_trace=reasoning_trace,
                confidence_score=confidence_score,
                metadata={
                    "claude_response_id": response.id,
                    "claude_model": response.model,
                    "claude_role": response.role,
                    "claude_stop_reason": response.stop_reason,
                    "supports_extended_thinking": supports_extended_thinking,
                    "extended_thinking_used": request.extended_thinking
                }
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg:
                raise RateLimitError(
                    f"Claude rate limit exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "model" in error_msg and "not found" in error_msg:
                raise ModelNotFoundError(
                    f"Claude model not found: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "quota" in error_msg or "billing" in error_msg:
                raise QuotaExceededError(
                    f"Claude quota exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"Claude completion failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_token_count(self, text: str, model: str) -> int:
        """
        Get token count for text using Claude's tokenizer
        
        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization
            
        Returns:
            Number of tokens in the text
        """
        try:
            if not self._client:
                await self.initialize()
            
            # Use Claude's count_tokens method
            token_count = await self._client.count_tokens(text)
            return token_count
            
        except Exception as e:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    async def validate_response(self, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Claude response against given criteria
        
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
            
            # Claude-specific validations
            if "claude_safety" in criteria and criteria["claude_safety"]:
                # Check for potential safety issues (basic check)
                safety_keywords = ["harmful", "dangerous", "illegal", "unethical"]
                if any(keyword in response.lower() for keyword in safety_keywords):
                    validation_results["warnings"].append("Response may contain potentially sensitive content")
                    validation_results["score"] -= 0.1
            
            # Calculate final score
            validation_results["score"] = max(0.0, validation_results["score"])
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
