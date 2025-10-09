"""
Google Gemini Provider Implementation
Author: Balaji Koneti

Implementation of the BaseLLMProvider interface for Google's Gemini models.
Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other Gemini models.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from .base import (
    BaseLLMProvider, ProviderType, ModelInfo, CompletionRequest, 
    CompletionResponse, ProviderError, RateLimitError, AuthenticationError,
    ModelNotFoundError, QuotaExceededError
)


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation
    
    Supports all Gemini models including Gemini 1.5 Pro, Gemini 1.5 Flash,
    and other Gemini models with full feature support.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Gemini provider
        
        Args:
            api_key: Google AI API key
            base_url: Optional custom base URL (not typically used for Gemini)
        """
        super().__init__(api_key, base_url)
        self._client: Optional[genai.GenerativeModel] = None
        self._configured = False
    
    @property
    def provider_type(self) -> ProviderType:
        """Return Gemini as the provider type"""
        return ProviderType.GEMINI
    
    async def initialize(self) -> None:
        """
        Initialize the Gemini client and validate credentials
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        try:
            # Configure Gemini with API key
            genai.configure(api_key=self.api_key)
            self._configured = True
            
            # Validate credentials by making a test call
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await model.generate_content_async("test")
            
            # Cache available models
            await self.get_available_models()
            
        except Exception as e:
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(
                    f"Gemini authentication failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"Failed to initialize Gemini provider: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available Gemini models
        
        Returns:
            List of ModelInfo objects for available models
        """
        if not self._configured:
            await self.initialize()
        
        try:
            # Define Gemini model information (costs and capabilities)
            model_specs = {
                # Latest Gemini 2.0 Flash Thinking
                "gemini-2.0-flash-thinking-exp": ModelInfo(
                    name="gemini-2.0-flash-thinking-exp",
                    provider=ProviderType.GEMINI,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.00015,  # $0.15 per 1M tokens
                    context_window=1000000,  # 1M tokens
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=True,
                    supports_extended_thinking=False,
                    supports_grounding=True,
                    reasoning_effort_levels=["low", "medium", "high"],
                    max_reasoning_tokens=8192,
                    reasoning_cost_multiplier=1.0
                ),
                # Standard Gemini Models
                "gemini-1.5-pro": ModelInfo(
                    name="gemini-1.5-pro",
                    provider=ProviderType.GEMINI,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.00125,  # $1.25 per 1M tokens
                    context_window=2000000,  # 2M tokens
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gemini-1.5-flash": ModelInfo(
                    name="gemini-1.5-flash",
                    provider=ProviderType.GEMINI,
                    max_tokens=8192,
                    cost_per_1k_tokens=0.000075,  # $0.075 per 1M tokens
                    context_window=1000000,  # 1M tokens
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gemini-1.0-pro": ModelInfo(
                    name="gemini-1.0-pro",
                    provider=ProviderType.GEMINI,
                    max_tokens=2048,
                    cost_per_1k_tokens=0.0005,  # $0.5 per 1M tokens
                    context_window=30720,  # 30K tokens
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                ),
                "gemini-pro": ModelInfo(
                    name="gemini-pro",
                    provider=ProviderType.GEMINI,
                    max_tokens=2048,
                    cost_per_1k_tokens=0.0005,  # $0.5 per 1M tokens
                    context_window=30720,  # 30K tokens
                    supports_streaming=True,
                    supports_functions=False,
                    supports_reasoning=False,
                    supports_extended_thinking=False,
                    supports_grounding=False
                )
            }
            
            # For Gemini, we'll return all defined models as they're generally available
            # In a real implementation, you might want to test each model
            available_models = list(model_specs.values())
            
            # Cache the models
            for model in available_models:
                self._models_cache[model.name] = model
            
            return available_models
            
        except Exception as e:
            raise ProviderError(
                f"Failed to get Gemini models: {str(e)}", 
                provider=self, 
                original_error=e
            )
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Gemini's generate content API
        
        Args:
            request: CompletionRequest with all necessary parameters
            
        Returns:
            CompletionResponse with generated content and metadata
            
        Raises:
            ProviderError: If the request fails
            RateLimitError: If rate limits are exceeded
            ModelNotFoundError: If the model is not found
        """
        if not self._configured:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Create the model instance
            model = genai.GenerativeModel(request.model)
            
            # Prepare the prompt
            prompt = request.prompt
            
            # Add system message if provided (Gemini doesn't have separate system messages)
            if request.system_message:
                prompt = f"System: {request.system_message}\n\nUser: {prompt}"
            
            # Get model info to check capabilities
            model_info = await self.get_model_info(request.model)
            supports_reasoning = model_info and model_info.supports_reasoning
            supports_grounding = model_info and model_info.supports_grounding
            
            # Prepare generation configuration
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_tokens or 1024,
                stop_sequences=request.stop if request.stop else None
            )
            
            # Add reasoning and grounding parameters if supported
            if supports_reasoning and request.thinking_mode:
                # Enable thinking mode for reasoning models
                generation_config.thinking_mode = True
            
            if supports_grounding and request.grounding:
                # Enable grounding and search
                generation_config.grounding = True
                if request.search_web:
                    generation_config.web_search = True
            
            # Make the API call
            response: GenerateContentResponse = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract content and metadata
            content = response.text if response.text else ""
            finish_reason = "stop"  # Gemini doesn't provide detailed finish reasons
            
            # Extract usage information (Gemini provides limited usage info)
            usage = {
                "prompt_tokens": 0,  # Gemini doesn't provide input token count
                "completion_tokens": 0,  # Gemini doesn't provide output token count
                "total_tokens": 0  # Will be estimated
            }
            
            # Estimate token usage
            if content:
                estimated_tokens = await self.get_token_count(content, request.model)
                usage["completion_tokens"] = estimated_tokens
                usage["total_tokens"] = estimated_tokens
            
            # Extract reasoning and grounding information
            reasoning_trace = None
            grounded_sources = None
            confidence_score = None
            
            if supports_reasoning and request.thinking_mode:
                # Extract reasoning trace from response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        # Look for reasoning in content parts
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and "thinking" in part.text.lower():
                                reasoning_trace = part.text
                                confidence_score = 0.8
                                break
            
            if supports_grounding and request.grounding:
                # Extract grounded sources
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        grounded_sources = []
                        for source in candidate.grounding_metadata.grounding_chunks:
                            grounded_sources.append({
                                "source": source.web.uri if hasattr(source, 'web') else "unknown",
                                "title": source.web.title if hasattr(source, 'web') else "unknown",
                                "snippet": source.content if hasattr(source, 'content') else ""
                            })
            
            # Create standardized response
            return CompletionResponse(
                content=content,
                model=request.model,
                provider=ProviderType.GEMINI,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                reasoning_trace=reasoning_trace,
                confidence_score=confidence_score,
                grounded_sources=grounded_sources,
                metadata={
                    "gemini_response_id": str(response.candidates[0].index) if response.candidates else None,
                    "gemini_model": request.model,
                    "gemini_finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                    "gemini_safety_ratings": [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        } for rating in response.candidates[0].safety_ratings
                    ] if response.candidates and response.candidates[0].safety_ratings else [],
                    "supports_reasoning": supports_reasoning,
                    "supports_grounding": supports_grounding,
                    "thinking_mode_used": request.thinking_mode,
                    "grounding_used": request.grounding
                }
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "quota" in error_msg:
                raise RateLimitError(
                    f"Gemini rate limit exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "model" in error_msg and "not found" in error_msg:
                raise ModelNotFoundError(
                    f"Gemini model not found: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            elif "billing" in error_msg or "payment" in error_msg:
                raise QuotaExceededError(
                    f"Gemini quota exceeded: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
            else:
                raise ProviderError(
                    f"Gemini completion failed: {str(e)}", 
                    provider=self, 
                    original_error=e
                )
    
    async def get_token_count(self, text: str, model: str) -> int:
        """
        Get token count for text using Gemini's tokenizer
        
        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization
            
        Returns:
            Number of tokens in the text
        """
        try:
            if not self._configured:
                await self.initialize()
            
            # Use Gemini's count_tokens method
            model_instance = genai.GenerativeModel(model)
            token_count = await model_instance.count_tokens_async(text)
            return token_count.total_tokens
            
        except Exception as e:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    async def validate_response(self, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Gemini response against given criteria
        
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
            
            # Gemini-specific validations
            if "gemini_safety" in criteria and criteria["gemini_safety"]:
                # Check for potential safety issues (basic check)
                safety_keywords = ["harmful", "dangerous", "illegal", "unethical"]
                if any(keyword in response.lower() for keyword in safety_keywords):
                    validation_results["warnings"].append("Response may contain potentially sensitive content")
                    validation_results["score"] -= 0.1
            
            # Check for Gemini-specific content policies
            if "content_policy" in criteria and criteria["content_policy"]:
                policy_keywords = ["violence", "hate", "harassment", "dangerous"]
                if any(keyword in response.lower() for keyword in policy_keywords):
                    validation_results["warnings"].append("Response may violate content policies")
                    validation_results["score"] -= 0.2
            
            # Calculate final score
            validation_results["score"] = max(0.0, validation_results["score"])
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
