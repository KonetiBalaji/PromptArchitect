"""
LLM Manager for Multi-Provider Orchestration
Author: Balaji Koneti

Manages multiple LLM providers with fallback support, load balancing,
and intelligent provider selection based on cost, performance, and availability.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from .llm_providers.base import (
    BaseLLMProvider, ProviderType, CompletionRequest, CompletionResponse,
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError,
    QuotaExceededError
)
from .llm_providers.openai_provider import OpenAIProvider
from .llm_providers.claude_provider import ClaudeProvider
from .llm_providers.gemini_provider import GeminiProvider
from .cache.cache_manager import CacheManager, CacheConfig


logger = structlog.get_logger(__name__)


class ProviderStatus(str, Enum):
    """Status of a provider"""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class ProviderInfo(BaseModel):
    """Information about a provider"""
    provider: BaseLLMProvider = Field(..., description="Provider instance")
    status: ProviderStatus = Field(default=ProviderStatus.ACTIVE, description="Current status")
    last_used: Optional[float] = Field(default=None, description="Last usage timestamp")
    error_count: int = Field(default=0, description="Number of consecutive errors")
    success_count: int = Field(default=0, description="Number of successful requests")
    total_cost: float = Field(default=0.0, description="Total cost incurred")
    avg_response_time: float = Field(default=0.0, description="Average response time")
    priority: int = Field(default=1, description="Priority level (1=highest)")


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_COST = "least_cost"
    FASTEST = "fastest"
    PRIORITY = "priority"
    RANDOM = "random"


class LLMManagerConfig(BaseModel):
    """Configuration for LLM Manager"""
    default_provider: ProviderType = Field(default=ProviderType.OPENAI, description="Default provider")
    fallback_providers: List[ProviderType] = Field(default_factory=list, description="Fallback providers in order")
    load_balancing_strategy: LoadBalancingStrategy = Field(default=LoadBalancingStrategy.PRIORITY, description="Load balancing strategy")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_config: Optional[CacheConfig] = Field(default=None, description="Cache configuration")
    cost_threshold: float = Field(default=1.0, description="Maximum cost per request in USD")
    response_timeout: float = Field(default=30.0, description="Request timeout in seconds")


class LLMManager:
    """
    Multi-provider LLM manager with intelligent orchestration
    
    Features:
    - Multi-provider support with fallback
    - Load balancing strategies
    - Response caching
    - Cost optimization
    - Performance monitoring
    - Automatic retry with exponential backoff
    """
    
    def __init__(self, config: LLMManagerConfig):
        """
        Initialize LLM manager
        
        Args:
            config: LLM manager configuration
        """
        self.config = config
        self.providers: Dict[ProviderType, ProviderInfo] = {}
        self.cache_manager: Optional[CacheManager] = None
        self._initialized = False
    
    async def initialize(self, api_keys: Dict[ProviderType, str]) -> None:
        """
        Initialize all providers with API keys
        
        Args:
            api_keys: Dictionary mapping provider types to API keys
            
        Raises:
            ValueError: If required providers are not configured
        """
        try:
            # Initialize cache manager if enabled
            if self.config.enable_caching and self.config.cache_config:
                self.cache_manager = CacheManager(self.config.cache_config)
                await self.cache_manager.connect()
                logger.info("Cache manager initialized")
            
            # Initialize providers
            for provider_type, api_key in api_keys.items():
                if not api_key:
                    logger.warning("Skipping provider without API key", provider=provider_type)
                    continue
                
                try:
                    provider = await self._create_provider(provider_type, api_key)
                    await provider.initialize()
                    
                    self.providers[provider_type] = ProviderInfo(
                        provider=provider,
                        status=ProviderStatus.ACTIVE
                    )
                    
                    logger.info("Provider initialized", provider=provider_type)
                    
                except Exception as e:
                    logger.error("Failed to initialize provider", 
                               provider=provider_type, error=str(e))
                    self.providers[provider_type] = ProviderInfo(
                        provider=None,  # Will be set later
                        status=ProviderStatus.UNAVAILABLE
                    )
            
            # Validate that we have at least one working provider
            active_providers = [p for p in self.providers.values() 
                              if p.status == ProviderStatus.ACTIVE]
            
            if not active_providers:
                raise ValueError("No providers could be initialized")
            
            self._initialized = True
            logger.info("LLM Manager initialized", 
                       active_providers=len(active_providers))
            
        except Exception as e:
            logger.error("Failed to initialize LLM Manager", error=str(e))
            raise
    
    async def _create_provider(self, provider_type: ProviderType, api_key: str) -> BaseLLMProvider:
        """
        Create a provider instance
        
        Args:
            provider_type: Type of provider to create
            api_key: API key for the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == ProviderType.OPENAI:
            return OpenAIProvider(api_key)
        elif provider_type == ProviderType.CLAUDE:
            return ClaudeProvider(api_key)
        elif provider_type == ProviderType.GEMINI:
            return GeminiProvider(api_key)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    async def generate_completion(self, request: CompletionRequest, 
                                preferred_provider: Optional[ProviderType] = None,
                                use_cache: bool = True) -> CompletionResponse:
        """
        Generate completion using the best available provider
        
        Args:
            request: Completion request
            preferred_provider: Preferred provider (if available)
            use_cache: Whether to use cached responses
            
        Returns:
            Completion response
            
        Raises:
            ProviderError: If all providers fail
        """
        if not self._initialized:
            raise RuntimeError("LLM Manager not initialized")
        
        # Check cache first if enabled
        if use_cache and self.cache_manager:
            cached_response = await self.cache_manager.get_cached_response(request)
            if cached_response:
                logger.debug("Using cached response", 
                           provider=cached_response.provider,
                           model=cached_response.model)
                return cached_response
        
        # Select provider
        provider_info = await self._select_provider(request, preferred_provider)
        
        if not provider_info:
            raise ProviderError("No available providers")
        
        # Generate completion with retries
        response = await self._generate_with_retries(provider_info, request)
        
        # Cache response if enabled
        if use_cache and self.cache_manager:
            await self.cache_manager.cache_response(request, response)
        
        # Update provider metrics
        await self._update_provider_metrics(provider_info, response)
        
        return response
    
    async def _select_provider(self, request: CompletionRequest, 
                             preferred_provider: Optional[ProviderType] = None) -> Optional[ProviderInfo]:
        """
        Select the best provider for the request
        
        Args:
            request: Completion request
            preferred_provider: Preferred provider
            
        Returns:
            Selected provider info or None if none available
        """
        # Filter available providers
        available_providers = [
            info for info in self.providers.values()
            if info.status == ProviderStatus.ACTIVE and info.provider is not None
        ]
        
        if not available_providers:
            return None
        
        # Check if preferred provider is available
        if preferred_provider and preferred_provider in self.providers:
            preferred_info = self.providers[preferred_provider]
            if preferred_info.status == ProviderStatus.ACTIVE:
                return preferred_info
        
        # Apply load balancing strategy
        if self.config.load_balancing_strategy == LoadBalancingStrategy.PRIORITY:
            return min(available_providers, key=lambda p: p.priority)
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_COST:
            # Estimate cost for each provider
            costs = []
            for provider_info in available_providers:
                try:
                    cost = await provider_info.provider.estimate_cost(request)
                    costs.append((cost, provider_info))
                except Exception:
                    costs.append((float('inf'), provider_info))
            
            return min(costs, key=lambda x: x[0])[1]
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.FASTEST:
            return min(available_providers, key=lambda p: p.avg_response_time)
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin based on last used time
            return min(available_providers, 
                      key=lambda p: p.last_used or 0)
        
        else:  # RANDOM
            import random
            return random.choice(available_providers)
    
    async def _generate_with_retries(self, provider_info: ProviderInfo, 
                                   request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion with retry logic
        
        Args:
            provider_info: Provider to use
            request: Completion request
            
        Returns:
            Completion response
            
        Raises:
            ProviderError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Check if provider is still available
                if provider_info.status != ProviderStatus.ACTIVE:
                    # Try to find an alternative provider
                    alternative = await self._select_provider(request)
                    if alternative and alternative != provider_info:
                        provider_info = alternative
                    else:
                        raise ProviderError("No available providers")
                
                # Make the request with timeout
                response = await asyncio.wait_for(
                    provider_info.provider.generate_completion(request),
                    timeout=self.config.response_timeout
                )
                
                # Reset error count on success
                provider_info.error_count = 0
                provider_info.success_count += 1
                provider_info.last_used = time.time()
                
                return response
                
            except (RateLimitError, QuotaExceededError) as e:
                # Update provider status
                if isinstance(e, RateLimitError):
                    provider_info.status = ProviderStatus.RATE_LIMITED
                else:
                    provider_info.status = ProviderStatus.QUOTA_EXCEEDED
                
                provider_info.error_count += 1
                last_error = e
                
                logger.warning("Provider error", 
                             provider=provider_info.provider.provider_type,
                             error=str(e),
                             attempt=attempt + 1)
                
                # Try alternative provider
                alternative = await self._select_provider(request)
                if alternative and alternative != provider_info:
                    provider_info = alternative
                    continue
                
                # Wait before retry
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                
            except Exception as e:
                provider_info.error_count += 1
                last_error = e
                
                logger.error("Provider request failed", 
                           provider=provider_info.provider.provider_type,
                           error=str(e),
                           attempt=attempt + 1)
                
                # Wait before retry
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # All retries failed
        raise ProviderError(
            f"All retry attempts failed. Last error: {str(last_error)}",
            provider=provider_info.provider,
            original_error=last_error
        )
    
    async def _update_provider_metrics(self, provider_info: ProviderInfo, 
                                     response: CompletionResponse) -> None:
        """
        Update provider performance metrics
        
        Args:
            provider_info: Provider info to update
            response: Completion response
        """
        # Update response time
        if provider_info.avg_response_time == 0:
            provider_info.avg_response_time = response.response_time
        else:
            # Exponential moving average
            alpha = 0.1
            provider_info.avg_response_time = (
                alpha * response.response_time + 
                (1 - alpha) * provider_info.avg_response_time
            )
        
        # Update cost
        if response.usage and "total_tokens" in response.usage:
            model_info = await provider_info.provider.get_model_info(response.model)
            if model_info:
                cost = (response.usage["total_tokens"] / 1000) * model_info.cost_per_1k_tokens
                provider_info.total_cost += cost
    
    async def get_available_models(self, provider_type: Optional[ProviderType] = None) -> Dict[str, List[str]]:
        """
        Get available models from providers
        
        Args:
            provider_type: Optional specific provider type
            
        Returns:
            Dictionary mapping provider types to list of model names
        """
        models = {}
        
        providers_to_check = [provider_type] if provider_type else list(self.providers.keys())
        
        for ptype in providers_to_check:
            if ptype in self.providers and self.providers[ptype].status == ProviderStatus.ACTIVE:
                try:
                    provider_models = await self.providers[ptype].provider.get_available_models()
                    models[ptype.value] = [model.name for model in provider_models]
                except Exception as e:
                    logger.warning("Failed to get models from provider", 
                                 provider=ptype, error=str(e))
                    models[ptype.value] = []
        
        return models
    
    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all providers
        
        Returns:
            Dictionary with provider status information
        """
        status = {}
        
        for provider_type, provider_info in self.providers.items():
            status[provider_type.value] = {
                "status": provider_info.status.value,
                "error_count": provider_info.error_count,
                "success_count": provider_info.success_count,
                "total_cost": provider_info.total_cost,
                "avg_response_time": provider_info.avg_response_time,
                "priority": provider_info.priority,
                "last_used": provider_info.last_used
            }
        
        return status
    
    async def reset_provider_status(self, provider_type: ProviderType) -> None:
        """
        Reset provider status to active
        
        Args:
            provider_type: Provider type to reset
        """
        if provider_type in self.providers:
            self.providers[provider_type].status = ProviderStatus.ACTIVE
            self.providers[provider_type].error_count = 0
            logger.info("Provider status reset", provider=provider_type)
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics if caching is enabled
        
        Returns:
            Cache statistics or None if caching is disabled
        """
        if self.cache_manager:
            return await self.cache_manager.get_cache_stats()
        return None
    
    async def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache if caching is enabled
        
        Args:
            pattern: Optional pattern to match keys
            
        Returns:
            Number of keys deleted
        """
        if self.cache_manager:
            return await self.cache_manager.clear_cache(pattern)
        return 0
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        if self.cache_manager:
            await self.cache_manager.disconnect()
        
        # Close provider connections if they have any
        for provider_info in self.providers.values():
            if provider_info.provider and hasattr(provider_info.provider, 'close'):
                try:
                    await provider_info.provider.close()
                except Exception as e:
                    logger.warning("Error closing provider", error=str(e))
        
        logger.info("LLM Manager closed")
