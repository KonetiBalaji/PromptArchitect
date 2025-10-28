"""
Redis-based Cache Manager for Prompt Architect
Author: Balaji Koneti

Provides caching functionality for prompts, responses, and template compilations
to improve performance and reduce costs.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from pydantic import BaseModel, Field
import structlog

from llm_providers.base import CompletionRequest, CompletionResponse, ProviderType


logger = structlog.get_logger(__name__)


class CacheConfig(BaseModel):
    """Configuration for cache settings"""
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds (1 hour)")
    prompt_ttl: int = Field(default=7200, description="TTL for generated prompts (2 hours)")
    response_ttl: int = Field(default=1800, description="TTL for LLM responses (30 minutes)")
    template_ttl: int = Field(default=86400, description="TTL for template compilations (24 hours)")
    max_cache_size: int = Field(default=10000, description="Maximum number of cached items")
    enable_compression: bool = Field(default=True, description="Enable compression for large items")
    cache_prefix: str = Field(default="prompt_architect", description="Prefix for cache keys")


class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    total_requests: int = Field(default=0, description="Total cache requests")
    hit_rate: float = Field(default=0.0, description="Cache hit rate percentage")
    total_savings: float = Field(default=0.0, description="Total cost savings in USD")
    last_reset: datetime = Field(default_factory=datetime.now, description="Last metrics reset time")


class CacheItem(BaseModel):
    """Cached item with metadata"""
    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime = Field(default_factory=datetime.now, description="Last access timestamp")
    size_bytes: int = Field(default=0, description="Size of cached item in bytes")
    cost_saved: float = Field(default=0.0, description="Estimated cost saved by caching")


class CacheManager:
    """
    Redis-based cache manager for Prompt Architect
    
    Provides intelligent caching for:
    - Generated prompts (by input hash)
    - LLM responses (with TTL)
    - Template compilations
    - Provider model information
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager
        
        Args:
            config: Cache configuration settings
        """
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self._connected = False
    
    async def connect(self) -> None:
        """
        Connect to Redis server
        
        Raises:
            ConnectionError: If unable to connect to Redis
        """
        try:
            # Create Redis connection
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            
            logger.info("Connected to Redis cache", url=self.config.redis_url)
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Disconnected from Redis cache")
    
    def _generate_key(self, prefix: str, *args: str) -> str:
        """
        Generate a cache key from prefix and arguments
        
        Args:
            prefix: Key prefix
            *args: Arguments to include in the key
            
        Returns:
            Generated cache key
        """
        # Create a hash of the arguments for consistent key generation
        key_data = "_".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{self.config.cache_prefix}:{prefix}:{key_hash}"
    
    def _serialize_value(self, value: Any) -> str:
        """
        Serialize value for storage in Redis
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized JSON string
        """
        if isinstance(value, (BaseModel, dict)):
            return json.dumps(value, default=str)
        else:
            return json.dumps(value, default=str)
    
    def _deserialize_value(self, value: str, target_type: type = None) -> Any:
        """
        Deserialize value from Redis storage
        
        Args:
            value: Serialized JSON string
            target_type: Optional target type for deserialization
            
        Returns:
            Deserialized value
        """
        try:
            data = json.loads(value)
            if target_type and hasattr(target_type, 'model_validate'):
                return target_type.model_validate(data)
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to deserialize cached value", error=str(e))
            return None
    
    async def get(self, key: str, target_type: type = None) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            target_type: Optional target type for deserialization
            
        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Get value from Redis
            value = await self.redis_client.get(key)
            
            if value is None:
                self.metrics.misses += 1
                self.metrics.total_requests += 1
                logger.debug("Cache miss", key=key)
                return None
            
            # Deserialize and return value
            deserialized_value = self._deserialize_value(value, target_type)
            
            # Update access metrics
            self.metrics.hits += 1
            self.metrics.total_requests += 1
            self.metrics.hit_rate = (self.metrics.hits / self.metrics.total_requests) * 100
            
            # Update access count in Redis
            await self.redis_client.hincrby(f"{key}:meta", "access_count", 1)
            await self.redis_client.hset(f"{key}:meta", "last_accessed", time.time())
            
            logger.debug("Cache hit", key=key)
            return deserialized_value
            
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            self.metrics.misses += 1
            self.metrics.total_requests += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Calculate TTL
            cache_ttl = ttl or self.config.default_ttl
            
            # Store value in Redis
            await self.redis_client.setex(key, cache_ttl, serialized_value)
            
            # Store metadata
            metadata = {
                "created_at": time.time(),
                "expires_at": time.time() + cache_ttl,
                "access_count": 0,
                "last_accessed": time.time(),
                "size_bytes": len(serialized_value.encode('utf-8')),
                "cost_saved": 0.0
            }
            await self.redis_client.hset(f"{key}:meta", mapping=metadata)
            await self.redis_client.expire(f"{key}:meta", cache_ttl)
            
            logger.debug("Cache set", key=key, ttl=cache_ttl)
            return True
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Delete value and metadata
            await self.redis_client.delete(key, f"{key}:meta")
            logger.debug("Cache delete", key=key)
            return True
            
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        try:
            exists = await self.redis_client.exists(key)
            return bool(exists)
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False
    
    async def get_prompt_cache_key(self, user_input: str, template_id: str, 
                                 provider: ProviderType, model: str) -> str:
        """
        Generate cache key for generated prompt
        
        Args:
            user_input: User input text
            template_id: Template identifier
            provider: LLM provider
            model: Model name
            
        Returns:
            Cache key for the prompt
        """
        return self._generate_key("prompt", user_input, template_id, provider.value, model)
    
    async def get_response_cache_key(self, request: CompletionRequest) -> str:
        """
        Generate cache key for LLM response
        
        Args:
            request: Completion request
            
        Returns:
            Cache key for the response
        """
        # Create a hash of the request for consistent key generation
        request_data = f"{request.prompt}_{request.model}_{request.temperature}_{request.max_tokens}"
        return self._generate_key("response", request_data)
    
    async def cache_prompt(self, user_input: str, template_id: str, 
                          provider: ProviderType, model: str, 
                          generated_prompt: str) -> bool:
        """
        Cache a generated prompt
        
        Args:
            user_input: Original user input
            template_id: Template identifier
            provider: LLM provider
            model: Model name
            generated_prompt: Generated prompt to cache
            
        Returns:
            True if cached successfully
        """
        key = await self.get_prompt_cache_key(user_input, template_id, provider, model)
        return await self.set(key, generated_prompt, self.config.prompt_ttl)
    
    async def get_cached_prompt(self, user_input: str, template_id: str, 
                               provider: ProviderType, model: str) -> Optional[str]:
        """
        Get cached prompt
        
        Args:
            user_input: Original user input
            template_id: Template identifier
            provider: LLM provider
            model: Model name
            
        Returns:
            Cached prompt or None if not found
        """
        key = await self.get_prompt_cache_key(user_input, template_id, provider, model)
        return await self.get(key)
    
    async def cache_response(self, request: CompletionRequest, 
                           response: CompletionResponse) -> bool:
        """
        Cache an LLM response
        
        Args:
            request: Original completion request
            response: Completion response to cache
            
        Returns:
            True if cached successfully
        """
        key = await self.get_response_cache_key(request)
        return await self.set(key, response.model_dump(), self.config.response_ttl)
    
    async def get_cached_response(self, request: CompletionRequest) -> Optional[CompletionResponse]:
        """
        Get cached LLM response
        
        Args:
            request: Completion request
            
        Returns:
            Cached response or None if not found
        """
        key = await self.get_response_cache_key(request)
        response_data = await self.get(key)
        
        if response_data:
            try:
                return CompletionResponse.model_validate(response_data)
            except Exception as e:
                logger.warning("Failed to deserialize cached response", error=str(e))
                return None
        
        return None
    
    async def get_metrics(self) -> CacheMetrics:
        """
        Get cache performance metrics
        
        Returns:
            Current cache metrics
        """
        # Update hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = (self.metrics.hits / self.metrics.total_requests) * 100
        
        return self.metrics
    
    async def reset_metrics(self) -> None:
        """Reset cache metrics"""
        self.metrics = CacheMetrics()
        logger.info("Cache metrics reset")
    
    async def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Optional pattern to match keys (default: all cache keys)
            
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            await self.connect()
        
        try:
            if pattern is None:
                pattern = f"{self.config.cache_prefix}:*"
            
            # Find all matching keys
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete all matching keys
                deleted_count = await self.redis_client.delete(*keys)
                logger.info("Cache cleared", pattern=pattern, deleted_count=deleted_count)
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error("Cache clear error", pattern=pattern, error=str(e))
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Get Redis info
            info = await self.redis_client.info()
            
            # Get cache-specific stats
            cache_keys = await self.redis_client.keys(f"{self.config.cache_prefix}:*")
            
            stats = {
                "redis_info": {
                    "used_memory": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "cache_info": {
                    "total_keys": len(cache_keys),
                    "prompt_keys": len([k for k in cache_keys if ":prompt:" in k]),
                    "response_keys": len([k for k in cache_keys if ":response:" in k]),
                    "template_keys": len([k for k in cache_keys if ":template:" in k])
                },
                "metrics": await self.get_metrics()
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}
