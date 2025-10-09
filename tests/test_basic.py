#!/usr/bin/env python3
"""
Basic Tests for Prompt Architect
Author: Balaji Koneti

Basic unit tests for the core functionality.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers.base import ProviderType, CompletionRequest, CompletionResponse
from cache.cache_manager import CacheConfig, CacheManager
from prompt_builder import PromptBuilder, PromptRequest, PromptTemplate


class TestPromptTemplates:
    """Test prompt template functionality"""
    
    def test_template_creation(self):
        """Test creating a prompt template"""
        template = PromptTemplate(
            id="test_template",
            name="Test Template",
            description="A test template",
            role_template="You are a {role}.",
            task_template="Complete this {task}.",
            context_template="Context: {context}",
            reasoning_template="Reason: {reasoning}",
            output_format_template="Format: {format}",
            stop_condition_template="Stop when {condition}",
            variables=["role", "task", "context", "reasoning", "format", "condition"]
        )
        
        assert template.id == "test_template"
        assert template.name == "Test Template"
        assert len(template.variables) == 6
    
    def test_template_filling(self):
        """Test filling template with variables"""
        template = PromptTemplate(
            id="test_template",
            name="Test Template",
            description="A test template",
            role_template="You are a {role}.",
            task_template="Complete this {task}.",
            context_template="Context: {context}",
            reasoning_template="Reason: {reasoning}",
            output_format_template="Format: {format}",
            stop_condition_template="Stop when {condition}",
            variables=["role", "task", "context", "reasoning", "format", "condition"]
        )
        
        variables = {
            "role": "developer",
            "task": "code review",
            "context": "Python project",
            "reasoning": "best practices",
            "format": "markdown",
            "condition": "done"
        }
        
        # Test template filling
        filled_role = template.role_template.format(**variables)
        assert filled_role == "You are a developer."
        
        filled_task = template.task_template.format(**variables)
        assert filled_task == "Complete this code review."


class TestPromptRequest:
    """Test prompt request functionality"""
    
    def test_prompt_request_creation(self):
        """Test creating a prompt request"""
        request = PromptRequest(
            user_input="Test input",
            template_id="test_template",
            variables={"key": "value"}
        )
        
        assert request.user_input == "Test input"
        assert request.template_id == "test_template"
        assert request.variables["key"] == "value"
        assert request.use_cache is True  # Default value
    
    def test_prompt_request_defaults(self):
        """Test prompt request default values"""
        request = PromptRequest(user_input="Test input")
        
        assert request.template_id is None
        assert request.role is None
        assert request.task is None
        assert request.context is None
        assert request.reasoning is None
        assert request.output_format is None
        assert request.stop_condition is None
        assert request.variables == {}
        assert request.provider is None
        assert request.model is None
        assert request.use_cache is True


class TestCompletionRequest:
    """Test completion request functionality"""
    
    def test_completion_request_creation(self):
        """Test creating a completion request"""
        request = CompletionRequest(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "gpt-4o"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.top_p == 1.0  # Default value
        assert request.frequency_penalty == 0.0  # Default value
        assert request.presence_penalty == 0.0  # Default value
        assert request.stop is None  # Default value
        assert request.stream is False  # Default value
        assert request.system_message is None  # Default value
    
    def test_completion_request_validation(self):
        """Test completion request validation"""
        # Test temperature validation
        with pytest.raises(ValueError):
            CompletionRequest(
                prompt="Test",
                model="gpt-4o",
                temperature=3.0  # Invalid: > 2.0
            )
        
        # Test top_p validation
        with pytest.raises(ValueError):
            CompletionRequest(
                prompt="Test",
                model="gpt-4o",
                top_p=1.5  # Invalid: > 1.0
            )


class TestCacheConfig:
    """Test cache configuration"""
    
    def test_cache_config_defaults(self):
        """Test cache configuration default values"""
        config = CacheConfig()
        
        assert config.redis_url == "redis://localhost:6379"
        assert config.default_ttl == 3600
        assert config.prompt_ttl == 7200
        assert config.response_ttl == 1800
        assert config.template_ttl == 86400
        assert config.max_cache_size == 10000
        assert config.enable_compression is True
        assert config.cache_prefix == "prompt_architect"
    
    def test_cache_config_custom(self):
        """Test custom cache configuration"""
        config = CacheConfig(
            redis_url="redis://custom:6379",
            default_ttl=1800,
            cache_prefix="custom_prefix"
        )
        
        assert config.redis_url == "redis://custom:6379"
        assert config.default_ttl == 1800
        assert config.cache_prefix == "custom_prefix"


@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality"""
    
    async def test_cache_manager_initialization(self):
        """Test cache manager initialization (without Redis)"""
        config = CacheConfig()
        cache_manager = CacheManager(config)
        
        # Test that cache manager can be created
        assert cache_manager.config == config
        assert cache_manager.redis_client is None
        assert cache_manager._connected is False
    
    async def test_prompt_builder_initialization(self):
        """Test prompt builder initialization"""
        # Mock LLM manager
        class MockLLMManager:
            pass
        
        prompt_builder = PromptBuilder(MockLLMManager(), None)
        
        # Test that prompt builder can be created
        assert prompt_builder.llm_manager is not None
        assert prompt_builder.cache_manager is None
        assert len(prompt_builder.templates) > 0  # Should have default templates
    
    def test_provider_type_enum(self):
        """Test provider type enum"""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.CLAUDE == "claude"
        assert ProviderType.GEMINI == "gemini"
        
        # Test enum values
        providers = list(ProviderType)
        assert len(providers) == 3
        assert ProviderType.OPENAI in providers
        assert ProviderType.CLAUDE in providers
        assert ProviderType.GEMINI in providers


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_completion_response_creation(self):
        """Test creating a completion response"""
        response = CompletionResponse(
            content="Test response",
            model="gpt-4o",
            provider=ProviderType.OPENAI,
            usage={"total_tokens": 100},
            finish_reason="stop",
            response_time=1.5
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.provider == ProviderType.OPENAI
        assert response.usage["total_tokens"] == 100
        assert response.finish_reason == "stop"
        assert response.response_time == 1.5
        assert response.metadata == {}  # Default value
    
    def test_completion_response_defaults(self):
        """Test completion response default values"""
        response = CompletionResponse(
            content="Test",
            model="gpt-4o",
            provider=ProviderType.OPENAI,
            usage={"total_tokens": 50},
            finish_reason="stop",
            response_time=1.0
        )
        
        assert response.metadata == {}
        assert response.timestamp is not None  # Should be set automatically


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
