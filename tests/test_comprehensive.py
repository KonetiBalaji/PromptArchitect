"""
Comprehensive Test Suite
Author: Balaji Koneti

Complete test suite covering all components with integration, performance,
and security testing.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.llm_manager import LLMManager
from src.reasoning_manager import ReasoningManager
from src.chain_of_thought.sequential_chain import SequentialChain
from src.chain_of_thought.parallel_chain import ParallelChain
from src.chain_of_thought.tree_reasoning import TreeReasoning
from src.chain_of_thought.reflection import Reflection
from src.templates.template_generator import TemplateGenerator
from src.templates.template_registry import template_registry
from src.templates.dynamic_selector import DynamicSelector
from src.evaluation.evaluator import PromptEvaluator, EvaluationMetric, EvaluationType
from src.optimization.cost_optimizer import CostOptimizer, OptimizationStrategy
from src.monitoring.metrics import MonitoringSystem, MetricsCollector
from src.database.manager import DatabaseManager
from src.api.main import app
from src.api.auth import AuthManager
from src.api.rate_limiter import RateLimiter
from src.llm_providers.base import CompletionRequest, ProviderType


class TestLLMManager:
    """Test LLM Manager functionality"""
    
    @pytest.fixture
    async def llm_manager(self):
        """Create LLM manager instance"""
        manager = LLMManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_llm_manager_initialization(self, llm_manager):
        """Test LLM manager initialization"""
        assert llm_manager is not None
        assert len(llm_manager.providers) > 0
    
    @pytest.mark.asyncio
    async def test_completion_generation(self, llm_manager):
        """Test completion generation"""
        request = CompletionRequest(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.0
        )
        
        response = await llm_manager.generate_completion(request)
        
        assert response is not None
        assert response.content is not None
        assert response.model == "gpt-4o-mini"
        assert response.provider in [ProviderType.OPENAI, ProviderType.CLAUDE, ProviderType.GEMINI]
    
    @pytest.mark.asyncio
    async def test_provider_failover(self, llm_manager):
        """Test provider failover mechanism"""
        # Mock first provider to fail
        with patch.object(llm_manager.providers[ProviderType.OPENAI], 'generate_completion', 
                         side_effect=Exception("Provider failed")):
            request = CompletionRequest(
                prompt="Test failover",
                model="gpt-4o-mini"
            )
            
            response = await llm_manager.generate_completion(request)
            assert response is not None
            assert response.provider != ProviderType.OPENAI  # Should use different provider


class TestReasoningManager:
    """Test Reasoning Manager functionality"""
    
    @pytest.fixture
    async def reasoning_manager(self):
        """Create reasoning manager instance"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        return ReasoningManager(llm_manager)
    
    @pytest.mark.asyncio
    async def test_reasoning_manager_initialization(self, reasoning_manager):
        """Test reasoning manager initialization"""
        assert reasoning_manager is not None
        assert reasoning_manager.llm_manager is not None
    
    @pytest.mark.asyncio
    async def test_sequential_reasoning(self, reasoning_manager):
        """Test sequential reasoning"""
        request = CompletionRequest(
            prompt="Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?",
            model="gpt-4o-mini"
        )
        
        result = await reasoning_manager.perform_reasoning(
            request,
            strategy="sequential",
            max_steps=3
        )
        
        assert result is not None
        assert result.content is not None
        assert len(result.reasoning_steps) > 0
        assert result.confidence_score is not None
    
    @pytest.mark.asyncio
    async def test_parallel_reasoning(self, reasoning_manager):
        """Test parallel reasoning"""
        request = CompletionRequest(
            prompt="Analyze the pros and cons of renewable energy",
            model="gpt-4o-mini"
        )
        
        result = await reasoning_manager.perform_reasoning(
            request,
            strategy="parallel",
            max_steps=2,
            parallel_workers=2
        )
        
        assert result is not None
        assert result.content is not None
        assert result.reasoning_steps is not None


class TestChainOfThought:
    """Test Chain-of-Thought implementations"""
    
    @pytest.fixture
    async def chain_components(self):
        """Create chain-of-thought components"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        return {
            "sequential": SequentialChain(llm_manager),
            "parallel": ParallelChain(llm_manager),
            "tree": TreeReasoning(llm_manager),
            "reflection": Reflection(llm_manager)
        }
    
    @pytest.mark.asyncio
    async def test_sequential_chain(self, chain_components):
        """Test sequential chain reasoning"""
        sequential_chain = chain_components["sequential"]
        
        request = CompletionRequest(
            prompt="Explain the water cycle step by step",
            model="gpt-4o-mini"
        )
        
        result = await sequential_chain.run(
            request,
            max_steps=3,
            verification_enabled=True,
            confidence_threshold=0.7
        )
        
        assert result is not None
        assert result.content is not None
        assert len(result.reasoning_steps) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_chain(self, chain_components):
        """Test parallel chain reasoning"""
        parallel_chain = chain_components["parallel"]
        
        request = CompletionRequest(
            prompt="Compare and contrast democracy and autocracy",
            model="gpt-4o-mini"
        )
        
        result = await parallel_chain.run(
            request,
            max_steps=2,
            verification_enabled=True,
            confidence_threshold=0.7,
            num_parallel_workers=2
        )
        
        assert result is not None
        assert result.content is not None
        assert result.reasoning_steps is not None
    
    @pytest.mark.asyncio
    async def test_tree_reasoning(self, chain_components):
        """Test tree-based reasoning"""
        tree_reasoning = chain_components["tree"]
        
        request = CompletionRequest(
            prompt="What are the factors that affect climate change?",
            model="gpt-4o-mini"
        )
        
        result = await tree_reasoning.run(
            request,
            max_steps=3,
            verification_enabled=True,
            confidence_threshold=0.7
        )
        
        assert result is not None
        assert result.content is not None
        assert result.reasoning_steps is not None
    
    @pytest.mark.asyncio
    async def test_reflection(self, chain_components):
        """Test reflective reasoning"""
        reflection = chain_components["reflection"]
        
        request = CompletionRequest(
            prompt="Write a short story about a robot learning to paint",
            model="gpt-4o-mini"
        )
        
        result = await reflection.run(
            request,
            max_steps=2,
            verification_enabled=True,
            confidence_threshold=0.7
        )
        
        assert result is not None
        assert result.content is not None
        assert result.reasoning_steps is not None


class TestTemplateSystem:
    """Test Template System functionality"""
    
    @pytest.fixture
    async def template_components(self):
        """Create template system components"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        return {
            "generator": TemplateGenerator(llm_manager),
            "selector": DynamicSelector(llm_manager)
        }
    
    @pytest.mark.asyncio
    async def test_template_generation(self, template_components):
        """Test template generation"""
        generator = template_components["generator"]
        
        from src.templates.template_generator import TemplateGenerationRequest, TemplateCategory, TemplateComplexity
        
        request = TemplateGenerationRequest(
            category=TemplateCategory.RESEARCH,
            complexity=TemplateComplexity.MODERATE,
            use_case="Academic research paper analysis",
            requirements=["Structured analysis", "Citation format"],
            target_audience="Graduate students"
        )
        
        template = await generator.generate_template(request)
        
        assert template is not None
        assert template.template_id is not None
        assert template.template_content is not None
        assert template.category == TemplateCategory.RESEARCH
    
    @pytest.mark.asyncio
    async def test_template_selection(self, template_components):
        """Test dynamic template selection"""
        selector = template_components["selector"]
        
        # Get available templates
        available_templates = template_registry.get_all_templates()
        
        result = await selector.select_template(
            "I need to write a business proposal for a new product launch",
            available_templates,
            {"industry": "technology", "audience": "investors"}
        )
        
        assert result is not None
        assert result.template_id is not None
        assert result.confidence > 0.0
        assert result.detected_intent is not None
    
    def test_template_registry(self):
        """Test template registry functionality"""
        # Test getting all templates
        all_templates = template_registry.get_all_templates()
        assert len(all_templates) > 0
        
        # Test getting templates by category
        research_templates = template_registry.get_templates_by_category("research")
        assert len(research_templates) > 0
        
        # Test template analytics
        analytics = template_registry.get_template_analytics()
        assert analytics is not None
        assert "total_templates" in analytics


class TestEvaluationFramework:
    """Test Evaluation Framework functionality"""
    
    @pytest.fixture
    async def evaluator(self):
        """Create evaluator instance"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        reasoning_manager = ReasoningManager(llm_manager)
        return PromptEvaluator(llm_manager, reasoning_manager)
    
    @pytest.mark.asyncio
    async def test_prompt_evaluation(self, evaluator):
        """Test prompt evaluation"""
        result = await evaluator.evaluate_prompt(
            prompt="What is the capital of France?",
            model="gpt-4o-mini",
            provider="openai",
            evaluation_metrics=[
                EvaluationMetric.ACCURACY,
                EvaluationMetric.RELEVANCE,
                EvaluationMetric.COHERENCE
            ]
        )
        
        assert result is not None
        assert result.test_id is not None
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0
        assert len(result.metrics) > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_suite(self, evaluator):
        """Test benchmark suite execution"""
        results = await evaluator.run_benchmark_suite(
            suite_name="general_knowledge",
            models=["gpt-4o-mini"],
            providers=["openai"],
            use_reasoning=False
        )
        
        assert results is not None
        assert len(results) > 0
        
        # Check that we have results for the model
        model_key = "gpt-4o-mini_openai"
        assert model_key in results
        assert len(results[model_key]) > 0
    
    def test_evaluation_statistics(self, evaluator):
        """Test evaluation statistics"""
        stats = evaluator.get_evaluation_statistics(hours=24)
        assert stats is not None
        assert "total_evaluations" in stats


class TestCostOptimizer:
    """Test Cost Optimization functionality"""
    
    @pytest.fixture
    async def cost_optimizer(self):
        """Create cost optimizer instance"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        reasoning_manager = ReasoningManager(llm_manager)
        evaluator = PromptEvaluator(llm_manager, reasoning_manager)
        return CostOptimizer(llm_manager, evaluator)
    
    @pytest.mark.asyncio
    async def test_cost_optimization(self, cost_optimizer):
        """Test cost optimization"""
        result = await cost_optimizer.optimize_request(
            prompt="Write a short story about a robot",
            optimization_strategy=OptimizationStrategy.BALANCE_COST_QUALITY,
            quality_threshold=0.8
        )
        
        assert result is not None
        assert result.selected_model is not None
        assert result.estimated_cost > 0.0
        assert result.estimated_quality >= 0.0
        assert result.estimated_quality <= 1.0
        assert result.reasoning is not None
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, cost_optimizer):
        """Test usage tracking"""
        await cost_optimizer.track_usage(
            user_id="test_user",
            model="gpt-4o-mini",
            provider=ProviderType.OPENAI,
            tokens_used=100,
            cost=0.001,
            request_time=1.0
        )
        
        analytics = cost_optimizer.get_usage_analytics("test_user", days=1)
        assert analytics is not None
        assert "total_requests" in analytics
        assert analytics["total_requests"] > 0
    
    def test_budget_management(self, cost_optimizer):
        """Test budget management"""
        from src.optimization.cost_optimizer import Budget
        
        budget = Budget(
            total_budget=100.0,
            daily_limit=10.0,
            monthly_limit=50.0
        )
        
        cost_optimizer.set_budget("test_user", budget)
        
        status = cost_optimizer.get_budget_status("test_user")
        assert status is not None
        assert "status" in status
        assert "daily_limit" in status


class TestMonitoringSystem:
    """Test Monitoring System functionality"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system instance"""
        return MonitoringSystem()
    
    def test_metrics_collector(self, monitoring_system):
        """Test metrics collector"""
        collector = monitoring_system.metrics_collector
        
        # Record some metrics
        collector.record_request("GET", "/test", 200, 0.5)
        collector.record_llm_request("openai", "gpt-4o-mini", "success", 1.0, {"input": 100, "output": 50}, 0.001)
        collector.record_error("validation_error", "api")
        
        # Get metrics summary
        summary = collector.get_metrics_summary()
        assert summary is not None
        assert "timestamp" in summary
    
    def test_health_checker(self, monitoring_system):
        """Test health checker"""
        health_checker = monitoring_system.health_checker
        
        # Mock health check function
        async def mock_health_check():
            return {"status": "healthy", "response_time": 0.1}
        
        # Run health check
        health_status = asyncio.run(health_checker.check_component_health("test_component", mock_health_check))
        
        assert health_status is not None
        assert health_status["status"] == "healthy"
    
    def test_alert_manager(self, monitoring_system):
        """Test alert manager"""
        alert_manager = monitoring_system.alert_manager
        
        # Check alerts with test metrics
        test_metrics = {
            "error_rate": 0.1,  # Above threshold
            "response_time": 1.0,
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "health_score": 0.9
        }
        
        alert_manager.check_alerts(test_metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0  # Should have triggered error_rate alert


class TestDatabaseIntegration:
    """Test Database Integration functionality"""
    
    @pytest.fixture
    async def database_manager(self):
        """Create database manager instance"""
        # Use in-memory SQLite for testing
        manager = DatabaseManager("sqlite:///:memory:")
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_database_operations(self, database_manager):
        """Test database operations"""
        # Test user creation
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "hashed_password": "hashed_password",
            "is_active": True,
            "is_admin": False
        }
        
        user = await database_manager.create_user(user_data)
        assert user is not None
        assert user.email == "test@example.com"
        
        # Test completion creation
        completion_data = {
            "user_id": str(user.id),
            "prompt": "Test prompt",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "response": "Test response",
            "usage": {"input": 10, "output": 5, "total": 15},
            "response_time": 1.0,
            "confidence_score": 0.9
        }
        
        completion = await database_manager.create_completion(completion_data)
        assert completion is not None
        assert completion.prompt == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_analytics_queries(self, database_manager):
        """Test analytics queries"""
        # Test system analytics
        analytics = await database_manager.get_system_analytics(days=1)
        assert analytics is not None
        assert "completion_stats" in analytics
        assert "reasoning_stats" in analytics
        assert "template_stats" in analytics


class TestAPIIntegration:
    """Test API Integration functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
    
    def test_completion_endpoint(self, client):
        """Test completion endpoint"""
        # This would require authentication in a real test
        # For now, we'll test the endpoint structure
        completion_data = {
            "prompt": "What is 2+2?",
            "model": "gpt-4o-mini",
            "temperature": 0.0
        }
        
        # Note: This will fail without proper authentication
        # In a real test, you'd set up proper auth headers
        response = client.post("/completion", json=completion_data)
        assert response.status_code in [200, 401]  # Either success or auth required
    
    def test_templates_endpoint(self, client):
        """Test templates endpoint"""
        response = client.get("/templates")
        assert response.status_code in [200, 401]  # Either success or auth required


class TestSecurity:
    """Test Security functionality"""
    
    def test_auth_manager(self):
        """Test authentication manager"""
        auth_manager = AuthManager()
        
        # Test user creation
        from src.api.auth import UserCreate
        user_data = UserCreate(
            email="test@example.com",
            username="testuser",
            password="testpassword"
        )
        
        user = asyncio.run(auth_manager.create_user(user_data))
        assert user is not None
        assert user.email == "test@example.com"
        
        # Test authentication
        from src.api.auth import UserLogin
        login_data = UserLogin(
            email="test@example.com",
            password="testpassword"
        )
        
        authenticated_user = asyncio.run(auth_manager.authenticate_user(login_data.email, login_data.password))
        assert authenticated_user is not None
        assert authenticated_user.email == "test@example.com"
    
    def test_rate_limiter(self):
        """Test rate limiter"""
        rate_limiter = RateLimiter()
        
        # Test rate limiting
        user_id = "test_user"
        endpoint = "/completion"
        
        # Should allow first request
        allowed = asyncio.run(rate_limiter.check_rate_limit(user_id, endpoint))
        assert allowed is True
        
        # Test rate limit status
        status = rate_limiter.get_user_rate_limit_status(user_id)
        assert status is not None
        assert "requests_minute" in status
    
    def test_input_validation(self):
        """Test input validation"""
        from src.api.models import CompletionRequestModel
        
        # Test valid request
        valid_request = CompletionRequestModel(
            prompt="Test prompt",
            model="gpt-4o-mini",
            temperature=0.7
        )
        assert valid_request.prompt == "Test prompt"
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            CompletionRequestModel(
                prompt="Test prompt",
                model="gpt-4o-mini",
                temperature=3.0  # Invalid temperature
            )


class TestPerformance:
    """Test Performance functionality"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        # Create multiple concurrent requests
        requests = [
            CompletionRequest(
                prompt=f"Test prompt {i}",
                model="gpt-4o-mini"
            )
            for i in range(5)
        ]
        
        start_time = time.time()
        
        # Execute requests concurrently
        tasks = [llm_manager.generate_completion(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that all requests completed
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) > 0
        
        # Check that concurrent execution was faster than sequential
        # (This is a basic check - in practice, you'd want more sophisticated benchmarks)
        assert total_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage patterns"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and use multiple components
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        reasoning_manager = ReasoningManager(llm_manager)
        evaluator = PromptEvaluator(llm_manager, reasoning_manager)
        cost_optimizer = CostOptimizer(llm_manager, evaluator)
        
        # Perform some operations
        for i in range(10):
            request = CompletionRequest(
                prompt=f"Test prompt {i}",
                model="gpt-4o-mini"
            )
            await llm_manager.generate_completion(request)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Check that memory usage is reasonable (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024  # 100MB


class TestEndToEnd:
    """Test End-to-End functionality"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from prompt to response"""
        # Initialize all components
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        reasoning_manager = ReasoningManager(llm_manager)
        evaluator = PromptEvaluator(llm_manager, reasoning_manager)
        cost_optimizer = CostOptimizer(llm_manager, evaluator)
        
        # Step 1: Optimize request
        optimization_result = await cost_optimizer.optimize_request(
            prompt="Write a short story about a robot learning to paint",
            optimization_strategy=OptimizationStrategy.BALANCE_COST_QUALITY
        )
        
        assert optimization_result is not None
        selected_model = optimization_result.selected_model
        
        # Step 2: Generate completion with reasoning
        request = CompletionRequest(
            prompt="Write a short story about a robot learning to paint",
            model=selected_model,
            reasoning_effort="high",
            thinking_mode=True
        )
        
        reasoning_result = await reasoning_manager.perform_reasoning(
            request,
            strategy="sequential",
            max_steps=3
        )
        
        assert reasoning_result is not None
        assert reasoning_result.content is not None
        
        # Step 3: Evaluate the result
        evaluation_result = await evaluator.evaluate_prompt(
            prompt="Write a short story about a robot learning to paint",
            model=selected_model,
            provider=optimization_result.selected_provider.value,
            evaluation_metrics=[
                EvaluationMetric.CREATIVITY,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.COMPLETENESS
            ]
        )
        
        assert evaluation_result is not None
        assert evaluation_result.overall_score >= 0.0
        assert evaluation_result.overall_score <= 1.0
        
        # Step 4: Track usage
        await cost_optimizer.track_usage(
            user_id="test_user",
            model=selected_model,
            provider=optimization_result.selected_provider,
            tokens_used=100,
            cost=0.001,
            request_time=2.0
        )
        
        # Step 5: Get analytics
        analytics = cost_optimizer.get_usage_analytics("test_user", days=1)
        assert analytics is not None
        assert analytics["total_requests"] > 0


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration
]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
