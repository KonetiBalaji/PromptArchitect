"""
Cost Optimization Engine
Author: Balaji Koneti

Intelligent cost optimization with model selection, prompt compression,
cost prediction, budgeting, and usage analytics.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from ..llm_manager import LLMManager
from ..llm_providers.base import CompletionRequest, ProviderType, ModelInfo
from ..evaluation.evaluator import PromptEvaluator, EvaluationMetric
import structlog

logger = structlog.get_logger(__name__)


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"
    BALANCE_COST_QUALITY = "balance_cost_quality"
    MAXIMIZE_QUALITY = "maximize_quality"
    MEET_BUDGET = "meet_budget"
    ADAPTIVE = "adaptive"


class ModelTier(str, Enum):
    """Model tiers for cost optimization"""
    BUDGET = "budget"      # Cheapest models
    STANDARD = "standard"  # Balanced cost/quality
    PREMIUM = "premium"    # Highest quality
    REASONING = "reasoning"  # Reasoning models


@dataclass
class CostProfile:
    """Cost profile for a model"""
    model: str
    provider: ProviderType
    cost_per_1k_tokens: float
    quality_score: float
    response_time: float
    tier: ModelTier
    use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of cost optimization"""
    selected_model: str
    selected_provider: ProviderType
    estimated_cost: float
    estimated_quality: float
    optimization_strategy: OptimizationStrategy
    reasoning: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Budget configuration"""
    total_budget: float
    daily_limit: float
    monthly_limit: float
    alert_threshold: float = 0.8  # Alert when 80% of budget is used
    current_usage: float = 0.0
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None


@dataclass
class UsageAnalytics:
    """Usage analytics for cost tracking"""
    user_id: str
    date: datetime
    total_requests: int
    total_tokens: int
    total_cost: float
    model_usage: Dict[str, int] = field(default_factory=dict)
    provider_usage: Dict[str, int] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)


class CostOptimizer:
    """Main cost optimization engine"""
    
    def __init__(self, llm_manager: LLMManager, evaluator: PromptEvaluator):
        self.llm_manager = llm_manager
        self.evaluator = evaluator
        self.cost_profiles: Dict[str, CostProfile] = {}
        self.usage_history: List[UsageAnalytics] = []
        self.budgets: Dict[str, Budget] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Initialize cost profiles
        self._initialize_cost_profiles()
        
        logger.info("CostOptimizer initialized")
    
    def _initialize_cost_profiles(self):
        """Initialize cost profiles for different models"""
        # OpenAI models
        self.cost_profiles["gpt-4o"] = CostProfile(
            model="gpt-4o",
            provider=ProviderType.OPENAI,
            cost_per_1k_tokens=0.005,  # $5 per 1M tokens
            quality_score=0.95,
            response_time=2.0,
            tier=ModelTier.PREMIUM,
            use_cases=["complex_reasoning", "creative_writing", "analysis"],
            limitations=["higher_cost", "slower_response"]
        )
        
        self.cost_profiles["gpt-4o-mini"] = CostProfile(
            model="gpt-4o-mini",
            provider=ProviderType.OPENAI,
            cost_per_1k_tokens=0.00015,  # $0.15 per 1M tokens
            quality_score=0.85,
            response_time=1.0,
            tier=ModelTier.STANDARD,
            use_cases=["general_purpose", "quick_tasks", "prototyping"],
            limitations=["lower_quality_for_complex_tasks"]
        )
        
        self.cost_profiles["gpt-3.5-turbo"] = CostProfile(
            model="gpt-3.5-turbo",
            provider=ProviderType.OPENAI,
            cost_per_1k_tokens=0.0005,  # $0.5 per 1M tokens
            quality_score=0.75,
            response_time=0.8,
            tier=ModelTier.BUDGET,
            use_cases=["simple_tasks", "high_volume", "testing"],
            limitations=["limited_reasoning", "lower_quality"]
        )
        
        # Reasoning models
        self.cost_profiles["o1-preview"] = CostProfile(
            model="o1-preview",
            provider=ProviderType.OPENAI,
            cost_per_1k_tokens=0.015,  # $15 per 1M tokens
            quality_score=0.98,
            response_time=10.0,
            tier=ModelTier.REASONING,
            use_cases=["complex_reasoning", "mathematical_problems", "analysis"],
            limitations=["very_expensive", "slow", "no_streaming"]
        )
        
        # Claude models
        self.cost_profiles["claude-3-5-sonnet-20241022"] = CostProfile(
            model="claude-3-5-sonnet-20241022",
            provider=ProviderType.CLAUDE,
            cost_per_1k_tokens=0.003,  # $3 per 1M tokens
            quality_score=0.92,
            response_time=2.5,
            tier=ModelTier.PREMIUM,
            use_cases=["analysis", "writing", "reasoning"],
            limitations=["higher_cost"]
        )
        
        self.cost_profiles["claude-3-haiku-20240307"] = CostProfile(
            model="claude-3-haiku-20240307",
            provider=ProviderType.CLAUDE,
            cost_per_1k_tokens=0.00025,  # $0.25 per 1M tokens
            quality_score=0.80,
            response_time=1.2,
            tier=ModelTier.BUDGET,
            use_cases=["quick_tasks", "simple_analysis"],
            limitations=["lower_quality_for_complex_tasks"]
        )
        
        # Gemini models
        self.cost_profiles["gemini-1.5-pro"] = CostProfile(
            model="gemini-1.5-pro",
            provider=ProviderType.GEMINI,
            cost_per_1k_tokens=0.00125,  # $1.25 per 1M tokens
            quality_score=0.88,
            response_time=1.8,
            tier=ModelTier.STANDARD,
            use_cases=["general_purpose", "multimodal"],
            limitations=["limited_reasoning"]
        )
        
        self.cost_profiles["gemini-1.5-flash"] = CostProfile(
            model="gemini-1.5-flash",
            provider=ProviderType.GEMINI,
            cost_per_1k_tokens=0.000075,  # $0.075 per 1M tokens
            quality_score=0.82,
            response_time=0.9,
            tier=ModelTier.BUDGET,
            use_cases=["high_volume", "quick_tasks"],
            limitations=["lower_quality"]
        )
    
    async def optimize_request(
        self,
        prompt: str,
        optimization_strategy: OptimizationStrategy,
        budget: Optional[Budget] = None,
        quality_threshold: float = 0.8,
        max_cost: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> OptimizationResult:
        """Optimize a request for cost and quality"""
        
        # Analyze prompt characteristics
        prompt_analysis = await self._analyze_prompt(prompt)
        
        # Get available models
        available_models = await self._get_available_models()
        
        # Filter models based on constraints
        candidate_models = await self._filter_candidate_models(
            available_models, prompt_analysis, budget, max_cost, quality_threshold
        )
        
        if not candidate_models:
            raise ValueError("No suitable models found for the given constraints")
        
        # Select optimal model based on strategy
        selected_model, selected_provider, reasoning = await self._select_optimal_model(
            candidate_models, prompt_analysis, optimization_strategy
        )
        
        # Estimate cost and quality
        estimated_cost = await self._estimate_cost(prompt, selected_model, selected_provider)
        estimated_quality = self.cost_profiles[selected_model].quality_score
        
        # Generate alternatives
        alternatives = await self._generate_alternatives(
            candidate_models, prompt_analysis, optimization_strategy
        )
        
        # Create optimization result
        result = OptimizationResult(
            selected_model=selected_model,
            selected_provider=selected_provider,
            estimated_cost=estimated_cost,
            estimated_quality=estimated_quality,
            optimization_strategy=optimization_strategy,
            reasoning=reasoning,
            alternatives=alternatives,
            metadata={
                "prompt_analysis": prompt_analysis,
                "user_id": user_id,
                "timestamp": datetime.now()
            }
        )
        
        # Store result
        self.optimization_history.append(result)
        
        logger.info("Request optimized", 
                   model=selected_model, 
                   cost=estimated_cost, 
                   quality=estimated_quality)
        
        return result
    
    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt characteristics"""
        analysis = {
            "length": len(prompt.split()),
            "complexity": "simple",
            "task_type": "general",
            "requires_reasoning": False,
            "requires_creativity": False,
            "requires_accuracy": False
        }
        
        # Analyze prompt length
        word_count = len(prompt.split())
        if word_count > 500:
            analysis["complexity"] = "complex"
        elif word_count > 100:
            analysis["complexity"] = "medium"
        
        # Analyze task type and requirements
        prompt_lower = prompt.lower()
        
        # Check for reasoning requirements
        reasoning_indicators = [
            "analyze", "explain", "reason", "logic", "why", "how",
            "compare", "evaluate", "assess", "determine"
        ]
        if any(indicator in prompt_lower for indicator in reasoning_indicators):
            analysis["requires_reasoning"] = True
            analysis["task_type"] = "reasoning"
        
        # Check for creativity requirements
        creativity_indicators = [
            "creative", "imagine", "story", "write", "generate",
            "invent", "design", "artistic", "poem", "song"
        ]
        if any(indicator in prompt_lower for indicator in creativity_indicators):
            analysis["requires_creativity"] = True
            analysis["task_type"] = "creative"
        
        # Check for accuracy requirements
        accuracy_indicators = [
            "accurate", "precise", "exact", "correct", "factual",
            "calculate", "compute", "solve", "determine"
        ]
        if any(indicator in prompt_lower for indicator in accuracy_indicators):
            analysis["requires_accuracy"] = True
            analysis["task_type"] = "analytical"
        
        return analysis
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        # This would check with the LLM manager for available models
        # For now, return all models in cost profiles
        return list(self.cost_profiles.keys())
    
    async def _filter_candidate_models(
        self,
        available_models: List[str],
        prompt_analysis: Dict[str, Any],
        budget: Optional[Budget],
        max_cost: Optional[float],
        quality_threshold: float
    ) -> List[str]:
        """Filter candidate models based on constraints"""
        candidates = []
        
        for model in available_models:
            profile = self.cost_profiles[model]
            
            # Check quality threshold
            if profile.quality_score < quality_threshold:
                continue
            
            # Check budget constraints
            if budget:
                estimated_cost = await self._estimate_cost("", model, profile.provider)
                if budget.current_usage + estimated_cost > budget.daily_limit:
                    continue
            
            # Check max cost constraint
            if max_cost:
                estimated_cost = await self._estimate_cost("", model, profile.provider)
                if estimated_cost > max_cost:
                    continue
            
            # Check task-specific requirements
            if prompt_analysis["requires_reasoning"] and profile.tier != ModelTier.REASONING:
                # For complex reasoning, prefer reasoning models but allow others
                if profile.quality_score < 0.9:
                    continue
            
            candidates.append(model)
        
        return candidates
    
    async def _select_optimal_model(
        self,
        candidate_models: List[str],
        prompt_analysis: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> Tuple[str, ProviderType, str]:
        """Select optimal model based on strategy"""
        
        if strategy == OptimizationStrategy.MINIMIZE_COST:
            # Select cheapest model that meets quality threshold
            best_model = min(candidate_models, 
                           key=lambda m: self.cost_profiles[m].cost_per_1k_tokens)
            reasoning = f"Selected {best_model} for minimum cost"
            
        elif strategy == OptimizationStrategy.MAXIMIZE_QUALITY:
            # Select highest quality model
            best_model = max(candidate_models, 
                           key=lambda m: self.cost_profiles[m].quality_score)
            reasoning = f"Selected {best_model} for maximum quality"
            
        elif strategy == OptimizationStrategy.BALANCE_COST_QUALITY:
            # Select model with best cost-quality ratio
            best_model = max(candidate_models, 
                           key=lambda m: self.cost_profiles[m].quality_score / 
                                       self.cost_profiles[m].cost_per_1k_tokens)
            reasoning = f"Selected {best_model} for optimal cost-quality balance"
            
        elif strategy == OptimizationStrategy.ADAPTIVE:
            # Adaptive selection based on prompt characteristics
            if prompt_analysis["requires_reasoning"]:
                reasoning_models = [m for m in candidate_models 
                                  if self.cost_profiles[m].tier == ModelTier.REASONING]
                if reasoning_models:
                    best_model = max(reasoning_models, 
                                   key=lambda m: self.cost_profiles[m].quality_score)
                    reasoning = f"Selected {best_model} for complex reasoning task"
                else:
                    best_model = max(candidate_models, 
                                   key=lambda m: self.cost_profiles[m].quality_score)
                    reasoning = f"Selected {best_model} for reasoning task (reasoning models unavailable)"
            else:
                best_model = max(candidate_models, 
                               key=lambda m: self.cost_profiles[m].quality_score / 
                                           self.cost_profiles[m].cost_per_1k_tokens)
                reasoning = f"Selected {best_model} for general task"
        else:
            # Default to balance strategy
            best_model = max(candidate_models, 
                           key=lambda m: self.cost_profiles[m].quality_score / 
                                       self.cost_profiles[m].cost_per_1k_tokens)
            reasoning = f"Selected {best_model} using default strategy"
        
        provider = self.cost_profiles[best_model].provider
        return best_model, provider, reasoning
    
    async def _estimate_cost(self, prompt: str, model: str, provider: ProviderType) -> float:
        """Estimate cost for a request"""
        profile = self.cost_profiles[model]
        
        # Estimate token count (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        
        # Estimate cost
        estimated_cost = (estimated_tokens / 1000) * profile.cost_per_1k_tokens
        
        return estimated_cost
    
    async def _generate_alternatives(
        self,
        candidate_models: List[str],
        prompt_analysis: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> List[Dict[str, Any]]:
        """Generate alternative model options"""
        alternatives = []
        
        for model in candidate_models[:3]:  # Top 3 alternatives
            profile = self.cost_profiles[model]
            estimated_cost = await self._estimate_cost("", model, profile.provider)
            
            alternative = {
                "model": model,
                "provider": profile.provider.value,
                "estimated_cost": estimated_cost,
                "quality_score": profile.quality_score,
                "response_time": profile.response_time,
                "tier": profile.tier.value,
                "reasoning": f"Alternative: {profile.tier.value} tier model"
            }
            
            alternatives.append(alternative)
        
        return alternatives
    
    async def track_usage(
        self,
        user_id: str,
        model: str,
        provider: ProviderType,
        tokens_used: int,
        cost: float,
        request_time: float
    ):
        """Track usage for analytics"""
        today = datetime.now().date()
        
        # Find or create today's analytics
        today_analytics = None
        for analytics in self.usage_history:
            if analytics.user_id == user_id and analytics.date.date() == today:
                today_analytics = analytics
                break
        
        if not today_analytics:
            today_analytics = UsageAnalytics(
                user_id=user_id,
                date=datetime.now(),
                total_requests=0,
                total_tokens=0,
                total_cost=0.0
            )
            self.usage_history.append(today_analytics)
        
        # Update analytics
        today_analytics.total_requests += 1
        today_analytics.total_tokens += tokens_used
        today_analytics.total_cost += cost
        
        # Update model usage
        if model not in today_analytics.model_usage:
            today_analytics.model_usage[model] = 0
        today_analytics.model_usage[model] += 1
        
        # Update provider usage
        provider_name = provider.value
        if provider_name not in today_analytics.provider_usage:
            today_analytics.provider_usage[provider_name] = 0
        today_analytics.provider_usage[provider_name] += 1
        
        # Update cost breakdown
        if model not in today_analytics.cost_breakdown:
            today_analytics.cost_breakdown[model] = 0.0
        today_analytics.cost_breakdown[model] += cost
        
        logger.info("Usage tracked", user_id=user_id, model=model, cost=cost)
    
    def set_budget(self, user_id: str, budget: Budget):
        """Set budget for a user"""
        self.budgets[user_id] = budget
        logger.info("Budget set", user_id=user_id, budget=budget.total_budget)
    
    def get_budget_status(self, user_id: str) -> Dict[str, Any]:
        """Get budget status for a user"""
        if user_id not in self.budgets:
            return {"status": "no_budget_set"}
        
        budget = self.budgets[user_id]
        
        # Calculate current usage
        today = datetime.now().date()
        today_usage = 0.0
        
        for analytics in self.usage_history:
            if analytics.user_id == user_id and analytics.date.date() == today:
                today_usage += analytics.total_cost
        
        budget.current_usage = today_usage
        
        # Calculate status
        daily_usage_percent = (today_usage / budget.daily_limit) * 100
        status = "healthy"
        
        if daily_usage_percent >= budget.alert_threshold * 100:
            status = "warning"
        if daily_usage_percent >= 100:
            status = "exceeded"
        
        return {
            "status": status,
            "daily_limit": budget.daily_limit,
            "current_usage": today_usage,
            "usage_percentage": daily_usage_percent,
            "remaining": budget.daily_limit - today_usage,
            "alert_threshold": budget.alert_threshold * 100
        }
    
    def get_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for a user"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_analytics = [
            analytics for analytics in self.usage_history
            if analytics.user_id == user_id and analytics.date >= cutoff_date
        ]
        
        if not user_analytics:
            return {"message": "No usage data found"}
        
        # Calculate totals
        total_requests = sum(analytics.total_requests for analytics in user_analytics)
        total_tokens = sum(analytics.total_tokens for analytics in user_analytics)
        total_cost = sum(analytics.total_cost for analytics in user_analytics)
        
        # Calculate averages
        avg_requests_per_day = total_requests / days
        avg_tokens_per_day = total_tokens / days
        avg_cost_per_day = total_cost / days
        
        # Model usage breakdown
        model_usage = {}
        for analytics in user_analytics:
            for model, count in analytics.model_usage.items():
                if model not in model_usage:
                    model_usage[model] = 0
                model_usage[model] += count
        
        # Provider usage breakdown
        provider_usage = {}
        for analytics in user_analytics:
            for provider, count in analytics.provider_usage.items():
                if provider not in provider_usage:
                    provider_usage[provider] = 0
                provider_usage[provider] += count
        
        return {
            "period_days": days,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "averages": {
                "requests_per_day": avg_requests_per_day,
                "tokens_per_day": avg_tokens_per_day,
                "cost_per_day": avg_cost_per_day
            },
            "model_usage": model_usage,
            "provider_usage": provider_usage,
            "daily_breakdown": [
                {
                    "date": analytics.date.date().isoformat(),
                    "requests": analytics.total_requests,
                    "tokens": analytics.total_tokens,
                    "cost": analytics.total_cost
                }
                for analytics in user_analytics
            ]
        }
    
    def get_cost_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        recommendations = []
        
        # Get user's usage analytics
        analytics = self.get_usage_analytics(user_id, days=30)
        
        if "message" in analytics:
            return [{"type": "info", "message": "No usage data available for recommendations"}]
        
        # Analyze model usage
        model_usage = analytics["model_usage"]
        total_requests = analytics["total_requests"]
        
        # Recommendation 1: Switch to cheaper models for simple tasks
        expensive_models = ["gpt-4o", "o1-preview", "claude-3-5-sonnet-20241022"]
        for model in expensive_models:
            if model in model_usage:
                usage_percent = (model_usage[model] / total_requests) * 100
                if usage_percent > 20:  # If using expensive model for >20% of requests
                    recommendations.append({
                        "type": "cost_savings",
                        "title": f"Consider switching from {model}",
                        "description": f"You're using {model} for {usage_percent:.1f}% of requests. Consider using cheaper alternatives for simple tasks.",
                        "potential_savings": f"Up to 60% cost reduction",
                        "action": f"Use gpt-4o-mini or gemini-1.5-flash for simple tasks"
                    })
        
        # Recommendation 2: Use reasoning models only when needed
        if "o1-preview" in model_usage:
            reasoning_usage = model_usage["o1-preview"]
            if reasoning_usage > 10:  # If using reasoning model frequently
                recommendations.append({
                    "type": "optimization",
                    "title": "Optimize reasoning model usage",
                    "description": f"You're using o1-preview for {reasoning_usage} requests. Use it only for complex reasoning tasks.",
                    "potential_savings": f"Up to 80% cost reduction for non-reasoning tasks",
                    "action": "Use o1-preview only for complex analysis and reasoning tasks"
                })
        
        # Recommendation 3: Batch similar requests
        if analytics["total_requests"] > 100:
            recommendations.append({
                "type": "efficiency",
                "title": "Consider request batching",
                "description": "You have high request volume. Consider batching similar requests to reduce costs.",
                "potential_savings": f"Up to 30% cost reduction",
                "action": "Group similar prompts and process them together"
            })
        
        return recommendations
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        # Calculate statistics
        total_optimizations = len(self.optimization_history)
        
        # Strategy usage
        strategy_usage = {}
        for result in self.optimization_history:
            strategy = result.optimization_strategy.value
            if strategy not in strategy_usage:
                strategy_usage[strategy] = 0
            strategy_usage[strategy] += 1
        
        # Model selection frequency
        model_selection = {}
        for result in self.optimization_history:
            model = result.selected_model
            if model not in model_selection:
                model_selection[model] = 0
            model_selection[model] += 1
        
        # Cost savings analysis
        cost_savings = []
        for result in self.optimization_history:
            if result.alternatives:
                cheapest_alternative = min(result.alternatives, key=lambda x: x["estimated_cost"])
                savings = result.estimated_cost - cheapest_alternative["estimated_cost"]
                cost_savings.append(savings)
        
        avg_cost_savings = statistics.mean(cost_savings) if cost_savings else 0.0
        
        return {
            "total_optimizations": total_optimizations,
            "strategy_usage": strategy_usage,
            "model_selection_frequency": model_selection,
            "average_cost_savings": avg_cost_savings,
            "total_potential_savings": sum(cost_savings),
            "optimization_success_rate": 1.0  # All optimizations are successful by definition
        }
