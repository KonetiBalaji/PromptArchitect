"""
Dynamic Template Selector using LLM-based Classification
Author: Balaji Koneti

Intelligently selects the best prompt template based on user input analysis
using LLM-powered classification and context understanding.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from ..llm_manager import LLMManager
from ..llm_providers.base import CompletionRequest, ProviderType
from ..prompt_builder import PromptTemplate, PromptRequest


logger = structlog.get_logger(__name__)


class ClassificationResult(BaseModel):
    """Result of template classification"""
    template_id: str = Field(..., description="Selected template ID")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Reasoning for selection")
    alternative_templates: List[Tuple[str, float]] = Field(default_factory=list, description="Alternative templates with scores")
    detected_intent: str = Field(..., description="Detected user intent")
    detected_domain: str = Field(..., description="Detected domain/field")
    complexity_score: float = Field(..., description="Complexity score (0-1)")


class ContextAnalysis(BaseModel):
    """Analysis of user input context"""
    intent: str = Field(..., description="Primary intent")
    domain: str = Field(..., description="Domain/field")
    complexity: str = Field(..., description="Complexity level")
    urgency: str = Field(..., description="Urgency level")
    audience: str = Field(..., description="Target audience")
    format_preference: str = Field(..., description="Preferred output format")
    length_requirement: str = Field(..., description="Length requirement")
    quality_requirement: str = Field(..., description="Quality requirement")


class DynamicSelector:
    """
    Dynamic template selector using LLM-based classification
    
    Features:
    - LLM-powered input analysis
    - Context-aware template selection
    - Confidence scoring
    - Alternative template suggestions
    - Learning from user feedback
    - Performance optimization
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize dynamic selector
        
        Args:
            llm_manager: LLM manager for classification
        """
        self.llm_manager = llm_manager
        self.classification_cache: Dict[str, ClassificationResult] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    async def select_template(self, user_input: str, 
                            available_templates: List[PromptTemplate],
                            context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Select the best template for user input
        
        Args:
            user_input: User input text
            available_templates: List of available templates
            context: Optional additional context
            
        Returns:
            Classification result with selected template
        """
        # Check cache first
        cache_key = self._generate_cache_key(user_input, available_templates)
        if cache_key in self.classification_cache:
            logger.debug("Using cached classification", cache_key=cache_key)
            return self.classification_cache[cache_key]
        
        # Analyze user input context
        context_analysis = await self._analyze_context(user_input, context)
        
        # Classify and select template
        classification = await self._classify_template(
            user_input, available_templates, context_analysis
        )
        
        # Cache the result
        self.classification_cache[cache_key] = classification
        
        logger.info("Template selected", 
                   template_id=classification.template_id,
                   confidence=classification.confidence,
                   intent=classification.detected_intent)
        
        return classification
    
    async def _analyze_context(self, user_input: str, 
                             context: Optional[Dict[str, Any]] = None) -> ContextAnalysis:
        """
        Analyze user input context using LLM
        
        Args:
            user_input: User input text
            context: Optional additional context
            
        Returns:
            Context analysis result
        """
        analysis_prompt = f"""
        Analyze the following user input and provide context analysis in JSON format:

        User Input: "{user_input}"

        Context: {context or "None"}

        Please analyze and return a JSON object with the following fields:
        - intent: Primary intent (summarization, question_answering, creative_writing, code_generation, analysis, other)
        - domain: Domain/field (technology, business, education, healthcare, finance, general, other)
        - complexity: Complexity level (simple, moderate, complex)
        - urgency: Urgency level (low, medium, high)
        - audience: Target audience (general, technical, expert, beginner, other)
        - format_preference: Preferred output format (text, json, markdown, structured, other)
        - length_requirement: Length requirement (short, medium, long, variable)
        - quality_requirement: Quality requirement (draft, standard, high, professional)

        Return only the JSON object, no additional text.
        """
        
        try:
            # Use a fast, cost-effective model for analysis
            request = CompletionRequest(
                prompt=analysis_prompt,
                model="gpt-4o-mini",  # Use mini model for cost efficiency
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            response = await self.llm_manager.generate_completion(request)
            
            # Parse JSON response
            analysis_data = json.loads(response.content)
            
            return ContextAnalysis(**analysis_data)
            
        except Exception as e:
            logger.warning("Context analysis failed, using defaults", error=str(e))
            # Return default analysis
            return ContextAnalysis(
                intent="other",
                domain="general",
                complexity="moderate",
                urgency="medium",
                audience="general",
                format_preference="text",
                length_requirement="medium",
                quality_requirement="standard"
            )
    
    async def _classify_template(self, user_input: str, 
                               available_templates: List[PromptTemplate],
                               context: ContextAnalysis) -> ClassificationResult:
        """
        Classify and select the best template
        
        Args:
            user_input: User input text
            available_templates: Available templates
            context: Context analysis
            
        Returns:
            Classification result
        """
        # Create template descriptions for classification
        template_descriptions = []
        for template in available_templates:
            template_descriptions.append({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "tags": template.tags
            })
        
        classification_prompt = f"""
        Based on the user input and context analysis, select the most appropriate template.

        User Input: "{user_input}"

        Context Analysis:
        - Intent: {context.intent}
        - Domain: {context.domain}
        - Complexity: {context.complexity}
        - Urgency: {context.urgency}
        - Audience: {context.audience}
        - Format Preference: {context.format_preference}
        - Length Requirement: {context.length_requirement}
        - Quality Requirement: {context.quality_requirement}

        Available Templates:
        {json.dumps(template_descriptions, indent=2)}

        Please analyze and return a JSON object with:
        - template_id: ID of the best matching template
        - confidence: Confidence score (0.0 to 1.0)
        - reasoning: Brief explanation of why this template was selected
        - alternative_templates: List of [template_id, score] pairs for other suitable templates
        - detected_intent: Refined intent based on analysis
        - detected_domain: Refined domain based on analysis
        - complexity_score: Complexity score (0.0 to 1.0)

        Return only the JSON object, no additional text.
        """
        
        try:
            request = CompletionRequest(
                prompt=classification_prompt,
                model="gpt-4o",  # Use main model for better classification
                temperature=0.2,  # Slightly higher for some creativity in reasoning
                max_tokens=800
            )
            
            response = await self.llm_manager.generate_completion(request)
            
            # Parse JSON response
            classification_data = json.loads(response.content)
            
            # Validate and create result
            result = ClassificationResult(
                template_id=classification_data["template_id"],
                confidence=float(classification_data["confidence"]),
                reasoning=classification_data["reasoning"],
                alternative_templates=[
                    (alt[0], float(alt[1])) 
                    for alt in classification_data.get("alternative_templates", [])
                ],
                detected_intent=classification_data.get("detected_intent", context.intent),
                detected_domain=classification_data.get("detected_domain", context.domain),
                complexity_score=float(classification_data.get("complexity_score", 0.5))
            )
            
            return result
            
        except Exception as e:
            logger.error("Template classification failed", error=str(e))
            # Fallback to simple rule-based selection
            return self._fallback_classification(user_input, available_templates, context)
    
    def _fallback_classification(self, user_input: str, 
                               available_templates: List[PromptTemplate],
                               context: ContextAnalysis) -> ClassificationResult:
        """
        Fallback classification using simple rules
        
        Args:
            user_input: User input text
            available_templates: Available templates
            context: Context analysis
            
        Returns:
            Classification result
        """
        input_lower = user_input.lower()
        
        # Simple keyword-based matching
        template_scores = {}
        
        for template in available_templates:
            score = 0.0
            
            # Match by category
            if context.intent == "summarization" and template.category == "summarization":
                score += 0.8
            elif context.intent == "question_answering" and template.category == "qa":
                score += 0.8
            elif context.intent == "creative_writing" and template.category == "creative":
                score += 0.8
            elif context.intent == "code_generation" and template.category == "programming":
                score += 0.8
            elif context.intent == "analysis" and template.category == "analysis":
                score += 0.8
            
            # Match by keywords
            for tag in template.tags:
                if tag in input_lower:
                    score += 0.2
            
            # Match by description keywords
            desc_lower = template.description.lower()
            if any(keyword in input_lower for keyword in ["summarize", "summary"]):
                if "summar" in desc_lower:
                    score += 0.3
            elif any(keyword in input_lower for keyword in ["question", "what", "how", "why"]):
                if "answer" in desc_lower or "question" in desc_lower:
                    score += 0.3
            elif any(keyword in input_lower for keyword in ["write", "create", "generate"]):
                if "write" in desc_lower or "creative" in desc_lower:
                    score += 0.3
            elif any(keyword in input_lower for keyword in ["code", "function", "program"]):
                if "code" in desc_lower or "program" in desc_lower:
                    score += 0.3
            elif any(keyword in input_lower for keyword in ["analyze", "analysis", "data"]):
                if "analyz" in desc_lower or "data" in desc_lower:
                    score += 0.3
            
            template_scores[template.id] = min(score, 1.0)
        
        # Select best template
        if template_scores:
            best_template_id = max(template_scores, key=template_scores.get)
            best_score = template_scores[best_template_id]
            
            # Create alternative templates
            alternatives = [
                (tid, score) for tid, score in template_scores.items() 
                if tid != best_template_id and score > 0.1
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            return ClassificationResult(
                template_id=best_template_id,
                confidence=best_score,
                reasoning=f"Selected based on keyword matching and category alignment",
                alternative_templates=alternatives[:3],  # Top 3 alternatives
                detected_intent=context.intent,
                detected_domain=context.domain,
                complexity_score=0.5  # Default complexity
            )
        else:
            # Default to first template
            return ClassificationResult(
                template_id=available_templates[0].id,
                confidence=0.5,
                reasoning="Default selection - no clear match found",
                alternative_templates=[],
                detected_intent=context.intent,
                detected_domain=context.domain,
                complexity_score=0.5
            )
    
    def _generate_cache_key(self, user_input: str, 
                          available_templates: List[PromptTemplate]) -> str:
        """
        Generate cache key for classification
        
        Args:
            user_input: User input text
            available_templates: Available templates
            
        Returns:
            Cache key
        """
        template_ids = sorted([t.id for t in available_templates])
        key_data = f"{user_input}_{'_'.join(template_ids)}"
        return f"classification_{hash(key_data) % 10000}"
    
    async def learn_from_feedback(self, classification: ClassificationResult, 
                                user_feedback: str, actual_performance: float) -> None:
        """
        Learn from user feedback to improve future classifications
        
        Args:
            classification: Original classification result
            user_feedback: User feedback about the selection
            actual_performance: Actual performance score (0-1)
        """
        # Store performance data
        template_id = classification.template_id
        if template_id not in self.performance_history:
            self.performance_history[template_id] = []
        
        self.performance_history[template_id].append(actual_performance)
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history[template_id]) > 100:
            self.performance_history[template_id] = self.performance_history[template_id][-100:]
        
        logger.info("Learned from feedback", 
                   template_id=template_id,
                   feedback=user_feedback,
                   performance=actual_performance)
    
    def get_template_performance(self, template_id: str) -> Dict[str, float]:
        """
        Get performance statistics for a template
        
        Args:
            template_id: Template identifier
            
        Returns:
            Performance statistics
        """
        if template_id not in self.performance_history:
            return {"avg_performance": 0.0, "total_feedback": 0, "recent_trend": 0.0}
        
        history = self.performance_history[template_id]
        
        if not history:
            return {"avg_performance": 0.0, "total_feedback": 0, "recent_trend": 0.0}
        
        avg_performance = sum(history) / len(history)
        total_feedback = len(history)
        
        # Calculate recent trend (last 10 vs previous 10)
        if len(history) >= 20:
            recent_avg = sum(history[-10:]) / 10
            previous_avg = sum(history[-20:-10]) / 10
            recent_trend = recent_avg - previous_avg
        else:
            recent_trend = 0.0
        
        return {
            "avg_performance": avg_performance,
            "total_feedback": total_feedback,
            "recent_trend": recent_trend
        }
    
    def clear_cache(self) -> None:
        """Clear classification cache"""
        self.classification_cache.clear()
        logger.info("Classification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics
        """
        return {
            "cached_classifications": len(self.classification_cache),
            "templates_with_feedback": len(self.performance_history),
            "total_feedback_entries": sum(len(history) for history in self.performance_history.values())
        }
