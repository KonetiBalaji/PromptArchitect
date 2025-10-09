"""
AI-Powered Template Generator
Author: Balaji Koneti

Generates custom prompt templates on-the-fly using LLM capabilities,
optimizes templates based on performance, and provides template
versioning and evolution for dynamic prompt engineering.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from ..llm_manager import LLMManager
from ..llm_providers.base import CompletionRequest, CompletionResponse, ProviderType
from .dynamic_selector import PromptTemplate

logger = structlog.get_logger(__name__)


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
    SCIENTIFIC = "scientific"
    ANALYTICAL = "analytical"


class TemplateComplexity(str, Enum):
    """Template complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TemplatePerformance(BaseModel):
    """Template performance metrics"""
    template_id: str = Field(..., description="Template identifier")
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: float = Field(default=0.0, description="Success rate (0-1)")
    average_quality_score: float = Field(default=0.0, description="Average quality score")
    user_satisfaction: float = Field(default=0.0, description="User satisfaction rating")
    cost_efficiency: float = Field(default=0.0, description="Cost efficiency score")
    response_time: float = Field(default=0.0, description="Average response time")
    last_updated: float = Field(default_factory=time.time, description="Last update timestamp")


class TemplateGenerationRequest(BaseModel):
    """Request for template generation"""
    category: TemplateCategory = Field(..., description="Template category")
    complexity: TemplateComplexity = Field(default=TemplateComplexity.MODERATE, description="Template complexity")
    use_case: str = Field(..., description="Specific use case description")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    constraints: List[str] = Field(default_factory=list, description="Constraints or limitations")
    target_audience: str = Field(default="general", description="Target audience")
    language: str = Field(default="english", description="Template language")
    reasoning_enabled: bool = Field(default=True, description="Enable reasoning integration")
    custom_instructions: Optional[str] = Field(default=None, description="Custom instructions")


class GeneratedTemplate(BaseModel):
    """Generated template with metadata"""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    complexity: TemplateComplexity = Field(..., description="Template complexity")
    template_content: str = Field(..., description="Template content")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    reasoning_integration: bool = Field(default=False, description="Whether reasoning is integrated")
    performance_metrics: TemplatePerformance = Field(..., description="Performance metrics")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    version: str = Field(default="1.0", description="Template version")


class TemplateOptimizationResult(BaseModel):
    """Result of template optimization"""
    original_template: GeneratedTemplate = Field(..., description="Original template")
    optimized_template: GeneratedTemplate = Field(..., description="Optimized template")
    improvements_made: List[str] = Field(..., description="List of improvements")
    performance_gain: float = Field(..., description="Expected performance improvement")
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict, description="Optimization details")


class TemplateGenerator:
    """
    AI-powered template generator
    
    Features:
    - Dynamic template generation based on use cases
    - Template optimization based on performance
    - Template versioning and evolution
    - Multi-category template support
    - Reasoning integration
    - Performance tracking and analytics
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize template generator
        
        Args:
            llm_manager: LLM manager for provider access
        """
        self.llm_manager = llm_manager
        self.generated_templates: Dict[str, GeneratedTemplate] = {}
        self.template_performance: Dict[str, TemplatePerformance] = {}
        self.generation_history: List[GeneratedTemplate] = []
    
    async def generate_template(self, request: TemplateGenerationRequest) -> GeneratedTemplate:
        """
        Generate a custom template based on requirements
        
        Args:
            request: Template generation request
            
        Returns:
            Generated template
        """
        try:
            logger.info("Generating template", 
                       category=request.category.value,
                       complexity=request.complexity.value,
                       use_case=request.use_case)
            
            # Generate template content
            template_content = await self._generate_template_content(request)
            
            # Extract variables from template
            variables = self._extract_template_variables(template_content)
            
            # Create template ID
            template_id = self._generate_template_id(request)
            
            # Create template name and description
            name, description = await self._generate_template_metadata(request, template_content)
            
            # Create performance metrics
            performance_metrics = TemplatePerformance(template_id=template_id)
            
            # Create generated template
            generated_template = GeneratedTemplate(
                template_id=template_id,
                name=name,
                description=description,
                category=request.category,
                complexity=request.complexity,
                template_content=template_content,
                variables=variables,
                reasoning_integration=request.reasoning_enabled,
                performance_metrics=performance_metrics,
                generation_metadata={
                    "use_case": request.use_case,
                    "requirements": request.requirements,
                    "constraints": request.constraints,
                    "target_audience": request.target_audience,
                    "language": request.language,
                    "generation_time": time.time()
                }
            )
            
            # Store template
            self.generated_templates[template_id] = generated_template
            self.template_performance[template_id] = performance_metrics
            self.generation_history.append(generated_template)
            
            logger.info("Template generated successfully", 
                       template_id=template_id,
                       variables_count=len(variables))
            
            return generated_template
            
        except Exception as e:
            logger.error("Template generation failed", error=str(e))
            raise
    
    async def _generate_template_content(self, request: TemplateGenerationRequest) -> str:
        """
        Generate template content using LLM
        
        Args:
            request: Template generation request
            
        Returns:
            Generated template content
        """
        # Create generation prompt
        generation_prompt = self._create_generation_prompt(request)
        
        # Create completion request
        completion_request = CompletionRequest(
            prompt=generation_prompt,
            model="gpt-4o",  # Use high-quality model for template generation
            temperature=0.7,
            max_tokens=2000,
            reasoning_effort="high" if request.complexity == TemplateComplexity.EXPERT else "medium",
            extended_thinking=request.complexity in [TemplateComplexity.COMPLEX, TemplateComplexity.EXPERT],
            thinking_mode=request.reasoning_enabled
        )
        
        # Select best provider
        provider_type = await self._select_template_provider(request)
        
        try:
            response = await self.llm_manager.generate_completion(
                completion_request, preferred_provider=provider_type
            )
            
            return response.reasoning_trace or response.content
            
        except Exception as e:
            logger.error("Template content generation failed", error=str(e))
            raise
    
    def _create_generation_prompt(self, request: TemplateGenerationRequest) -> str:
        """
        Create prompt for template generation
        
        Args:
            request: Template generation request
            
        Returns:
            Generation prompt
        """
        prompt = f"""
        You are an expert prompt engineer. Generate a high-quality prompt template for the following requirements:
        
        Category: {request.category.value.title()}
        Complexity: {request.complexity.value.title()}
        Use Case: {request.use_case}
        Target Audience: {request.target_audience}
        Language: {request.language}
        
        Requirements:
        {chr(10).join(f"- {req}" for req in request.requirements) if request.requirements else "- No specific requirements"}
        
        Constraints:
        {chr(10).join(f"- {constraint}" for constraint in request.constraints) if request.constraints else "- No specific constraints"}
        
        Custom Instructions:
        {request.custom_instructions if request.custom_instructions else "None"}
        
        Generate a comprehensive prompt template that:
        1. Is well-structured and clear
        2. Includes appropriate sections (role, task, context, etc.)
        3. Uses variables for customization (format: {{variable_name}})
        4. Incorporates reasoning elements if needed
        5. Is optimized for the specified complexity level
        6. Follows best practices for prompt engineering
        
        """
        
        if request.reasoning_enabled:
            prompt += """
        Since reasoning is enabled, include:
        - Step-by-step reasoning instructions
        - Verification and validation steps
        - Confidence assessment requirements
        - Error checking and correction guidance
        """
        
        if request.complexity == TemplateComplexity.SIMPLE:
            prompt += """
        Keep the template simple and straightforward, suitable for basic use cases.
        """
        elif request.complexity == TemplateComplexity.MODERATE:
            prompt += """
        Create a moderately complex template with good structure and clear instructions.
        """
        elif request.complexity == TemplateComplexity.COMPLEX:
            prompt += """
        Create a complex template with detailed instructions, multiple sections, and advanced features.
        """
        elif request.complexity == TemplateComplexity.EXPERT:
            prompt += """
        Create an expert-level template with sophisticated structure, advanced reasoning, and comprehensive coverage.
        """
        
        prompt += """
        
        Provide the template in the following format:
        
        **Template Name:** [Name]
        **Description:** [Brief description]
        **Template:**
        [The actual template content with variables in {{variable_name}} format]
        
        **Variables:**
        - {{variable_name}}: [Description of what this variable should contain]
        
        **Usage Notes:**
        [Any special instructions for using this template]
        """
        
        return prompt
    
    def _extract_template_variables(self, template_content: str) -> List[str]:
        """
        Extract variables from template content
        
        Args:
            template_content: Template content
            
        Returns:
            List of variable names
        """
        import re
        
        # Find variables in {{variable_name}} format
        variable_pattern = r'\{\{([^}]+)\}\}'
        matches = re.findall(variable_pattern, template_content)
        
        # Remove duplicates and clean
        variables = list(set(match.strip() for match in matches))
        
        return variables
    
    def _generate_template_id(self, request: TemplateGenerationRequest) -> str:
        """
        Generate unique template ID
        
        Args:
            request: Template generation request
            
        Returns:
            Unique template ID
        """
        import hashlib
        
        # Create hash from request parameters
        request_str = f"{request.category.value}_{request.complexity.value}_{request.use_case}_{time.time()}"
        template_id = hashlib.md5(request_str.encode()).hexdigest()[:12]
        
        return f"template_{template_id}"
    
    async def _generate_template_metadata(self, request: TemplateGenerationRequest, 
                                        template_content: str) -> Tuple[str, str]:
        """
        Generate template name and description
        
        Args:
            request: Template generation request
            template_content: Generated template content
            
        Returns:
            Tuple of (name, description)
        """
        # Extract name and description from template content
        import re
        
        name_match = re.search(r'\*\*Template Name:\*\*\s*(.+)', template_content)
        desc_match = re.search(r'\*\*Description:\*\*\s*(.+)', template_content)
        
        name = name_match.group(1).strip() if name_match else f"{request.category.value.title()} Template"
        description = desc_match.group(1).strip() if desc_match else f"Generated template for {request.use_case}"
        
        return name, description
    
    async def _select_template_provider(self, request: TemplateGenerationRequest) -> ProviderType:
        """
        Select best provider for template generation
        
        Args:
            request: Template generation request
            
        Returns:
            Provider type
        """
        # Get available providers
        available_providers = [
            ptype for ptype, pinfo in self.llm_manager.providers.items()
            if pinfo.status.value == "active" and pinfo.provider
        ]
        
        if not available_providers:
            raise RuntimeError("No available providers")
        
        # Prefer high-quality models for template generation
        priority_order = [ProviderType.OPENAI, ProviderType.CLAUDE, ProviderType.GEMINI]
        
        for provider_type in priority_order:
            if provider_type in available_providers:
                return provider_type
        
        return available_providers[0]
    
    async def optimize_template(self, template_id: str, 
                              performance_data: Optional[Dict[str, Any]] = None) -> TemplateOptimizationResult:
        """
        Optimize template based on performance data
        
        Args:
            template_id: Template ID to optimize
            performance_data: Performance data for optimization
            
        Returns:
            Template optimization result
        """
        if template_id not in self.generated_templates:
            raise ValueError(f"Template {template_id} not found")
        
        original_template = self.generated_templates[template_id]
        
        try:
            logger.info("Optimizing template", template_id=template_id)
            
            # Analyze performance data
            performance_analysis = await self._analyze_template_performance(
                original_template, performance_data
            )
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(
                original_template, performance_analysis
            )
            
            # Apply optimizations
            optimized_content = await self._apply_optimizations(
                original_template, optimization_suggestions
            )
            
            # Create optimized template
            optimized_template = GeneratedTemplate(
                template_id=f"{template_id}_optimized",
                name=f"{original_template.name} (Optimized)",
                description=f"Optimized version of {original_template.description}",
                category=original_template.category,
                complexity=original_template.complexity,
                template_content=optimized_content,
                variables=original_template.variables,
                reasoning_integration=original_template.reasoning_integration,
                performance_metrics=TemplatePerformance(template_id=f"{template_id}_optimized"),
                generation_metadata={
                    "optimization_of": template_id,
                    "optimization_time": time.time(),
                    "suggestions_applied": optimization_suggestions
                },
                version=f"{float(original_template.version) + 0.1:.1f}"
            )
            
            # Calculate performance gain
            performance_gain = self._calculate_performance_gain(
                original_template, optimized_template, performance_analysis
            )
            
            result = TemplateOptimizationResult(
                original_template=original_template,
                optimized_template=optimized_template,
                improvements_made=optimization_suggestions,
                performance_gain=performance_gain,
                optimization_metadata={
                    "optimization_time": time.time(),
                    "performance_analysis": performance_analysis
                }
            )
            
            logger.info("Template optimization completed", 
                       template_id=template_id,
                       performance_gain=performance_gain)
            
            return result
            
        except Exception as e:
            logger.error("Template optimization failed", error=str(e))
            raise
    
    async def _analyze_template_performance(self, template: GeneratedTemplate, 
                                          performance_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze template performance
        
        Args:
            template: Template to analyze
            performance_data: Performance data
            
        Returns:
            Performance analysis
        """
        analysis = {
            "template_id": template.template_id,
            "usage_count": template.performance_metrics.usage_count,
            "success_rate": template.performance_metrics.success_rate,
            "quality_score": template.performance_metrics.average_quality_score,
            "satisfaction": template.performance_metrics.user_satisfaction,
            "cost_efficiency": template.performance_metrics.cost_efficiency,
            "response_time": template.performance_metrics.response_time
        }
        
        if performance_data:
            analysis.update(performance_data)
        
        # Identify performance issues
        issues = []
        if analysis["success_rate"] < 0.7:
            issues.append("Low success rate")
        if analysis["quality_score"] < 0.6:
            issues.append("Low quality score")
        if analysis["satisfaction"] < 0.6:
            issues.append("Low user satisfaction")
        if analysis["response_time"] > 30.0:
            issues.append("Slow response time")
        
        analysis["issues"] = issues
        analysis["overall_score"] = (
            analysis["success_rate"] * 0.3 +
            analysis["quality_score"] * 0.3 +
            analysis["satisfaction"] * 0.2 +
            analysis["cost_efficiency"] * 0.2
        )
        
        return analysis
    
    async def _generate_optimization_suggestions(self, template: GeneratedTemplate, 
                                               performance_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate optimization suggestions
        
        Args:
            template: Template to optimize
            performance_analysis: Performance analysis
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze issues and generate suggestions
        for issue in performance_analysis.get("issues", []):
            if issue == "Low success rate":
                suggestions.append("Improve clarity and specificity of instructions")
                suggestions.append("Add more detailed examples and context")
            elif issue == "Low quality score":
                suggestions.append("Enhance template structure and organization")
                suggestions.append("Add quality checkpoints and validation steps")
            elif issue == "Low user satisfaction":
                suggestions.append("Improve user experience and ease of use")
                suggestions.append("Add customization options and flexibility")
            elif issue == "Slow response time":
                suggestions.append("Optimize template for faster processing")
                suggestions.append("Reduce complexity where possible")
        
        # General optimization suggestions
        if template.complexity == TemplateComplexity.EXPERT:
            suggestions.append("Consider simplifying for better accessibility")
        
        if not template.reasoning_integration:
            suggestions.append("Add reasoning and validation steps")
        
        if len(template.variables) > 10:
            suggestions.append("Reduce number of variables for simplicity")
        
        return suggestions
    
    async def _apply_optimizations(self, template: GeneratedTemplate, 
                                 suggestions: List[str]) -> str:
        """
        Apply optimization suggestions to template
        
        Args:
            template: Original template
            suggestions: Optimization suggestions
            
        Returns:
            Optimized template content
        """
        optimization_prompt = f"""
        Original template:
        {template.template_content}
        
        Optimization suggestions:
        {chr(10).join(f"- {suggestion}" for suggestion in suggestions)}
        
        Please optimize the template by applying these suggestions while maintaining:
        1. The core functionality and purpose
        2. All existing variables and their purposes
        3. The overall structure and quality
        4. Compatibility with the original use case
        
        Provide the optimized template content.
        """
        
        optimization_request = CompletionRequest(
            prompt=optimization_prompt,
            model="gpt-4o",
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            provider_type = await self._select_template_provider(
                TemplateGenerationRequest(
                    category=template.category,
                    complexity=template.complexity,
                    use_case="template optimization"
                )
            )
            
            response = await self.llm_manager.generate_completion(
                optimization_request, preferred_provider=provider_type
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Template optimization application failed", error=str(e))
            return template.template_content  # Return original if optimization fails
    
    def _calculate_performance_gain(self, original: GeneratedTemplate, 
                                  optimized: GeneratedTemplate,
                                  performance_analysis: Dict[str, Any]) -> float:
        """
        Calculate expected performance gain
        
        Args:
            original: Original template
            optimized: Optimized template
            performance_analysis: Performance analysis
            
        Returns:
            Expected performance gain (0-1)
        """
        # Simple heuristic based on number of improvements
        improvements_count = len(optimized.generation_metadata.get("suggestions_applied", []))
        
        # Base improvement from addressing issues
        issues_count = len(performance_analysis.get("issues", []))
        base_improvement = min(0.3, issues_count * 0.1)
        
        # Additional improvement from optimizations
        optimization_improvement = min(0.2, improvements_count * 0.05)
        
        total_improvement = base_improvement + optimization_improvement
        return min(1.0, total_improvement)
    
    def get_template(self, template_id: str) -> Optional[GeneratedTemplate]:
        """
        Get template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            Template or None if not found
        """
        return self.generated_templates.get(template_id)
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[GeneratedTemplate]:
        """
        Get templates by category
        
        Args:
            category: Template category
            
        Returns:
            List of templates in category
        """
        return [
            template for template in self.generated_templates.values()
            if template.category == category
        ]
    
    def get_templates_by_complexity(self, complexity: TemplateComplexity) -> List[GeneratedTemplate]:
        """
        Get templates by complexity
        
        Args:
            complexity: Template complexity
            
        Returns:
            List of templates with complexity
        """
        return [
            template for template in self.generated_templates.values()
            if template.complexity == complexity
        ]
    
    def update_template_performance(self, template_id: str, 
                                  performance_data: Dict[str, Any]) -> None:
        """
        Update template performance metrics
        
        Args:
            template_id: Template ID
            performance_data: Performance data
        """
        if template_id in self.template_performance:
            metrics = self.template_performance[template_id]
            
            # Update metrics
            if "usage_count" in performance_data:
                metrics.usage_count += performance_data["usage_count"]
            if "success_rate" in performance_data:
                metrics.success_rate = performance_data["success_rate"]
            if "quality_score" in performance_data:
                metrics.average_quality_score = performance_data["quality_score"]
            if "satisfaction" in performance_data:
                metrics.user_satisfaction = performance_data["satisfaction"]
            if "cost_efficiency" in performance_data:
                metrics.cost_efficiency = performance_data["cost_efficiency"]
            if "response_time" in performance_data:
                metrics.response_time = performance_data["response_time"]
            
            metrics.last_updated = time.time()
            
            # Update template
            if template_id in self.generated_templates:
                self.generated_templates[template_id].performance_metrics = metrics
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get template generation statistics
        
        Returns:
            Dictionary with generation statistics
        """
        if not self.generation_history:
            return {"total_templates_generated": 0}
        
        total_templates = len(self.generation_history)
        
        # Category distribution
        category_dist = {}
        for template in self.generation_history:
            category = template.category.value
            category_dist[category] = category_dist.get(category, 0) + 1
        
        # Complexity distribution
        complexity_dist = {}
        for template in self.generation_history:
            complexity = template.complexity.value
            complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1
        
        # Average performance
        avg_performance = {
            "usage_count": sum(t.performance_metrics.usage_count for t in self.generation_history) / total_templates,
            "success_rate": sum(t.performance_metrics.success_rate for t in self.generation_history) / total_templates,
            "quality_score": sum(t.performance_metrics.average_quality_score for t in self.generation_history) / total_templates,
            "satisfaction": sum(t.performance_metrics.user_satisfaction for t in self.generation_history) / total_templates
        }
        
        return {
            "total_templates_generated": total_templates,
            "category_distribution": category_dist,
            "complexity_distribution": complexity_dist,
            "average_performance": avg_performance,
            "reasoning_enabled_templates": sum(
                1 for t in self.generation_history if t.reasoning_integration
            ),
            "average_variables_per_template": sum(
                len(t.variables) for t in self.generation_history
            ) / total_templates
        }
