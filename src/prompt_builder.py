"""
Enhanced Prompt Builder with Multi-LLM Support and Caching
Author: Balaji Koneti

Builds structured prompts with role, task, context, reasoning, output format,
and stop conditions. Integrates with multi-LLM support and caching.
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import structlog

from llm_manager import LLMManager, LLMManagerConfig
from llm_providers.base import CompletionRequest, CompletionResponse, ProviderType
from cache.cache_manager import CacheManager, CacheConfig


logger = structlog.get_logger(__name__)


class PromptTemplate(BaseModel):
    """Template for structured prompts

    Supports two shapes:
    - Sectioned templates (role/task/context/reasoning/output/stop)
    - Raw templates via `template` (single monolithic prompt)
    """
    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable template name")
    description: str = Field(..., description="Template description")
    # Sectioned shape
    role_template: Optional[str] = Field(default=None, description="Role definition template")
    task_template: Optional[str] = Field(default=None, description="Task description template")
    context_template: Optional[str] = Field(default=None, description="Context template")
    reasoning_template: Optional[str] = Field(default=None, description="Reasoning/strategy template")
    output_format_template: Optional[str] = Field(default=None, description="Output format template")
    stop_condition_template: Optional[str] = Field(default=None, description="Stop condition template")
    # Raw monolithic shape
    template: Optional[str] = Field(default=None, description="Raw monolithic template content with {variables}")
    variables: List[str] = Field(default_factory=list, description="Required template variables")
    category: str = Field(default="general", description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")


class PromptRequest(BaseModel):
    """Request for prompt generation"""
    user_input: str = Field(..., description="User input text")
    template_id: Optional[str] = Field(default=None, description="Specific template to use")
    role: Optional[str] = Field(default=None, description="Custom role override")
    task: Optional[str] = Field(default=None, description="Custom task override")
    context: Optional[str] = Field(default=None, description="Custom context override")
    reasoning: Optional[str] = Field(default=None, description="Custom reasoning override")
    output_format: Optional[str] = Field(default=None, description="Custom output format override")
    stop_condition: Optional[str] = Field(default=None, description="Custom stop condition override")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    provider: Optional[ProviderType] = Field(default=None, description="Preferred LLM provider")
    model: Optional[str] = Field(default=None, description="Preferred model")
    use_cache: bool = Field(default=True, description="Whether to use cached prompts")


class GeneratedPrompt(BaseModel):
    """Generated structured prompt"""
    template_id: str = Field(..., description="Template used")
    role: str = Field(..., description="Generated role")
    task: str = Field(..., description="Generated task")
    context: str = Field(..., description="Generated context")
    reasoning: str = Field(..., description="Generated reasoning")
    output_format: str = Field(..., description="Generated output format")
    stop_condition: str = Field(..., description="Generated stop condition")
    full_prompt: str = Field(..., description="Complete structured prompt")
    variables_used: Dict[str, Any] = Field(default_factory=dict, description="Variables used")
    generation_time: float = Field(..., description="Time taken to generate")
    cached: bool = Field(default=False, description="Whether this was from cache")


class PromptBuilder:
    """
    Enhanced prompt builder with multi-LLM support and caching
    
    Features:
    - Structured prompt generation
    - Template-based system
    - Multi-LLM integration
    - Response caching
    - Dynamic template selection
    - Cost optimization
    """
    
    def __init__(self, llm_manager: LLMManager, cache_manager: Optional[CacheManager] = None):
        """
        Initialize prompt builder
        
        Args:
            llm_manager: LLM manager for multi-provider support
            cache_manager: Optional cache manager for prompt caching
        """
        self.llm_manager = llm_manager
        self.cache_manager = cache_manager
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates"""
        default_templates = [
            PromptTemplate(
                id="summarization",
                name="Document Summarization",
                description="Summarize documents with focus on key points",
                role_template="You are an expert {domain} analyst with extensive experience in document analysis and summarization.",
                task_template="Summarize the following {document_type} with focus on {focus_areas}.",
                context_template="The document is a {document_type} discussing {topic}. It contains {length} words and covers {sections}.",
                reasoning_template="Identify the main arguments, key findings, and actionable insights. Prioritize information that is most relevant to {audience}.",
                output_format_template="Provide a structured summary in {format} format with the following sections: {sections}.",
                stop_condition_template="Stop after providing the summary. Do not include commentary or additional analysis.",
                variables=["domain", "document_type", "focus_areas", "topic", "length", "sections", "audience", "format"],
                category="summarization",
                tags=["document", "analysis", "summary"]
            ),
            PromptTemplate(
                id="qa",
                name="Question Answering",
                description="Answer questions with detailed explanations",
                role_template="You are a knowledgeable {domain} expert with deep understanding of {subject_area}.",
                task_template="Answer the following question about {topic}: {question}",
                context_template="The question relates to {context}. Consider the following factors: {factors}.",
                reasoning_template="Provide a comprehensive answer by: 1) Understanding the question, 2) Gathering relevant information, 3) Synthesizing the answer, 4) Providing examples if helpful.",
                output_format_template="Structure your answer as: {format}",
                stop_condition_template="Provide a complete answer without asking follow-up questions.",
                variables=["domain", "subject_area", "topic", "question", "context", "factors", "format"],
                category="qa",
                tags=["question", "answer", "explanation"]
            ),
            PromptTemplate(
                id="creative_writing",
                name="Creative Writing",
                description="Generate creative content with specific style and tone",
                role_template="You are a creative writer specializing in {genre} with a {style} writing style.",
                task_template="Write a {content_type} about {topic} in the style of {style}.",
                context_template="The piece should be {length} words, target {audience}, and convey {mood}.",
                reasoning_template="Consider the narrative structure, character development, and thematic elements. Ensure the writing flows naturally and engages the reader.",
                output_format_template="Format the output as {format} with proper {structure}.",
                stop_condition_template="Complete the piece without leaving it unfinished.",
                variables=["genre", "style", "content_type", "topic", "length", "audience", "mood", "format", "structure"],
                category="creative",
                tags=["writing", "creative", "story"]
            ),
            PromptTemplate(
                id="code_generation",
                name="Code Generation",
                description="Generate code with proper structure and documentation",
                role_template="You are a senior {language} developer with expertise in {framework} and {domain}.",
                task_template="Generate {code_type} code that {functionality}.",
                context_template="The code should be {quality_level}, follow {standards}, and be compatible with {environment}.",
                reasoning_template="Consider: 1) Code structure and organization, 2) Error handling, 3) Performance optimization, 4) Documentation and comments, 5) Testing considerations.",
                output_format_template="Provide the code with {documentation_level} documentation and {comment_style} comments.",
                stop_condition_template="Include only the code and documentation. Do not provide additional explanations.",
                variables=["language", "framework", "domain", "code_type", "functionality", "quality_level", "standards", "environment", "documentation_level", "comment_style"],
                category="programming",
                tags=["code", "programming", "development"]
            ),
            PromptTemplate(
                id="analysis",
                name="Data Analysis",
                description="Analyze data and provide insights",
                role_template="You are a data analyst with expertise in {domain} and experience with {tools}.",
                task_template="Analyze the following {data_type} data and provide insights about {analysis_focus}.",
                context_template="The dataset contains {data_description} and was collected {collection_method}.",
                reasoning_template="Follow this analytical approach: 1) Data exploration, 2) Pattern identification, 3) Statistical analysis, 4) Insight generation, 5) Recommendation formulation.",
                output_format_template="Present your analysis in {format} with the following sections: {sections}.",
                stop_condition_template="Provide actionable insights and recommendations. Do not include raw data.",
                variables=["domain", "tools", "data_type", "analysis_focus", "data_description", "collection_method", "format", "sections"],
                category="analysis",
                tags=["data", "analysis", "insights"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
        
        logger.info("Loaded default templates", count=len(default_templates))
    
    async def generate_prompt(self, request: PromptRequest) -> GeneratedPrompt:
        """
        Generate a structured prompt from user input
        
        Args:
            request: Prompt generation request
            
        Returns:
            Generated structured prompt
            
        Raises:
            ValueError: If template is not found or variables are missing
        """
        start_time = time.time()
        
        # Check cache first
        if request.use_cache and self.cache_manager:
            cached_prompt = await self._get_cached_prompt(request)
            if cached_prompt:
                logger.debug("Using cached prompt", template_id=cached_prompt.template_id)
                return cached_prompt
        
        # Select template
        template = await self._select_template(request)
        
        # If raw template is provided, use it directly
        if template.template:
            filled = self._fill_template(template.template, request.variables)
            prompt_components = {
                "role": request.role or "assistant",
                "task": request.task or "",
                "context": request.context or "",
                "reasoning": request.reasoning or "",
                "output_format": request.output_format or "",
                "stop_condition": request.stop_condition or ""
            }
            full_prompt = filled
        else:
            # Generate prompt components and assemble sectioned prompt
            prompt_components = await self._generate_components(request, template)
            full_prompt = self._build_full_prompt(prompt_components, request.user_input)
        
        # Create generated prompt
        generated_prompt = GeneratedPrompt(
            template_id=template.id,
            role=prompt_components["role"],
            task=prompt_components["task"],
            context=prompt_components["context"],
            reasoning=prompt_components["reasoning"],
            output_format=prompt_components["output_format"],
            stop_condition=prompt_components["stop_condition"],
            full_prompt=full_prompt,
            variables_used=request.variables,
            generation_time=time.time() - start_time
        )
        
        # Cache the generated prompt
        if request.use_cache and self.cache_manager:
            await self._cache_prompt(request, generated_prompt)
        
        logger.info("Generated prompt", 
                   template_id=template.id,
                   generation_time=generated_prompt.generation_time)
        
        return generated_prompt
    
    async def _select_template(self, request: PromptRequest) -> PromptTemplate:
        """
        Select the appropriate template for the request
        
        Args:
            request: Prompt generation request
            
        Returns:
            Selected template
            
        Raises:
            ValueError: If template is not found
        """
        # Use specific template if requested
        if request.template_id:
            if request.template_id not in self.templates:
                raise ValueError(f"Template not found: {request.template_id}")
            return self.templates[request.template_id]
        
        # Auto-select template based on content analysis
        # This is a simplified version - in a real implementation,
        # you might use an LLM to analyze the input and select the best template
        template = await self._auto_select_template(request.user_input)
        return template
    
    async def _auto_select_template(self, user_input: str) -> PromptTemplate:
        """
        Automatically select template based on user input analysis
        
        Args:
            user_input: User input text
            
        Returns:
            Selected template
        """
        # Simple keyword-based template selection
        # In a real implementation, this could use an LLM for more sophisticated analysis
        input_lower = user_input.lower()
        
        if any(keyword in input_lower for keyword in ["summarize", "summary", "overview"]):
            return self.templates["summarization"]
        elif any(keyword in input_lower for keyword in ["question", "what", "how", "why", "explain"]):
            return self.templates["qa"]
        elif any(keyword in input_lower for keyword in ["write", "story", "creative", "poem"]):
            return self.templates["creative_writing"]
        elif any(keyword in input_lower for keyword in ["code", "function", "class", "program"]):
            return self.templates["code_generation"]
        elif any(keyword in input_lower for keyword in ["analyze", "data", "insights", "trends"]):
            return self.templates["analysis"]
        else:
            # Default to QA template
            return self.templates["qa"]
    
    async def _generate_components(self, request: PromptRequest, 
                                 template: PromptTemplate) -> Dict[str, str]:
        """
        Generate prompt components using LLM or templates
        
        Args:
            request: Prompt generation request
            template: Selected template
            
        Returns:
            Dictionary of generated components
        """
        components = {}
        
        # Use provided values or generate from templates
        components["role"] = request.role or self._fill_template(
            template.role_template or "assistant", request.variables
        )
        components["task"] = request.task or self._fill_template(
            template.task_template or "", request.variables
        )
        components["context"] = request.context or self._fill_template(
            template.context_template or "", request.variables
        )
        components["reasoning"] = request.reasoning or self._fill_template(
            template.reasoning_template or "", request.variables
        )
        components["output_format"] = request.output_format or self._fill_template(
            template.output_format_template or "", request.variables
        )
        components["stop_condition"] = request.stop_condition or self._fill_template(
            template.stop_condition_template or "", request.variables
        )
        
        return components
    
    def _fill_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Fill template with variables
        
        Args:
            template: Template string with {variable} placeholders
            variables: Variables to substitute
            
        Returns:
            Filled template string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            # If variable is missing, use a default value
            missing_var = str(e).strip("'\"")
            logger.warning("Missing template variable", variable=missing_var)
            return template.replace(f"{{{missing_var}}}", f"[{missing_var}]")
    
    def _build_full_prompt(self, components: Dict[str, str], user_input: str) -> str:
        """
        Build the complete structured prompt
        
        Args:
            components: Generated prompt components
            user_input: Original user input
            
        Returns:
            Complete structured prompt
        """
        prompt_parts = [
            f"You are {components['role']}.",
            "",
            f"**Task:** {components['task']}",
            "",
            f"**Context:** {components['context']}",
            "",
            f"**Reasoning / Strategy:** {components['reasoning']}",
            "",
            f"**Output Format:** {components['output_format']}",
            "",
            f"**Stop Conditions:** {components['stop_condition']}",
            "",
            "---",
            "",
            user_input
        ]
        
        return "\n".join(prompt_parts)
    
    async def _get_cached_prompt(self, request: PromptRequest) -> Optional[GeneratedPrompt]:
        """
        Get cached prompt if available
        
        Args:
            request: Prompt generation request
            
        Returns:
            Cached prompt or None
        """
        if not self.cache_manager:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Get from cache
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            try:
                prompt = GeneratedPrompt.model_validate(cached_data)
                prompt.cached = True
                return prompt
            except Exception as e:
                logger.warning("Failed to deserialize cached prompt", error=str(e))
        
        return None
    
    async def _cache_prompt(self, request: PromptRequest, 
                          generated_prompt: GeneratedPrompt) -> None:
        """
        Cache generated prompt
        
        Args:
            request: Original request
            generated_prompt: Generated prompt to cache
        """
        if not self.cache_manager:
            return
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Cache the prompt
        await self.cache_manager.set(
            cache_key, 
            generated_prompt.model_dump(),
            ttl=self.cache_manager.config.prompt_ttl
        )
    
    def _generate_cache_key(self, request: PromptRequest) -> str:
        """
        Generate cache key for request
        
        Args:
            request: Prompt generation request
            
        Returns:
            Cache key
        """
        # Create a hash of the request for consistent key generation
        key_data = f"{request.user_input}_{request.template_id}_{request.role}_{request.task}_{request.context}_{request.reasoning}_{request.output_format}_{request.stop_condition}_{str(request.variables)}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"prompt_builder:generated:{key_hash}"
    
    async def generate_completion(self, request: PromptRequest, 
                                completion_request: Optional[CompletionRequest] = None) -> CompletionResponse:
        """
        Generate prompt and get completion from LLM
        
        Args:
            request: Prompt generation request
            completion_request: Optional completion request overrides
            
        Returns:
            LLM completion response
        """
        # Generate the structured prompt
        generated_prompt = await self.generate_prompt(request)
        
        # Create completion request
        if completion_request is None:
            completion_request = CompletionRequest(
                prompt=generated_prompt.full_prompt,
                model=request.model or "gpt-4o",
                temperature=0.7
            )
        else:
            # Update prompt in the completion request
            completion_request.prompt = generated_prompt.full_prompt
        
        # Get completion from LLM manager
        response = await self.llm_manager.generate_completion(
            completion_request,
            preferred_provider=request.provider,
            use_cache=request.use_cache
        )
        
        return response
    
    def add_template(self, template: PromptTemplate) -> None:
        """
        Add a new prompt template
        
        Args:
            template: Template to add
        """
        self.templates[template.id] = template
        logger.info("Added template", template_id=template.id)
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get template by ID
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """
        List available templates
        
        Args:
            category: Optional category filter
            
        Returns:
            List of templates
        """
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics if caching is enabled
        
        Returns:
            Cache statistics or None if caching is disabled
        """
        if self.cache_manager:
            return await self.cache_manager.get_cache_stats()
        return None
