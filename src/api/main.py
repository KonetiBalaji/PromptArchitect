"""
FastAPI REST API Server
Author: Balaji Koneti

Comprehensive REST API for the Visionary Prompt Architect with authentication,
rate limiting, and full feature support including reasoning models and templates.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
import uvicorn

from ..llm_manager import LLMManager
from ..reasoning_manager import ReasoningManager, ReasoningConfig, ReasoningStrategy
from ..chain_of_thought.sequential_chain import SequentialChain, SequentialChainConfig
from ..chain_of_thought.parallel_chain import ParallelChain, ParallelChainConfig
from ..chain_of_thought.tree_reasoning import TreeReasoning, TreeReasoningConfig
from ..chain_of_thought.reflection import Reflection, ReflectionConfig
from ..templates.template_generator import TemplateGenerator, TemplateGenerationRequest, TemplateCategory, TemplateComplexity
from ..templates.template_registry import template_registry
from ..templates.dynamic_selector import DynamicSelector
from ..llm_providers.base import CompletionRequest, ProviderType
from .auth import AuthManager, User, UserCreate, UserLogin
from .rate_limiter import RateLimiter
from .models import (
    CompletionRequestModel, CompletionResponseModel,
    ReasoningRequestModel, ReasoningResponseModel,
    TemplateGenerationRequestModel, TemplateResponseModel,
    TemplateSelectionRequestModel, TemplateSelectionResponseModel,
    AnalyticsResponseModel, HealthResponseModel
)

logger = structlog.get_logger(__name__)

# Global components
llm_manager: Optional[LLMManager] = None
reasoning_manager: Optional[ReasoningManager] = None
template_generator: Optional[TemplateGenerator] = None
dynamic_selector: Optional[DynamicSelector] = None
auth_manager: Optional[AuthManager] = None
rate_limiter: Optional[RateLimiter] = None
chain_implementations: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llm_manager, reasoning_manager, template_generator, dynamic_selector, auth_manager, rate_limiter, chain_implementations
    
    # Startup
    logger.info("Starting Visionary Prompt Architect API...")
    
    try:
        # Initialize LLM Manager
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        # Initialize Reasoning Manager
        reasoning_manager = ReasoningManager(llm_manager)
        
        # Initialize Template Generator
        template_generator = TemplateGenerator(llm_manager)
        
        # Initialize Dynamic Selector
        dynamic_selector = DynamicSelector(llm_manager)
        
        # Initialize Chain-of-Thought implementations
        chain_implementations = {
            "sequential": SequentialChain(llm_manager),
            "parallel": ParallelChain(llm_manager),
            "tree": TreeReasoning(llm_manager),
            "reflection": Reflection(llm_manager)
        }
        
        # Initialize Auth Manager
        auth_manager = AuthManager()
        
        # Initialize Rate Limiter
        rate_limiter = RateLimiter()
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error("Failed to initialize components", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Visionary Prompt Architect API...")


# Create FastAPI app
app = FastAPI(
    title="Visionary Prompt Architect API",
    description="Advanced AI-powered prompt engineering with reasoning models and dynamic templates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Security
security = HTTPBearer()


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Auth manager not initialized")
    
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_rate_limit(request: Request, user: User = Depends(get_current_user)):
    """Check rate limit for user"""
    if not rate_limiter:
        raise HTTPException(status_code=500, detail="Rate limiter not initialized")
    
    if not await rate_limiter.check_rate_limit(user.id, request.url.path):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# Health check endpoint
@app.get("/health", response_model=HealthResponseModel)
async def health_check():
    """Health check endpoint"""
    return HealthResponseModel(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        components={
            "llm_manager": llm_manager is not None,
            "reasoning_manager": reasoning_manager is not None,
            "template_generator": template_generator is not None,
            "auth_manager": auth_manager is not None,
            "rate_limiter": rate_limiter is not None
        }
    )


# Authentication endpoints
@app.post("/auth/register", response_model=Dict[str, str])
async def register(user_data: UserCreate):
    """Register a new user"""
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Auth manager not initialized")
    
    try:
        user = await auth_manager.create_user(user_data)
        token = await auth_manager.create_token(user)
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=Dict[str, str])
async def login(credentials: UserLogin):
    """Login user"""
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Auth manager not initialized")
    
    try:
        user = await auth_manager.authenticate_user(credentials.email, credentials.password)
        token = await auth_manager.create_token(user)
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")


# Completion endpoints
@app.post("/completion", response_model=CompletionResponseModel)
async def create_completion(
    request: CompletionRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Generate completion using LLM"""
    if not llm_manager:
        raise HTTPException(status_code=500, detail="LLM manager not initialized")
    
    try:
        # Convert request to internal format
        completion_request = CompletionRequest(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            stream=request.stream,
            system_message=request.system_message,
            reasoning_effort=request.reasoning_effort,
            max_completion_tokens=request.max_completion_tokens,
            extended_thinking=request.extended_thinking,
            thinking_mode=request.thinking_mode,
            grounding=request.grounding,
            search_web=request.search_web
        )
        
        # Generate completion
        response = await llm_manager.generate_completion(
            completion_request,
            preferred_provider=request.preferred_provider
        )
        
        # Convert response to API format
        return CompletionResponseModel(
            content=response.content,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
            finish_reason=response.finish_reason,
            response_time=response.response_time,
            timestamp=response.timestamp,
            reasoning_steps=response.reasoning_steps,
            thinking_blocks=response.thinking_blocks,
            reasoning_trace=response.reasoning_trace,
            confidence_score=response.confidence_score,
            verification_results=response.verification_results,
            grounded_sources=response.grounded_sources,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error("Completion generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Reasoning endpoints
@app.post("/reasoning", response_model=ReasoningResponseModel)
async def create_reasoning(
    request: ReasoningRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Generate reasoning using advanced chain-of-thought"""
    if not reasoning_manager:
        raise HTTPException(status_code=500, detail="Reasoning manager not initialized")
    
    try:
        # Convert request to internal format
        completion_request = CompletionRequest(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
            extended_thinking=request.extended_thinking,
            thinking_mode=request.thinking_mode
        )
        
        # Create reasoning config
        config = ReasoningConfig(
            strategy=ReasoningStrategy(request.strategy),
            max_steps=request.max_steps,
            verification_enabled=request.verification_enabled,
            confidence_threshold=request.confidence_threshold,
            timeout=request.timeout,
            parallel_workers=request.parallel_workers
        )
        
        # Generate reasoning
        result = await reasoning_manager.reason(completion_request, config)
        
        # Convert result to API format
        return ReasoningResponseModel(
            strategy=result.strategy,
            steps=result.steps,
            final_answer=result.final_answer,
            confidence=result.confidence,
            total_time=result.total_time,
            provider_used=result.provider_used,
            model_used=result.model_used,
            verification_passed=result.verification_passed,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error("Reasoning generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Chain-of-thought endpoints
@app.post("/reasoning/sequential", response_model=ReasoningResponseModel)
async def sequential_reasoning(
    request: ReasoningRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Sequential chain-of-thought reasoning"""
    if "sequential" not in chain_implementations:
        raise HTTPException(status_code=500, detail="Sequential chain not initialized")
    
    try:
        completion_request = CompletionRequest(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        config = SequentialChainConfig(
            max_steps=request.max_steps,
            verification_enabled=request.verification_enabled,
            confidence_threshold=request.confidence_threshold
        )
        
        result = await chain_implementations["sequential"].reason(completion_request, config)
        
        return ReasoningResponseModel(
            strategy="sequential",
            steps=result.steps,
            final_answer=result.final_answer,
            confidence=result.overall_confidence,
            total_time=result.total_time,
            provider_used=result.provider_used,
            model_used=result.model_used,
            verification_passed=result.verification_passed,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error("Sequential reasoning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reasoning/parallel", response_model=ReasoningResponseModel)
async def parallel_reasoning(
    request: ReasoningRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Parallel chain-of-thought reasoning"""
    if "parallel" not in chain_implementations:
        raise HTTPException(status_code=500, detail="Parallel chain not initialized")
    
    try:
        completion_request = CompletionRequest(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        config = ParallelChainConfig(
            num_paths=request.parallel_workers or 3,
            consensus_threshold=request.confidence_threshold
        )
        
        result = await chain_implementations["parallel"].reason(completion_request, config)
        
        return ReasoningResponseModel(
            strategy="parallel",
            steps=result.paths,
            final_answer=result.final_answer,
            confidence=result.consensus.agreement_level,
            total_time=result.total_time,
            provider_used=result.provider_used,
            model_used=result.model_used,
            verification_passed=True,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error("Parallel reasoning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Template endpoints
@app.post("/templates/generate", response_model=TemplateResponseModel)
async def generate_template(
    request: TemplateGenerationRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Generate AI-powered template"""
    if not template_generator:
        raise HTTPException(status_code=500, detail="Template generator not initialized")
    
    try:
        # Convert request to internal format
        generation_request = TemplateGenerationRequest(
            category=TemplateCategory(request.category),
            complexity=TemplateComplexity(request.complexity),
            use_case=request.use_case,
            requirements=request.requirements,
            constraints=request.constraints,
            target_audience=request.target_audience,
            language=request.language,
            reasoning_enabled=request.reasoning_enabled,
            custom_instructions=request.custom_instructions
        )
        
        # Generate template
        template = await template_generator.generate_template(generation_request)
        
        # Convert to API format
        return TemplateResponseModel(
            template_id=template.template_id,
            name=template.name,
            description=template.description,
            category=template.category,
            complexity=template.complexity,
            template_content=template.template_content,
            variables=template.variables,
            reasoning_integration=template.reasoning_integration,
            created_at=template.created_at,
            version=template.version,
            metadata=template.generation_metadata
        )
        
    except Exception as e:
        logger.error("Template generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/templates/select", response_model=TemplateSelectionResponseModel)
async def select_template(
    request: TemplateSelectionRequestModel,
    user: User = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """Select best template using dynamic selector"""
    if not dynamic_selector:
        raise HTTPException(status_code=500, detail="Dynamic selector not initialized")
    
    try:
        # Get available templates
        available_templates = template_registry.get_all_templates()
        
        # Select template
        result = await dynamic_selector.select_template(
            request.user_input,
            available_templates,
            request.context
        )
        
        return TemplateSelectionResponseModel(
            template_id=result.template_id,
            confidence=result.confidence,
            detected_intent=result.detected_intent,
            reasoning=result.reasoning,
            context_analysis=result.context_analysis,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error("Template selection failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates", response_model=List[TemplateResponseModel])
async def list_templates(
    category: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """List available templates"""
    try:
        if category:
            templates = template_registry.get_templates_by_category(category)
        else:
            templates = template_registry.get_all_templates()
        
        return [
            TemplateResponseModel(
                template_id=template.id,
                name=template.name,
                description=template.description,
                category=template.category,
                complexity="moderate",  # Default
                template_content=template.template,
                variables=template.variables,
                reasoning_integration=template.reasoning_enabled,
                created_at=datetime.now(),
                version="1.0",
                metadata={}
            )
            for template in templates
        ]
        
    except Exception as e:
        logger.error("Template listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.get("/analytics", response_model=AnalyticsResponseModel)
async def get_analytics(
    user: User = Depends(get_current_user)
):
    """Get system analytics"""
    try:
        # Get reasoning stats
        reasoning_stats = reasoning_manager.get_reasoning_stats() if reasoning_manager else {}
        
        # Get template stats
        template_stats = template_generator.get_generation_stats() if template_generator else {}
        
        # Get template registry analytics
        registry_analytics = template_registry.get_template_analytics()
        
        # Get provider status
        provider_status = {}
        if llm_manager:
            for provider_type, provider_info in llm_manager.providers.items():
                provider_status[provider_type.value] = {
                    "status": provider_info.status.value,
                    "models": len(provider_info.available_models),
                    "response_time": provider_info.avg_response_time,
                    "success_rate": provider_info.success_rate
                }
        
        return AnalyticsResponseModel(
            reasoning_stats=reasoning_stats,
            template_stats=template_stats,
            registry_analytics=registry_analytics,
            provider_status=provider_status,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error("Analytics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
