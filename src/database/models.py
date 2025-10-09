"""
Database Models
Author: Balaji Koneti

SQLAlchemy models for data persistence with PostgreSQL and MongoDB support.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel
import uuid

Base = declarative_base()


# Enums
class ProviderType(str, Enum):
    """LLM Provider types"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ReasoningStrategy(str, Enum):
    """Reasoning strategy types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    TREE = "tree"
    REFLECTION = "reflection"


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
    MARKETING = "marketing"
    CUSTOM = "custom"


# SQLAlchemy Models
class User(Base):
    """User model for authentication and user management"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    profile_data = Column(JSON, nullable=True)
    
    # Relationships
    completions = relationship("Completion", back_populates="user")
    reasoning_sessions = relationship("ReasoningSession", back_populates="user")
    templates = relationship("Template", back_populates="creator")
    analytics = relationship("UserAnalytics", back_populates="user")


class Completion(Base):
    """Completion model for storing LLM completions"""
    __tablename__ = "completions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    model = Column(String(100), nullable=False, index=True)
    provider = Column(String(50), nullable=False, index=True)
    response = Column(Text, nullable=False)
    usage = Column(JSON, nullable=True)  # Token usage statistics
    response_time = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    reasoning_steps = Column(JSON, nullable=True)
    thinking_blocks = Column(JSON, nullable=True)
    reasoning_trace = Column(Text, nullable=True)
    verification_results = Column(JSON, nullable=True)
    grounded_sources = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="completions")


class ReasoningSession(Base):
    """Reasoning session model for storing chain-of-thought reasoning"""
    __tablename__ = "reasoning_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    model = Column(String(100), nullable=False, index=True)
    provider = Column(String(50), nullable=False, index=True)
    steps = Column(JSON, nullable=False)  # Reasoning steps
    final_answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    total_time = Column(Float, nullable=True)
    verification_passed = Column(Boolean, default=False, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="reasoning_sessions")


class Template(Base):
    """Template model for storing AI-generated templates"""
    __tablename__ = "templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, index=True)
    complexity = Column(String(20), nullable=False, index=True)
    template_content = Column(Text, nullable=False)
    variables = Column(JSON, nullable=True)  # Template variables
    reasoning_integration = Column(Boolean, default=False, nullable=False)
    use_case = Column(Text, nullable=True)
    requirements = Column(JSON, nullable=True)
    constraints = Column(JSON, nullable=True)
    target_audience = Column(String(200), nullable=True)
    language = Column(String(10), default="en", nullable=False)
    version = Column(String(20), default="1.0", nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    rating = Column(Float, nullable=True)
    generation_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User", back_populates="templates")
    selections = relationship("TemplateSelection", back_populates="template")


class TemplateSelection(Base):
    """Template selection model for tracking template usage"""
    __tablename__ = "template_selections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    template_id = Column(UUID(as_uuid=True), ForeignKey("templates.id"), nullable=False, index=True)
    user_input = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    detected_intent = Column(String(200), nullable=True)
    reasoning = Column(Text, nullable=True)
    context_analysis = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    template = relationship("Template", back_populates="selections")


class UserAnalytics(Base):
    """User analytics model for tracking user behavior and metrics"""
    __tablename__ = "user_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    completions_count = Column(Integer, default=0, nullable=False)
    reasoning_sessions_count = Column(Integer, default=0, nullable=False)
    templates_generated_count = Column(Integer, default=0, nullable=False)
    templates_selected_count = Column(Integer, default=0, nullable=False)
    total_tokens_used = Column(Integer, default=0, nullable=False)
    total_cost = Column(Float, default=0.0, nullable=False)
    avg_response_time = Column(Float, nullable=True)
    avg_confidence_score = Column(Float, nullable=True)
    preferred_models = Column(JSON, nullable=True)
    preferred_providers = Column(JSON, nullable=True)
    preferred_strategies = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="analytics")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_analytics_user_date', 'user_id', 'date'),
    )


class SystemMetrics(Base):
    """System metrics model for tracking system performance"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    tags = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)


class ErrorLog(Base):
    """Error log model for tracking system errors"""
    __tablename__ = "error_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    error_type = Column(String(100), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    endpoint = Column(String(200), nullable=True, index=True)
    request_data = Column(JSON, nullable=True)
    severity = Column(String(20), default="error", nullable=False, index=True)
    resolved = Column(Boolean, default=False, nullable=False)
    metadata = Column(JSON, nullable=True)


# Pydantic Models for API
class UserCreate(BaseModel):
    """User creation model"""
    email: str
    username: str
    password: str


class UserResponse(BaseModel):
    """User response model"""
    id: str
    email: str
    username: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class CompletionCreate(BaseModel):
    """Completion creation model"""
    prompt: str
    model: str
    provider: str
    response: str
    usage: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    confidence_score: Optional[float] = None
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    thinking_blocks: Optional[List[str]] = None
    reasoning_trace: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None
    grounded_sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Completion response model"""
    id: str
    user_id: str
    prompt: str
    model: str
    provider: str
    response: str
    usage: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    confidence_score: Optional[float] = None
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    thinking_blocks: Optional[List[str]] = None
    reasoning_trace: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None
    grounded_sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TemplateCreate(BaseModel):
    """Template creation model"""
    name: str
    description: Optional[str] = None
    category: str
    complexity: str
    template_content: str
    variables: Optional[List[str]] = None
    reasoning_integration: bool = False
    use_case: Optional[str] = None
    requirements: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    target_audience: Optional[str] = None
    language: str = "en"
    is_public: bool = True
    generation_metadata: Optional[Dict[str, Any]] = None


class TemplateResponse(BaseModel):
    """Template response model"""
    id: str
    creator_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    category: str
    complexity: str
    template_content: str
    variables: Optional[List[str]] = None
    reasoning_integration: bool
    use_case: Optional[str] = None
    requirements: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    target_audience: Optional[str] = None
    language: str
    version: str
    is_public: bool
    usage_count: int
    rating: Optional[float] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    """Analytics response model"""
    user_id: str
    date: datetime
    completions_count: int
    reasoning_sessions_count: int
    templates_generated_count: int
    templates_selected_count: int
    total_tokens_used: int
    total_cost: float
    avg_response_time: Optional[float] = None
    avg_confidence_score: Optional[float] = None
    preferred_models: Optional[Dict[str, Any]] = None
    preferred_providers: Optional[Dict[str, Any]] = None
    preferred_strategies: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True
