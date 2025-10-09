"""
Database Manager
Author: Balaji Koneti

Manages database connections, sessions, and operations for PostgreSQL and MongoDB.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import structlog

from .models import (
    Base, User, Completion, ReasoningSession, Template, TemplateSelection,
    UserAnalytics, SystemMetrics, ErrorLog
)

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Database manager for handling PostgreSQL and MongoDB connections"""
    
    def __init__(self, database_url: str, mongodb_url: Optional[str] = None):
        self.database_url = database_url
        self.mongodb_url = mongodb_url
        
        # PostgreSQL engine
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        # MongoDB client (placeholder for future implementation)
        self.mongodb_client = None
        
        logger.info("DatabaseManager initialized", database_url=database_url)
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Create PostgreSQL engines
            self.engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}  # For SQLite compatibility
            )
            
            # Create async engine if URL supports it
            if self.database_url.startswith("postgresql://"):
                async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
                self.async_engine = create_async_engine(async_url)
                self.async_session_factory = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create tables
            await self.create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def create_tables(self):
        """Create database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session"""
        if self.async_session_factory:
            async with self.async_session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
                finally:
                    await session.close()
        else:
            # Fallback to synchronous session
            session = self.session_factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
    
    # User operations
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        async with self.get_session() as session:
            user = User(**user_data)
            session.add(user)
            await session.flush()
            await session.refresh(user)
            return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        async with self.get_session() as session:
            return await session.get(User, user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email}
            )
            return result.fetchone()
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information"""
        async with self.get_session() as session:
            user = await session.get(User, user_id)
            if user:
                for key, value in kwargs.items():
                    setattr(user, key, value)
                await session.flush()
                await session.refresh(user)
            return user
    
    # Completion operations
    async def create_completion(self, completion_data: Dict[str, Any]) -> Completion:
        """Create a new completion record"""
        async with self.get_session() as session:
            completion = Completion(**completion_data)
            session.add(completion)
            await session.flush()
            await session.refresh(completion)
            return completion
    
    async def get_completions_by_user(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Completion]:
        """Get completions by user"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM completions WHERE user_id = :user_id ORDER BY created_at DESC LIMIT :limit OFFSET :offset"),
                {"user_id": user_id, "limit": limit, "offset": offset}
            )
            return result.fetchall()
    
    async def get_completion_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get completion statistics"""
        async with self.get_session() as session:
            if user_id:
                result = await session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_completions,
                            AVG(response_time) as avg_response_time,
                            AVG(confidence_score) as avg_confidence,
                            SUM((usage->>'total_tokens')::int) as total_tokens
                        FROM completions 
                        WHERE user_id = :user_id
                    """),
                    {"user_id": user_id}
                )
            else:
                result = await session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_completions,
                            AVG(response_time) as avg_response_time,
                            AVG(confidence_score) as avg_confidence,
                            SUM((usage->>'total_tokens')::int) as total_tokens
                        FROM completions
                    """)
                )
            return result.fetchone()._asdict()
    
    # Reasoning session operations
    async def create_reasoning_session(self, session_data: Dict[str, Any]) -> ReasoningSession:
        """Create a new reasoning session"""
        async with self.get_session() as session:
            reasoning_session = ReasoningSession(**session_data)
            session.add(reasoning_session)
            await session.flush()
            await session.refresh(reasoning_session)
            return reasoning_session
    
    async def get_reasoning_sessions_by_user(self, user_id: str, limit: int = 100) -> List[ReasoningSession]:
        """Get reasoning sessions by user"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM reasoning_sessions WHERE user_id = :user_id ORDER BY created_at DESC LIMIT :limit"),
                {"user_id": user_id, "limit": limit}
            )
            return result.fetchall()
    
    # Template operations
    async def create_template(self, template_data: Dict[str, Any]) -> Template:
        """Create a new template"""
        async with self.get_session() as session:
            template = Template(**template_data)
            session.add(template)
            await session.flush()
            await session.refresh(template)
            return template
    
    async def get_templates_by_category(self, category: str, limit: int = 100) -> List[Template]:
        """Get templates by category"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM templates WHERE category = :category AND is_public = true ORDER BY usage_count DESC LIMIT :limit"),
                {"category": category, "limit": limit}
            )
            return result.fetchall()
    
    async def get_all_templates(self, limit: int = 1000) -> List[Template]:
        """Get all public templates"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM templates WHERE is_public = true ORDER BY usage_count DESC LIMIT :limit"),
                {"limit": limit}
            )
            return result.fetchall()
    
    async def update_template_usage(self, template_id: str):
        """Update template usage count"""
        async with self.get_session() as session:
            await session.execute(
                text("UPDATE templates SET usage_count = usage_count + 1 WHERE id = :template_id"),
                {"template_id": template_id}
            )
    
    # Analytics operations
    async def create_user_analytics(self, analytics_data: Dict[str, Any]) -> UserAnalytics:
        """Create user analytics record"""
        async with self.get_session() as session:
            analytics = UserAnalytics(**analytics_data)
            session.add(analytics)
            await session.flush()
            await session.refresh(analytics)
            return analytics
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> List[UserAnalytics]:
        """Get user analytics for specified days"""
        async with self.get_session() as session:
            start_date = datetime.now() - timedelta(days=days)
            result = await session.execute(
                text("SELECT * FROM user_analytics WHERE user_id = :user_id AND date >= :start_date ORDER BY date DESC"),
                {"user_id": user_id, "start_date": start_date}
            )
            return result.fetchall()
    
    async def get_system_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get system-wide analytics"""
        async with self.get_session() as session:
            start_date = datetime.now() - timedelta(days=days)
            
            # Get completion stats
            completion_stats = await session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_completions,
                        AVG(response_time) as avg_response_time,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM completions 
                    WHERE created_at >= :start_date
                """),
                {"start_date": start_date}
            )
            
            # Get reasoning stats
            reasoning_stats = await session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        AVG(total_time) as avg_reasoning_time,
                        AVG(confidence) as avg_confidence,
                        strategy,
                        COUNT(*) as strategy_count
                    FROM reasoning_sessions 
                    WHERE created_at >= :start_date
                    GROUP BY strategy
                """),
                {"start_date": start_date}
            )
            
            # Get template stats
            template_stats = await session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_templates,
                        COUNT(DISTINCT category) as categories_count,
                        SUM(usage_count) as total_usage,
                        AVG(rating) as avg_rating
                    FROM templates 
                    WHERE created_at >= :start_date
                """),
                {"start_date": start_date}
            )
            
            return {
                "completion_stats": completion_stats.fetchone()._asdict(),
                "reasoning_stats": [row._asdict() for row in reasoning_stats.fetchall()],
                "template_stats": template_stats.fetchone()._asdict()
            }
    
    # System metrics operations
    async def create_system_metric(self, metric_data: Dict[str, Any]) -> SystemMetrics:
        """Create system metric record"""
        async with self.get_session() as session:
            metric = SystemMetrics(**metric_data)
            session.add(metric)
            await session.flush()
            await session.refresh(metric)
            return metric
    
    async def get_system_metrics(self, metric_type: str, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics for specified hours"""
        async with self.get_session() as session:
            start_time = datetime.now() - timedelta(hours=hours)
            result = await session.execute(
                text("SELECT * FROM system_metrics WHERE metric_type = :metric_type AND timestamp >= :start_time ORDER BY timestamp DESC"),
                {"metric_type": metric_type, "start_time": start_time}
            )
            return result.fetchall()
    
    # Error logging operations
    async def log_error(self, error_data: Dict[str, Any]) -> ErrorLog:
        """Log system error"""
        async with self.get_session() as session:
            error_log = ErrorLog(**error_data)
            session.add(error_log)
            await session.flush()
            await session.refresh(error_log)
            return error_log
    
    async def get_error_logs(self, severity: Optional[str] = None, hours: int = 24) -> List[ErrorLog]:
        """Get error logs"""
        async with self.get_session() as session:
            start_time = datetime.now() - timedelta(hours=hours)
            
            if severity:
                result = await session.execute(
                    text("SELECT * FROM error_logs WHERE severity = :severity AND timestamp >= :start_time ORDER BY timestamp DESC"),
                    {"severity": severity, "start_time": start_time}
                )
            else:
                result = await session.execute(
                    text("SELECT * FROM error_logs WHERE timestamp >= :start_time ORDER BY timestamp DESC"),
                    {"start_time": start_time}
                )
            return result.fetchall()
    
    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return {
                    "status": "healthy",
                    "database": "connected",
                    "timestamp": datetime.now()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")
