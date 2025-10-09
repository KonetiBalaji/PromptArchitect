"""
Authentication Manager
Author: Balaji Koneti

Handles user authentication, JWT tokens, and user management for the API.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key-change-in-production"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


@dataclass
class User:
    """User data class"""
    id: str
    email: str
    username: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = None
    last_login: Optional[datetime] = None


class UserCreate(BaseModel):
    """User creation request model"""
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    """User login request model"""
    email: EmailStr
    password: str


class AuthManager:
    """Authentication manager for user management and JWT tokens"""
    
    def __init__(self):
        # In-memory user storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.user_by_email: Dict[str, User] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = User(
            id="admin-001",
            email="admin@promptarchitect.com",
            username="admin",
            hashed_password=pwd_context.hash("admin123"),
            is_active=True,
            is_admin=True,
            created_at=datetime.now()
        )
        self.users[admin_user.id] = admin_user
        self.user_by_email[admin_user.email] = admin_user
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        if user_data.email in self.user_by_email:
            raise ValueError("User with this email already exists")
        
        # Generate user ID
        user_id = f"user-{secrets.token_hex(8)}"
        
        # Hash password
        hashed_password = pwd_context.hash(user_data.password)
        
        # Create user
        user = User(
            id=user_id,
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=False,
            created_at=datetime.now()
        )
        
        # Store user
        self.users[user_id] = user
        self.user_by_email[user.email] = user
        
        return user
    
    async def authenticate_user(self, email: str, password: str) -> User:
        """Authenticate user with email and password"""
        user = self.user_by_email.get(email)
        if not user:
            raise ValueError("Invalid email or password")
        
        if not user.is_active:
            raise ValueError("User account is disabled")
        
        if not pwd_context.verify(password, user.hashed_password):
            raise ValueError("Invalid email or password")
        
        # Update last login
        user.last_login = datetime.now()
        
        return user
    
    async def create_token(self, user: User) -> str:
        """Create JWT token for user"""
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user.id,
            "email": user.email,
            "username": user.username,
            "is_admin": user.is_admin,
            "exp": expire
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> User:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise ValueError("Invalid token")
        except jwt.PyJWTError:
            raise ValueError("Invalid token")
        
        user = self.users.get(user_id)
        if user is None:
            raise ValueError("User not found")
        
        if not user.is_active:
            raise ValueError("User account is disabled")
        
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.user_by_email.get(email)
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ["username", "is_active", "is_admin"]
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Remove from both dictionaries
        del self.users[user_id]
        del self.user_by_email[user.email]
        
        return True
    
    async def list_users(self) -> list[User]:
        """List all users"""
        return list(self.users.values())
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.is_active)
        admin_users = sum(1 for user in self.users.values() if user.is_admin)
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users,
            "regular_users": total_users - admin_users
        }
