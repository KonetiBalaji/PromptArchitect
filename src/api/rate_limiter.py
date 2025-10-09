"""
Rate Limiter
Author: Balaji Koneti

Implements rate limiting for API endpoints to prevent abuse and ensure fair usage.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in a short burst


@dataclass
class RateLimitStatus:
    """Rate limit status for a user"""
    requests_minute: deque
    requests_hour: deque
    requests_day: deque
    last_request: float
    blocked_until: Optional[float] = None


class RateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self):
        # User rate limit tracking
        self.user_limits: Dict[str, RateLimitStatus] = defaultdict(
            lambda: RateLimitStatus(
                requests_minute=deque(),
                requests_hour=deque(),
                requests_day=deque(),
                last_request=0.0
            )
        )
        
        # Default rate limits
        self.default_limits = RateLimit()
        
        # Endpoint-specific rate limits
        self.endpoint_limits: Dict[str, RateLimit] = {
            "/completion": RateLimit(requests_per_minute=30, requests_per_hour=500),
            "/reasoning": RateLimit(requests_per_minute=20, requests_per_hour=300),
            "/templates/generate": RateLimit(requests_per_minute=10, requests_per_hour=100),
            "/templates/select": RateLimit(requests_per_minute=60, requests_per_hour=1000),
            "/analytics": RateLimit(requests_per_minute=10, requests_per_hour=100)
        }
    
    async def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """
        Check if user is within rate limits for the endpoint
        
        Args:
            user_id: User identifier
            endpoint: API endpoint path
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Get rate limits for endpoint
        limits = self.endpoint_limits.get(endpoint, self.default_limits)
        
        # Get user's rate limit status
        user_status = self.user_limits[user_id]
        
        # Check if user is temporarily blocked
        if user_status.blocked_until and current_time < user_status.blocked_until:
            return False
        
        # Clean old requests from tracking
        self._clean_old_requests(user_status, current_time)
        
        # Check rate limits
        if not self._check_limits(user_status, limits, current_time):
            # Block user temporarily (exponential backoff)
            block_duration = min(60, 2 ** len(user_status.requests_minute))
            user_status.blocked_until = current_time + block_duration
            return False
        
        # Record this request
        user_status.requests_minute.append(current_time)
        user_status.requests_hour.append(current_time)
        user_status.requests_day.append(current_time)
        user_status.last_request = current_time
        
        return True
    
    def _clean_old_requests(self, user_status: RateLimitStatus, current_time: float):
        """Clean old requests from tracking queues"""
        # Clean minute queue (older than 60 seconds)
        while user_status.requests_minute and current_time - user_status.requests_minute[0] > 60:
            user_status.requests_minute.popleft()
        
        # Clean hour queue (older than 3600 seconds)
        while user_status.requests_hour and current_time - user_status.requests_hour[0] > 3600:
            user_status.requests_hour.popleft()
        
        # Clean day queue (older than 86400 seconds)
        while user_status.requests_day and current_time - user_status.requests_day[0] > 86400:
            user_status.requests_day.popleft()
    
    def _check_limits(self, user_status: RateLimitStatus, limits: RateLimit, current_time: float) -> bool:
        """Check if user is within all rate limits"""
        # Check minute limit
        if len(user_status.requests_minute) >= limits.requests_per_minute:
            return False
        
        # Check hour limit
        if len(user_status.requests_hour) >= limits.requests_per_hour:
            return False
        
        # Check day limit
        if len(user_status.requests_day) >= limits.requests_per_day:
            return False
        
        # Check burst limit (requests in last 10 seconds)
        recent_requests = sum(1 for req_time in user_status.requests_minute 
                            if current_time - req_time <= 10)
        if recent_requests >= limits.burst_limit:
            return False
        
        return True
    
    def get_user_rate_limit_status(self, user_id: str) -> Dict[str, any]:
        """Get current rate limit status for a user"""
        if user_id not in self.user_limits:
            return {
                "requests_minute": 0,
                "requests_hour": 0,
                "requests_day": 0,
                "blocked_until": None,
                "last_request": None
            }
        
        user_status = self.user_limits[user_id]
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(user_status, current_time)
        
        return {
            "requests_minute": len(user_status.requests_minute),
            "requests_hour": len(user_status.requests_hour),
            "requests_day": len(user_status.requests_day),
            "blocked_until": user_status.blocked_until,
            "last_request": user_status.last_request
        }
    
    def reset_user_rate_limit(self, user_id: str):
        """Reset rate limit for a specific user"""
        if user_id in self.user_limits:
            del self.user_limits[user_id]
    
    def set_endpoint_rate_limit(self, endpoint: str, limits: RateLimit):
        """Set custom rate limits for an endpoint"""
        self.endpoint_limits[endpoint] = limits
    
    def get_rate_limit_stats(self) -> Dict[str, any]:
        """Get overall rate limiting statistics"""
        current_time = time.time()
        active_users = 0
        blocked_users = 0
        total_requests = 0
        
        for user_status in self.user_limits.values():
            self._clean_old_requests(user_status, current_time)
            
            if user_status.last_request > current_time - 3600:  # Active in last hour
                active_users += 1
            
            if user_status.blocked_until and user_status.blocked_until > current_time:
                blocked_users += 1
            
            total_requests += len(user_status.requests_day)
        
        return {
            "active_users": active_users,
            "blocked_users": blocked_users,
            "total_requests_today": total_requests,
            "endpoint_limits": {
                endpoint: {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "requests_per_day": limits.requests_per_day
                }
                for endpoint, limits in self.endpoint_limits.items()
            }
        }
