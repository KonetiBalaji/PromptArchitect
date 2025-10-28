"""
Template Registry
Author: Balaji Koneti

Central registry for all specialized templates with categorization,
search functionality, and template management capabilities.
"""

from typing import Dict, List, Optional, Any
from .dynamic_selector import PromptTemplate
from .specialized.research_templates import RESEARCH_TEMPLATES
from .specialized.legal_templates import LEGAL_TEMPLATES
from .specialized.medical_templates import MEDICAL_TEMPLATES
from .specialized.financial_templates import FINANCIAL_TEMPLATES
from .specialized.creative_templates import CREATIVE_TEMPLATES
try:
    from .external.chatgpt_prompts import load_chatgpt_prompts
except Exception:  # pragma: no cover - optional external loader
    load_chatgpt_prompts = None  # type: ignore


class TemplateRegistry:
    """
    Central registry for all prompt templates
    
    Features:
    - Template categorization and organization
    - Search and filtering capabilities
    - Template metadata management
    - Usage statistics and analytics
    - Template versioning and updates
    """
    
    def __init__(self):
        """Initialize template registry with all specialized templates"""
        self.templates: Dict[str, PromptTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        self.tags: Dict[str, List[str]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load all specialized templates
        self._load_specialized_templates()
        
        # Initialize usage statistics
        self._initialize_usage_stats()
    
    def _load_specialized_templates(self):
        """Load all specialized templates into the registry"""
        # Research templates
        for template_id, template in RESEARCH_TEMPLATES.items():
            self.templates[template_id] = template
            self._add_to_category("research", template_id)
            self._add_tags(template_id, ["research", "analysis", "academic", "scientific"])
        
        # Legal templates
        for template_id, template in LEGAL_TEMPLATES.items():
            self.templates[template_id] = template
            self._add_to_category("legal", template_id)
            self._add_tags(template_id, ["legal", "compliance", "contracts", "analysis"])
        
        # Medical templates
        for template_id, template in MEDICAL_TEMPLATES.items():
            self.templates[template_id] = template
            self._add_to_category("medical", template_id)
            self._add_tags(template_id, ["medical", "clinical", "healthcare", "diagnosis"])
        
        # Financial templates
        for template_id, template in FINANCIAL_TEMPLATES.items():
            self.templates[template_id] = template
            self._add_to_category("financial", template_id)
            self._add_tags(template_id, ["financial", "investment", "analysis", "risk"])
        
        # Creative templates
        for template_id, template in CREATIVE_TEMPLATES.items():
            self.templates[template_id] = template
            self._add_to_category("creative", template_id)
            self._add_tags(template_id, ["creative", "content", "marketing", "writing"])

        # External community prompts (optional)
        if load_chatgpt_prompts:
            for tpl in load_chatgpt_prompts():
                # Avoid collisions
                if tpl.id in self.templates:
                    continue
                self.templates[tpl.id] = tpl
                self._add_to_category(tpl.category, tpl.id)
                self._add_tags(tpl.id, list(set((tpl.tags or []) + ["community"])) )
    
    def _add_to_category(self, category: str, template_id: str):
        """Add template to category"""
        if category not in self.categories:
            self.categories[category] = []
        if template_id not in self.categories[category]:
            self.categories[category].append(template_id)
    
    def _add_tags(self, template_id: str, tags: List[str]):
        """Add tags to template"""
        self.tags[template_id] = tags
    
    def _initialize_usage_stats(self):
        """Initialize usage statistics for all templates"""
        for template_id in self.templates:
            self.usage_stats[template_id] = {
                "usage_count": 0,
                "last_used": None,
                "success_rate": 0.0,
                "average_quality_score": 0.0,
                "user_ratings": [],
                "common_use_cases": [],
                "performance_metrics": {}
            }
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get template by ID
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[PromptTemplate]:
        """
        Get templates by category
        
        Args:
            category: Template category
            
        Returns:
            List of templates in category
        """
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_templates_by_tags(self, tags: List[str]) -> List[PromptTemplate]:
        """
        Get templates by tags
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of templates matching tags
        """
        matching_templates = []
        
        for template_id, template in self.templates.items():
            template_tags = self.tags.get(template_id, [])
            if any(tag in template_tags for tag in tags):
                matching_templates.append(template)
        
        return matching_templates
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """
        Search templates by name, description, or content
        
        Args:
            query: Search query
            
        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        matching_templates = []
        
        for template in self.templates.values():
            # Search in name, description, and template content (raw or sectioned)
            content_blob = ""
            # Prefer raw template if present
            if getattr(template, "template", None):
                content_blob = getattr(template, "template", "") or ""
            else:
                # Concatenate sectioned fields defensively
                parts = [
                    getattr(template, "role_template", "") or "",
                    getattr(template, "task_template", "") or "",
                    getattr(template, "context_template", "") or "",
                    getattr(template, "reasoning_template", "") or "",
                    getattr(template, "output_format_template", "") or "",
                    getattr(template, "stop_condition_template", "") or "",
                ]
                content_blob = "\n".join(parts)
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                query_lower in content_blob.lower()):
                matching_templates.append(template)
        
        return matching_templates
    
    def get_all_templates(self) -> List[PromptTemplate]:
        """
        Get all templates
        
        Returns:
            List of all templates
        """
        return list(self.templates.values())
    
    def get_categories(self) -> List[str]:
        """
        Get all available categories
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())
    
    def get_template_count(self) -> int:
        """
        Get total number of templates
        
        Returns:
            Total template count
        """
        return len(self.templates)
    
    def get_category_count(self) -> int:
        """
        Get number of categories
        
        Returns:
            Number of categories
        """
        return len(self.categories)
    
    def update_usage_stats(self, template_id: str, usage_data: Dict[str, Any]):
        """
        Update usage statistics for a template
        
        Args:
            template_id: Template identifier
            usage_data: Usage data to update
        """
        if template_id in self.usage_stats:
            stats = self.usage_stats[template_id]
            
            # Update usage count
            if "usage_count" in usage_data:
                stats["usage_count"] += usage_data["usage_count"]
            
            # Update last used timestamp
            if "last_used" in usage_data:
                stats["last_used"] = usage_data["last_used"]
            
            # Update success rate
            if "success_rate" in usage_data:
                stats["success_rate"] = usage_data["success_rate"]
            
            # Update quality score
            if "quality_score" in usage_data:
                stats["average_quality_score"] = usage_data["quality_score"]
            
            # Add user rating
            if "user_rating" in usage_data:
                stats["user_ratings"].append(usage_data["user_rating"])
            
            # Add use case
            if "use_case" in usage_data:
                use_case = usage_data["use_case"]
                if use_case not in stats["common_use_cases"]:
                    stats["common_use_cases"].append(use_case)
            
            # Update performance metrics
            if "performance_metrics" in usage_data:
                stats["performance_metrics"].update(usage_data["performance_metrics"])
    
    def get_usage_stats(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for a template
        
        Args:
            template_id: Template identifier
            
        Returns:
            Usage statistics or None if not found
        """
        return self.usage_stats.get(template_id)
    
    def get_popular_templates(self, limit: int = 10) -> List[PromptTemplate]:
        """
        Get most popular templates by usage count
        
        Args:
            limit: Maximum number of templates to return
            
        Returns:
            List of popular templates
        """
        # Sort templates by usage count
        sorted_templates = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]["usage_count"],
            reverse=True
        )
        
        # Get template objects
        popular_templates = []
        for template_id, _ in sorted_templates[:limit]:
            if template_id in self.templates:
                popular_templates.append(self.templates[template_id])
        
        return popular_templates
    
    def get_high_quality_templates(self, min_quality_score: float = 0.8) -> List[PromptTemplate]:
        """
        Get templates with high quality scores
        
        Args:
            min_quality_score: Minimum quality score threshold
            
        Returns:
            List of high-quality templates
        """
        high_quality_templates = []
        
        for template_id, template in self.templates.items():
            stats = self.usage_stats.get(template_id, {})
            quality_score = stats.get("average_quality_score", 0.0)
            
            if quality_score >= min_quality_score:
                high_quality_templates.append(template)
        
        return high_quality_templates
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive template analytics
        
        Returns:
            Dictionary with template analytics
        """
        total_templates = len(self.templates)
        total_usage = sum(stats["usage_count"] for stats in self.usage_stats.values())
        
        # Category distribution
        category_distribution = {}
        for category, template_ids in self.categories.items():
            category_usage = sum(
                self.usage_stats.get(tid, {}).get("usage_count", 0)
                for tid in template_ids
            )
            category_distribution[category] = {
                "template_count": len(template_ids),
                "usage_count": category_usage
            }
        
        # Quality distribution
        quality_scores = [
            stats.get("average_quality_score", 0.0)
            for stats in self.usage_stats.values()
        ]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Success rate distribution
        success_rates = [
            stats.get("success_rate", 0.0)
            for stats in self.usage_stats.values()
        ]
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        return {
            "total_templates": total_templates,
            "total_usage": total_usage,
            "category_distribution": category_distribution,
            "average_quality_score": avg_quality,
            "average_success_rate": avg_success_rate,
            "categories": list(self.categories.keys()),
            "most_popular_templates": [
                {"id": t.id, "name": t.name, "usage_count": self.usage_stats.get(t.id, {}).get("usage_count", 0)}
                for t in self.get_popular_templates(5)
            ]
        }
    
    def add_template(self, template: PromptTemplate):
        """
        Add a new template to the registry
        
        Args:
            template: Template to add
        """
        self.templates[template.id] = template
        self._add_to_category(template.category, template.id)
        
        # Initialize usage stats
        self.usage_stats[template.id] = {
            "usage_count": 0,
            "last_used": None,
            "success_rate": 0.0,
            "average_quality_score": 0.0,
            "user_ratings": [],
            "common_use_cases": [],
            "performance_metrics": {}
        }
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template from the registry
        
        Args:
            template_id: Template identifier
            
        Returns:
            True if removed, False if not found
        """
        if template_id in self.templates:
            template = self.templates[template_id]
            
            # Remove from templates
            del self.templates[template_id]
            
            # Remove from category
            if template.category in self.categories:
                if template_id in self.categories[template.category]:
                    self.categories[template.category].remove(template_id)
            
            # Remove tags
            if template_id in self.tags:
                del self.tags[template_id]
            
            # Remove usage stats
            if template_id in self.usage_stats:
                del self.usage_stats[template_id]
            
            return True
        
        return False
    
    def get_template_recommendations(self, user_preferences: Dict[str, Any]) -> List[PromptTemplate]:
        """
        Get template recommendations based on user preferences
        
        Args:
            user_preferences: User preferences and history
            
        Returns:
            List of recommended templates
        """
        recommendations = []
        
        # Get user's preferred categories
        preferred_categories = user_preferences.get("categories", [])
        
        # Get user's usage history
        usage_history = user_preferences.get("usage_history", [])
        
        # Get user's quality preferences
        min_quality = user_preferences.get("min_quality_score", 0.0)
        
        # Score templates based on preferences
        template_scores = {}
        
        for template_id, template in self.templates.items():
            score = 0.0
            
            # Category preference
            if template.category in preferred_categories:
                score += 2.0
            
            # Quality score
            stats = self.usage_stats.get(template_id, {})
            quality_score = stats.get("average_quality_score", 0.0)
            if quality_score >= min_quality:
                score += quality_score
            
            # Success rate
            success_rate = stats.get("success_rate", 0.0)
            score += success_rate
            
            # Popularity (usage count)
            usage_count = stats.get("usage_count", 0)
            if usage_count > 0:
                score += min(1.0, usage_count / 100.0)  # Normalize usage count
            
            template_scores[template_id] = score
        
        # Sort by score and return top recommendations
        sorted_templates = sorted(
            template_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for template_id, _ in sorted_templates[:10]:  # Top 10 recommendations
            if template_id in self.templates:
                recommendations.append(self.templates[template_id])
        
        return recommendations


# Global template registry instance
template_registry = TemplateRegistry()
