"""
Specialized Templates Module
Author: Balaji Koneti

This module contains specialized templates for different domains including
research, legal, medical, financial, and creative applications.
"""

from .research_templates import RESEARCH_TEMPLATES
from .legal_templates import LEGAL_TEMPLATES
from .medical_templates import MEDICAL_TEMPLATES
from .financial_templates import FINANCIAL_TEMPLATES
from .creative_templates import CREATIVE_TEMPLATES

__all__ = [
    "RESEARCH_TEMPLATES",
    "LEGAL_TEMPLATES", 
    "MEDICAL_TEMPLATES",
    "FINANCIAL_TEMPLATES",
    "CREATIVE_TEMPLATES"
]
