"""AI Analysis module for figures and tables"""

from .ai_analyzer import AIAnalyzer, analyze_single
from .config import AnalysisConfig

__all__ = [
    'AIAnalyzer',
    'analyze_single',
    'AnalysisConfig'
]