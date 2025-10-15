"""
Configuration for AI Analysis Module
"""

import os
from dotenv import load_dotenv

load_dotenv()


class AnalysisConfig:
    """Configuration for Gemini AI analysis"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2  # seconds (exponential backoff)
    
    # Rate Limiting (Gemini free tier: 15 RPM)
    RATE_LIMIT_RPM = 15
    BATCH_SIZE = 10
    
    # Output Modes
    SUMMARY_MODES = ["text", "json", "detailed"]
    DEFAULT_SUMMARY_MODE = "text"
    
    # Table Processing
    TABLE_FORMAT = "markdown"  # markdown, csv, plain
    TABLE_MAX_ROWS = None  # None = all rows, or set limit (e.g., 20)
    
    # Prompts
    FIGURE_PROMPT_TEMPLATE = """
You are a scientific figure analyst. Analyze this research figure and provide:

1. **Figure Type**: (graph/chart/diagram/architecture/photo/other)
2. **Main Content**: What does it show?
3. **Key Observations**: 2-3 specific insights

Caption: {caption}

Be concise and focus on scientific content.
"""
    
    TABLE_TEXT_PROMPT_TEMPLATE = """
You are a scientific data analyst. Summarize the key findings from this table.

Caption: {caption}

Table:
{table_text}

Provide:
1. Main metrics/columns
2. Key comparisons or trends
3. One important conclusion

Be concise (3-4 sentences).
"""
    
    TABLE_JSON_PROMPT_TEMPLATE = """
You are a scientific data analyst. Analyze this table and return JSON:

{{
  "metrics": ["list of key column names or measured metrics"],
  "observations": ["main trends, comparisons, or patterns"],
  "insight": "one-sentence conclusion"
}}

Caption: {caption}

Table:
{table_text}

Return ONLY valid JSON, no markdown formatting.
"""
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "[ERROR] GEMINI_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
        
        if cls.DEFAULT_SUMMARY_MODE not in cls.SUMMARY_MODES:
            raise ValueError(
                f"Invalid DEFAULT_SUMMARY_MODE: {cls.DEFAULT_SUMMARY_MODE}. "
                f"Must be one of {cls.SUMMARY_MODES}"
            )
        
        print("[OK] Configuration validated")


# Validate on import
try:
    AnalysisConfig.validate()
except Exception as e:
    print(f"[WARNING] Configuration warning: {e}")