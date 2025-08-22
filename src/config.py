"""
Configuration module for Process Copilot Mini
What to learn here: Environment variable management, centralized config patterns,
and how to make applications configurable without hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration class that loads settings from environment variables.
    This pattern allows easy configuration changes without modifying code.
    """
    
    # Paths
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    PDF_DIR = DATA_DIR / "pdfs"
    INDEX_DIR = Path(os.getenv("INDEX_DIR", "./models/vector_index"))
    
    # Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "-2.0"))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Chunking parameters can REDUCED FOR MEMORY EFFICIENCY
    CHUNK_SIZE = 600   # Reduced from 600 to avoid MemoryError
    CHUNK_OVERLAP = 100  # Reduced from 100 to avoid MemoryError
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PDF_DIR.mkdir(exist_ok=True)  
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)
