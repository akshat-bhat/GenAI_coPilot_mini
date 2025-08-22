"""
Utility functions for Process Copilot Mini
What to learn here: Logging setup, text chunking strategies, and helper functions
that make the codebase more modular and testable.
"""

import logging
import re
from typing import List, Tuple
from .config import Config

def setup_logging() -> logging.Logger:
    """
    Set up structured logging for the application.
    Why this matters: Proper logging is critical for debugging production issues
    and monitoring application behavior.
    """
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console output
        ]
    )
    return logging.getLogger("process-copilot")

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    FIXED VERSION: Prevents infinite loops and memory issues
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP
    
    # Safety check
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    max_chunks = 1000  # Safety limit to prevent infinite loops
    chunk_count = 0
    
    while start < len(text) and chunk_count < max_chunks:
        # Find end position
        end = min(start + chunk_size, len(text))
        
        # Try to break at sentence boundaries for better context
        if end < len(text):
            # Look for sentence endings within the last 50 characters  
            search_start = max(start, end - 50)
            sentence_pattern = r'[.!?]\s+'
            
            # Find sentence boundaries in reverse order (closest to end)
            text_segment = text[search_start:end]
            matches = list(re.finditer(sentence_pattern, text_segment))
            
            if matches:
                # Use the last sentence boundary found
                last_match = matches[-1]
                end = search_start + last_match.end()
        
        # Extract chunk
        chunk = text[start:end].strip()
        
        if chunk and len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position with safety check
        new_start = end - overlap
        
        # Ensure we're making progress to prevent infinite loops
        if new_start <= start:
            new_start = start + max(1, chunk_size // 2)  # Force progress
        
        start = new_start
        chunk_count += 1
        
        # Additional safety: if we're near the end, just take the rest
        if len(text) - start < chunk_size // 2:
            remaining = text[start:].strip()
            if remaining and remaining not in chunks:
                chunks.append(remaining)
            break
    
    # Log warning if we hit the safety limit
    if chunk_count >= max_chunks:
        import logging
        logger = logging.getLogger("process-copilot")
        logger.warning(f"Hit maximum chunk limit ({max_chunks}) for safety. Text may be truncated.")
    
    return chunks


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and formatting artifacts.
    
    Why text cleaning matters:
    - PDF extraction often produces messy text with irregular spacing
    - Clean text improves embedding quality and search relevance
    - Consistent formatting makes results more readable
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\x20-\x7E]', ' ', text)  # Keep only printable ASCII
    
    # Clean up sentence boundaries
    text = re.sub(r'\s+([.!?])', r'\1', text)
    
    return text.strip()

def extract_page_number(text: str) -> List[int]:
    """
    Extract page numbers from text (simple heuristic).
    This is a basic implementation - in production you'd want more sophisticated
    page number detection or rely on PDF metadata.
    """
    # Look for patterns like "Page 1", "- 1 -", etc.
    patterns = [
        r'(?:page|pg\.?)\s*(\d+)',
        r'-\s*(\d+)\s*-',
        r'^\s*(\d+)\s*$'
    ]
    
    page_nums = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            try:
                page_nums.append(int(match.group(1)))
            except (ValueError, IndexError):
                continue
    
    return page_nums

def format_citations(results: List[dict]) -> List[dict]:
    """
    Format search results into citation format.
    
    Args:
        results: List of search results with 'title', 'page', 'score' keys
        
    Returns:
        List of formatted citations
    """
    citations = []
    for result in results:
        
        # Skip if result is None or not a dict
        if not result or not isinstance(result, dict):
            continue
        
        raw_score = result.get('score', 0.0)

        # Convert L2 distance to similarity score (0-1)
        if raw_score < 0:
            # L2 dist. are -ve, convert to similarity
            similarity_score = 1.0 / (1.0 + abs(raw_score))
        else:
            # already converted score
            similarity_score = raw_score


        citation = {
            'title': result.get('title', 'Unknown'),
            'page': result.get('page', 'N/A'), 
            'score': round(similarity_score, 3)  # Now in range 0-1
        }
        citations.append(citation)
    
    return citations