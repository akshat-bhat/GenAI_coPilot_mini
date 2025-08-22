"""
Document ingestion module for Process Copilot Mini
What to learn here: PDF text extraction, metadata handling, chunking strategies,
and how to prepare documents for vector search.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from pypdf import PdfReader

from .config import Config
from .utils import chunk_text, clean_text, setup_logging

logger = setup_logging()

class DocumentChunk:
    """
    Represents a chunk of text from a document with metadata.
    Why use a class: Encapsulates chunk data and provides consistent interface.
    """
    def __init__(self, text: str, title: str, page: int, chunk_id: str):
        self.text = text
        self.title = title
        self.page = page
        self.chunk_id = chunk_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'title': self.title, 
            'page': self.page,
            'chunk_id': self.chunk_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary"""
        return cls(
            text=data['text'],
            title=data['title'],
            page=data['page'],
            chunk_id=data['chunk_id']
        )

class DocumentIngester:
    """
    Handles PDF ingestion, text extraction, and chunking.
    
    Key responsibilities:
    - Extract text from PDFs page by page
    - Create overlapping chunks with metadata
    - Handle various PDF formats and edge cases
    - Persist processed chunks to disk
    """
    
    def __init__(self):
        self.chunks_file = Config.DATA_DIR / "processed_chunks.jsonl"
        Config.ensure_directories()
    
    def extract_pdf_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file page by page.
        
        Why page-by-page extraction:
        - Preserves page number information for citations
        - Allows better error handling for corrupted pages
        - Enables page-specific processing if needed
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts with 'page', 'text' keys
        """
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            
            logger.info(f"Processing PDF: {pdf_path.name} ({len(reader.pages)} pages)")
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text.strip():  # Only include pages with content
                        cleaned_text = clean_text(text)
                        pages_text.append({
                            'page': page_num,
                            'text': cleaned_text
                        })
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num} from {pdf_path.name}: {e}")
                    continue
            
            return pages_text
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path.name}: {e}")
            return []
    
    def create_chunks(self, pages_text: List[Dict[str, Any]], title: str) -> List[DocumentChunk]:
        """
        Create overlapping text chunks with metadata.
        
        Chunking strategy:
        - Combine text from multiple pages if chunks are small
        - Preserve page boundaries where possible
        - Add overlap to maintain context
        - Track original page numbers for citations
        
        Args:
            pages_text: List of page text dictionaries
            title: Document title (filename)
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_counter = 0
        
        for page_data in pages_text:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Create chunks from this page
            page_chunks = chunk_text(page_text)
            
            for chunk_text_content in page_chunks:
                chunk_id = f"{title}_p{page_num}_c{chunk_counter}"
                
                chunk = DocumentChunk(
                    text=chunk_text_content,
                    title=title,
                    page=page_num,
                    chunk_id=chunk_id
                )
                
                chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks for {title}")
        return chunks
    
    def save_chunks(self, chunks: List[DocumentChunk]):
        """
        Save processed chunks to JSONL file.
        
        Why JSONL format:
        - Easy to append new documents
        - Can be processed line-by-line for large files
        - Human readable for debugging
        - Compatible with many data processing tools
        """
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(chunks)} chunks to {self.chunks_file}")
    
    def load_chunks(self) -> List[DocumentChunk]:
        """Load processed chunks from disk"""
        if not self.chunks_file.exists():
            return []
        
        chunks = []
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        chunks.append(DocumentChunk.from_dict(data))
            
            logger.info(f"Loaded {len(chunks)} chunks from disk")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []
    
    def ingest_pdfs(self, pdf_dir: Path = None) -> List[DocumentChunk]:
        """
        Process all PDFs in the specified directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            List of all processed chunks
        """
        pdf_dir = pdf_dir or Config.PDF_DIR
        
        if not pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_dir} does not exist")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            
            # Extract text from PDF
            pages_text = self.extract_pdf_text(pdf_file)
            
            if not pages_text:
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue
            
            # Create chunks
            title = pdf_file.stem  # Filename without extension
            chunks = self.create_chunks(pages_text, title)
            
            all_chunks.extend(chunks)
        
        if all_chunks:
            # Save all chunks to disk
            self.save_chunks(all_chunks)
            logger.info(f"Total chunks processed: {len(all_chunks)}")
        
        return all_chunks

def main():
    """
    Standalone script to ingest PDFs.
    Run: python -m src.ingest
    """
    ingester = DocumentIngester()
    chunks = ingester.ingest_pdfs()
    
    if chunks:
        print(f"Successfully processed {len(chunks)} chunks from PDF files")
        print(f"Chunks saved to: {ingester.chunks_file}")
    else:
        print("No chunks were processed. Check if PDF files exist in data/pdfs/")

if __name__ == "__main__":
    main()
