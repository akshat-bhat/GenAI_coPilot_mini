"""
Tests for document ingestion module
What to learn here: Unit testing patterns for data processing,
mocking file operations, and testing text processing functions.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingest import DocumentIngester, DocumentChunk
from src.config import Config


class TestDocumentChunk:
    """Test DocumentChunk class functionality"""
    
    def test_create_chunk(self):
        """Test basic chunk creation"""
        chunk = DocumentChunk(
            text="This is test text",
            title="test_doc", 
            page=1,
            chunk_id="test_doc_p1_c0"
        )
        
        assert chunk.text == "This is test text"
        assert chunk.title == "test_doc"
        assert chunk.page == 1
        assert chunk.chunk_id == "test_doc_p1_c0"
    
    def test_to_dict(self):
        """Test chunk serialization to dictionary"""
        chunk = DocumentChunk("text", "title", 2, "id")
        data = chunk.to_dict()
        
        expected = {
            'text': 'text',
            'title': 'title',
            'page': 2, 
            'chunk_id': 'id'
        }
        
        assert data == expected
    
    def test_from_dict(self):
        """Test chunk creation from dictionary"""
        data = {
            'text': 'sample text',
            'title': 'sample_doc',
            'page': 3,
            'chunk_id': 'sample_doc_p3_c1'
        }
        
        chunk = DocumentChunk.from_dict(data)
        
        assert chunk.text == 'sample text'
        assert chunk.title == 'sample_doc' 
        assert chunk.page == 3
        assert chunk.chunk_id == 'sample_doc_p3_c1'


class TestDocumentIngester:
    """Test DocumentIngester functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.ingester = DocumentIngester()
    
    def test_create_chunks_single_page(self):
        """Test chunk creation from single page"""
        pages_text = [
            {'page': 1, 'text': 'This is a short test document.'}
        ]
        
        chunks = self.ingester.create_chunks(pages_text, "test_doc")
        
        assert len(chunks) == 1
        assert chunks[0].text == 'This is a short test document.'
        assert chunks[0].title == "test_doc"
        assert chunks[0].page == 1
        assert chunks[0].chunk_id == "test_doc_p1_c0"
    
    def test_create_chunks_multiple_pages(self):
        """Test chunk creation from multiple pages"""
        pages_text = [
            {'page': 1, 'text': 'First page content.'},
            {'page': 2, 'text': 'Second page content.'}
        ]
        
        chunks = self.ingester.create_chunks(pages_text, "multi_doc")
        
        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2
        assert chunks[0].title == "multi_doc"
        assert chunks[1].title == "multi_doc"
    
    @patch('src.ingest.PdfReader')
    def test_extract_pdf_text_success(self, mock_pdf_reader):
        """Test successful PDF text extraction"""
        # Mock PDF reader
        mock_reader = MagicMock()
        mock_pdf_reader.return_value = mock_reader
        
        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock() 
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader.pages = [mock_page1, mock_page2]
        
        # Test extraction
        test_path = Path("test.pdf")
        result = self.ingester.extract_pdf_text(test_path)
        
        assert len(result) == 2
        assert result[0]['page'] == 1
        assert result[0]['text'] == "Page 1 content"
        assert result[1]['page'] == 2
        assert result[1]['text'] == "Page 2 content"
    
    @patch('src.ingest.PdfReader')
    def test_extract_pdf_text_failure(self, mock_pdf_reader):
        """Test PDF extraction with errors"""
        # Mock PDF reader to raise exception
        mock_pdf_reader.side_effect = Exception("PDF parsing error")
        
        test_path = Path("bad.pdf")
        result = self.ingester.extract_pdf_text(test_path)
        
        assert result == []
    
    def test_save_and_load_chunks(self):
        """Test chunk persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override chunks file path
            self.ingester.chunks_file = Path(temp_dir) / "test_chunks.jsonl"
            
            # Create test chunks
            chunks = [
                DocumentChunk("text1", "doc1", 1, "id1"),
                DocumentChunk("text2", "doc2", 2, "id2")
            ]
            
            # Save chunks
            self.ingester.save_chunks(chunks)
            
            # Verify file exists
            assert self.ingester.chunks_file.exists()
            
            # Load chunks back
            loaded_chunks = self.ingester.load_chunks()
            
            assert len(loaded_chunks) == 2
            assert loaded_chunks[0].text == "text1"
            assert loaded_chunks[1].text == "text2"
    
    def test_load_chunks_no_file(self):
        """Test loading chunks when file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set non-existent file path
            self.ingester.chunks_file = Path(temp_dir) / "nonexistent.jsonl"
            
            chunks = self.ingester.load_chunks()
            
            assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__])