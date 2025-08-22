"""
Tests for RAG system functionality
What to learn here: Testing retrieval-generation pipelines,
mocking vector search, and validating confidence gating.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.rag import RAGSystem
from src.config import Config


class TestRAGSystem:
    """Test RAG system functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.rag = RAGSystem()
    
    @patch('src.rag.VectorIndex')
    def test_retrieve_success(self, mock_vector_index):
        """Test successful retrieval"""
        # Mock search results
        mock_results = [
            {'text': 'relevant content', 'title': 'doc1', 'page': 1, 'score': 0.9, 'chunk_id': 'id1'},
            {'text': 'also relevant', 'title': 'doc2', 'page': 2, 'score': 0.8, 'chunk_id': 'id2'}
        ]
        
        mock_index_instance = MagicMock()
        mock_index_instance.search.return_value = mock_results
        mock_index_instance.load_index.return_value = True
        mock_vector_index.return_value = mock_index_instance
        
        rag = RAGSystem()
        results = rag.retrieve("test query")
        
        assert len(results) == 2
        assert results[0]['score'] == 0.9
        assert results[1]['score'] == 0.8
    
    def test_check_retrieval_confidence_high(self):
        """Test confidence check with high score"""
        results = [{'score': 0.5}]  # Above threshold (0.35)
        
        confidence_ok = self.rag._check_retrieval_confidence(results, "temperature range")
        
        assert confidence_ok is True
    
    def test_check_retrieval_confidence_low(self):
        """Test confidence check with low score"""
        results = [{'score': 0.1}]  # Below threshold (0.35)
        
        confidence_ok = self.rag._check_retrieval_confidence(results, "temperature range")
        
        assert confidence_ok is False
    
    def test_check_retrieval_confidence_empty(self):
        """Test confidence check with no results"""
        results = []
        
        confidence_ok = self.rag._check_retrieval_confidence(results, "temperature range")
        
        assert confidence_ok is False
    
    def test_format_context(self):
        """Test context formatting for prompt"""
        results = [
            {'text': 'First content', 'title': 'Doc1', 'page': 1},
            {'text': 'Second content', 'title': 'Doc2', 'page': 3}
        ]
        
        context = self.rag._format_context(results)
        
        assert 'First content' in context
        assert 'First content' in context
        assert 'Second content' in context
        assert 'Second content' in context
    
    def test_generate_answer_basic(self):
        """Test basic answer generation"""
        query = "What is temperature?"
        context = "The normal temperature range is 20-25 degrees Celsius."
        
        answer = self.rag._generate_answer(query, context)
        
        assert len(answer) > 0
        assert isinstance(answer, str)
    
    @patch('src.rag.VectorIndex')
    def test_ask_high_confidence(self, mock_vector_index):
        """Test ask method with high confidence results"""
        # Mock high-confidence search results
        mock_results = [
            {'text': 'Temperature range is 20-25°C', 'title': 'Manual', 'page': 5, 'score': 0.8, 'chunk_id': 'id1'}
        ]
        
        mock_index_instance = MagicMock()
        mock_index_instance.search.return_value = mock_results
        mock_index_instance.load_index.return_value = True
        mock_vector_index.return_value = mock_index_instance
        
        rag = RAGSystem()
        result = rag.ask("What is the temperature range?")
        
        assert 'answer' in result
        assert 'citations' in result
        assert len(result['citations']) == 1
        assert result['citations'][0]['title'] == 'Manual'
    
    @patch('src.rag.VectorIndex') 
    def test_ask_low_confidence(self, mock_vector_index):
        """Test ask method with low confidence results"""
        # Mock low-confidence search results
        mock_results = [
            {'text': 'irrelevant content', 'title': 'Doc', 'page': 1, 'score': 0.1, 'chunk_id': 'id1'}
        ]
        
        mock_index_instance = MagicMock()
        mock_index_instance.search.return_value = mock_results
        mock_index_instance.load_index.return_value = True
        mock_vector_index.return_value = mock_index_instance
        
        rag = RAGSystem()
        result = rag.ask("What is the temperature range?")
        
        assert "I don't know" in result['answer']
        assert len(result['citations']) == 0


if __name__ == "__main__":
    pytest.main([__file__])