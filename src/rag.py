"""
RAG (Retrieval-Augmented Generation) module for Process Copilot Mini
What to learn here: Combining retrieval with generation, prompt engineering,
confidence scoring, and grounding responses in source documents.
"""

from typing import List, Dict, Any, Optional
import logging

from .index import VectorIndex
from .config import Config
from .utils import setup_logging, format_citations


logger = setup_logging()

class RAGSystem:
    """
    Retrieval-Augmented Generation system that combines document search
    with response generation.
    
    Key principles:
    - Grounding: All answers must be based on retrieved documents
    - Confidence: Return "I don't know" when retrieval confidence is low
    - Citations: Always provide source references for transparency
    - No hallucination: Never generate information not found in documents
    """
    
    def __init__(self):
        self.index = VectorIndex()
        self._ensure_index_loaded()
    
    def _ensure_index_loaded(self):
        """Ensure the vector index is loaded and ready"""
        if not self.index.load_index():
            logger.warning("Vector index not available. Please build index first.")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks with metadata
        """
        k = k or Config.RETRIEVAL_K
        
        try:
            results = self.index.search(query, k=k)
            logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _check_retrieval_confidence(self, results: List[Dict[str, Any]], query: str = "") -> bool:
        """
        Check if retrieval results meet confidence threshold.
        ENHANCED: Add domain relevance checking for industrial queries
        """
        if not results:
            logger.info("No results returned from search")
            return False

        # Check if results is actually a list of dicts
        if not isinstance(results, list) or not results:
            logger.warning("Results is not a proper list")
            return False

        top_result = results[0]
        if not isinstance(top_result, dict):
            logger.warning("Top result is not a dictionary")
            return False

        top_score = top_result.get('score', 0.0)
        logger.info(f"Top result score: {top_score}, threshold: {Config.SCORE_THRESHOLD}")

        if top_score < 0:
            # L2 distances: smaller absolute value = better match
            similarity = 1.0 / (1.0 + abs(top_score))  # Convert to 0-1 range
            confidence_threshold = 0.5  # Stricter base threshold
        else:
            # Already converted similarity scores
            similarity = top_score
            confidence_threshold = Config.SCORE_THRESHOLD

        # Add domain relevance check
        query_words = set(query.lower().split())
        industrial_keywords = {
            'temperature', 'pressure', 'alarm', 'control', 'valve', 'sensor', 
            'calibration', 'maintenance', 'safety', 'procedure', 'process',
            'emergency', 'shutdown', 'troubleshooting', 'psi', 'celsius', 'degrees',
            'flow', 'pump', 'system', 'operating', 'calibrate', 'high', 'low',
            'normal', 'range', 'setpoint', 'instrumentation'
        }
        
        # Check if query has industrial relevance
        has_industrial_context = bool(query_words.intersection(industrial_keywords))
        
        if not has_industrial_context:
            # For completely off-topic queries, use very high threshold
            confidence_threshold = 0.8
            logger.info(f"No industrial keywords detected in query '{query}', using very strict threshold: {confidence_threshold}")
        
        passes_confidence = similarity >= confidence_threshold
        logger.info(f"Confidence check: {passes_confidence} (similarity {similarity:.3f} >= {confidence_threshold})")
        
        return passes_confidence

        
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for prompt.
        
        Args:
            results: Retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Include source information in context
            # context_part = f"[Source {i}: {result['title']}, page {result['page']}]\n{result['text']}"
            
            if not isinstance(result, dict):
                continue
                
            text = result.get('text', '')
            
            # Ensure text is a string
            if isinstance(text, list):
                text = ' '.join(str(item) for item in text)
            elif not isinstance(text, str):
                text = str(text)
            
            # Now safely strip
            text = text.strip()
            
            if text:  # Only add non-empty text
                context_parts.append(text)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate clean answer based on query and retrieved context.
        
        For this educational version, we use a template-based approach.
        In production, you would plug in an LLM here (OpenAI, Hugging Face, etc.)
        
        Args:
            query: User's question
            context: Retrieved document context
            
        Returns:
            Generated answer
        """
        
        # Template-based generation for educational purposes
        # TODO: Replace with actual LLM (OpenAI GPT, Llama, etc.)
        
        if not context or not context.strip():
            return "No relevant information found in the documents."
        
        # Remove source annotations from context
        clean_context = ""
        for line in context.split('\n'):
            if not line.strip().startswith('[Source'):
                clean_context += line + " "
        
        # Extract the most relevant information
        query_lower = query.lower()
        
        # For temperature range questions
        if "temperature" in query_lower and "range" in query_lower:
            if "Normal Operating Range:" in clean_context:
                # Extract temperature range info
                parts = clean_context.split("Normal Operating Range:")
                if len(parts) > 1:
                    temp_info = parts[1].split("High Alarm Setpoint:")[0].strip()
                    return f"The normal operating temperature range is {temp_info}."
        
        # For alarm/procedure questions  
        if "alarm" in query_lower or "procedure" in query_lower:
            # Look for numbered procedures
            sentences = clean_context.split('.')
            procedures = []
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ["1.", "2.", "3.", "check", "verify", "reduce"]):
                    if len(sentence) > 10 and len(sentence) < 200:
                        procedures.append(sentence)
            
            if procedures:
                return ". ".join(procedures[:3]) + "."
        
        # For calibration questions
        if "calibration" in query_lower or "calibrate" in query_lower:
            if "calibration procedure:" in clean_context.lower():
                parts = clean_context.lower().split("calibration procedure:")
                if len(parts) > 1:
                    cal_info = parts[1].split("preventive maintenance:")[0].strip()
                    return f"Calibration procedure: {cal_info}."
        
        # Default: return first meaningful sentences
        sentences = [s.strip() for s in clean_context.split('.') if len(s.strip()) > 20]
        if sentences:
            # Return first 2-3 sentences that seem most relevant
            answer_sentences = sentences[:2]
            answer = ". ".join(answer_sentences)
            if not answer.endswith('.'):
                answer += '.'
            return answer
        
        return clean_context[:300] + "..." if len(clean_context) > 300 else clean_context


    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main RAG pipeline: retrieve relevant documents and generate answer.
        
        Process:
        1. Retrieve relevant document chunks
        2. Check retrieval confidence
        3. Generate answer from context (or return "I don't know")
        4. Format citations
        
        Args:
            query: User's question
            
        Returns:
            Dict with answer and citations
        """
        
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Step 1: Retrieve relevant chunks
            results = self.retrieve(query)

            # Ensure results is a list
            if not isinstance(results, list):
                logger.error(f"Retrieve returned non-list: {type(results)}")
                results = []
            
            # Step 2: Check confidence
            if not self._check_retrieval_confidence(results, query):
                return {
                    "answer": "I don't know the answer to that question based on the available documents. Please try rephrasing your question or asking about topics covered in the technical manuals.",
                    "citations": []
                }
            
            # Step 3: Format context and generate answer
            context = self._format_context(results)
            answer = self._generate_answer(query, context)
            
            # Step 4: Format citations - ensure we pass valid results
            valid_results = [r for r in results if isinstance(r, dict)]
            citations = format_citations(valid_results)
            
            logger.info(f"Generated answer with {len(citations)} citations")
            
            return {
                "answer": answer,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "citations": []
            }

def main():
    """
    Test the RAG system with sample queries.
    Run: python -m src.rag
    """
    print("🤖 Testing RAG System")
    print("=" * 50)
    
    rag = RAGSystem()
    
    # Test queries
    test_queries = [
        "What is the operating temperature range?",
        "How do you calibrate the pressure sensor?", 
        "What are the safety procedures for high temperature alarms?",
        "Tell me about unicorns"  # Should return "I don't know"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        result = rag.ask(query)
        
        print(f"Answer: {result['answer']}")
        
        if result['citations']:
            print(f"\nCitations ({len(result['citations'])}):")
            for j, citation in enumerate(result['citations'], 1):
                print(f"  {j}. {citation['title']}, page {citation['page']} (score: {citation['score']})")
        else:
            print("No citations")
        
        print()

if __name__ == "__main__":
    main()