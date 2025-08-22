"""
FastAPI application for Process Copilot Mini
What to learn here: RESTful API design, request/response models, error handling,
and how to expose ML/AI capabilities through clean HTTP interfaces.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .config import Config
from .rag import RAGSystem
from .alarms import AlarmAnalyzer
from .utils import setup_logging


# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Process Copilot Mini",
    description="AI-powered industrial process assistant with RAG and alarm analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web UI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
rag_system = RAGSystem()
alarm_analyzer = AlarmAnalyzer()

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for document Q&A"""
    query: str = Field(..., description="Question about technical documents", min_length=1, max_length=500)

class Citation(BaseModel):
    """Citation model for source references"""
    title: str
    page: int 
    score: float

class QueryResponse(BaseModel):
    """Response model for document Q&A"""
    answer: str
    citations: List[Citation]

class AlarmResponse(BaseModel):
    """Response model for alarm analysis"""
    summary_from_data: str
    answer: str
    citations: List[Citation]

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Why health checks matter:
    - Load balancers use this to route traffic
    - Monitoring systems check service availability  
    - Helps diagnose deployment issues
    - Standard practice for production APIs
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Document Q&A endpoint using RAG.
    
    Process:
    1. Validate input query
    2. Retrieve relevant document chunks
    3. Generate grounded answer with citations
    4. Return structured response
    
    Args:
        request: Query request with user question
        
    Returns:
        Answer with source citations
        
    Raises:
        HTTPException: For invalid requests or processing errors
    """
    logger.info(f"Received question: '{request.query}'")
    
    try:
        # Process query through RAG system
        result = rag_system.ask(request.query)
        
        # Convert to response format
        citations = [
            Citation(
                title=cite['title'],
                page=cite['page'], 
                score=cite['score']
            )
            for cite in result.get('citations', [])
        ]
        
        response = QueryResponse(
            answer=result.get('answer', 'No answer generated'),
            citations=citations
        )
        
        logger.info(f"Generated answer with {len(citations)} citations")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process question: {str(e)}"
        )

@app.get("/explain_alarm", response_model=AlarmResponse)
async def explain_alarm(
    tag: str = Query(..., description="Process tag identifier (e.g., 'Temp_101')"),
    start: str = Query(..., description="Start time in ISO format (e.g., '2024-08-20T15:00:00')"),
    end: str = Query(..., description="End time in ISO format (e.g., '2024-08-20T16:00:00')")
):
    """
    Alarm analysis endpoint combining data analysis with document guidance.
    
    Process:
    1. Validate time window parameters
    2. Load and filter alarm data
    3. Compute statistical summaries and trends
    4. Retrieve relevant procedural guidance
    5. Combine data insights with document citations
    
    Args:
        tag: Process tag to analyze
        start: Analysis start time  
        end: Analysis end time
        
    Returns:
        Data summary and procedural guidance with citations
        
    Raises:
        HTTPException: For invalid parameters or processing errors
    """
    logger.info(f"Alarm analysis requested - Tag: {tag}, Window: {start} to {end}")
    
    try:
        # Validate time format (basic check)
        try:
            datetime.fromisoformat(start.replace('Z', '+00:00'))
            datetime.fromisoformat(end.replace('Z', '+00:00'))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time format. Use ISO format (e.g., '2024-08-20T15:00:00'): {e}"
            )
        
        # Process alarm analysis
        result = alarm_analyzer.explain_alarm(tag, start, end)
        
        # Convert to response format  
        citations = [
            Citation(
                title=cite['title'],
                page=cite['page'],
                score=cite['score'] 
            )
            for cite in result.get('citations', [])
        ]
        
        response = AlarmResponse(
            summary_from_data=result.get('summary_from_data', 'No data summary available'),
            answer=result.get('answer', 'No guidance available'),
            citations=citations
        )
        
        logger.info(f"Completed alarm analysis with {len(citations)} citations")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in alarm analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze alarm: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Process Copilot Mini API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation", 
            "ask": "POST /ask - Document Q&A with RAG",
            "explain_alarm": "GET /explain_alarm - Alarm analysis with guidance"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    
    Why global handlers matter:
    - Ensures consistent error responses
    - Prevents sensitive error details from leaking
    - Enables centralized error logging
    - Improves API reliability
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return HTTPException(
        status_code=500,
        detail="Internal server error occurred"
    )

if __name__ == "__main__":
    """
    Run the application directly for development.
    For production, use: uvicorn src.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    logger.info("Starting Process Copilot Mini API...")
    logger.info(f"Host: {Config.API_HOST}, Port: {Config.API_PORT}")
    
    uvicorn.run(
        "src.app:app",  # module:app
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,  # Enable auto-reload for development
        log_level=Config.LOG_LEVEL.lower()
    )