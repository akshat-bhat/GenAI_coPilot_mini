"""
Streamlit UI for Process Copilot Mini
What to learn here: Building intuitive interfaces for AI applications,
handling user interactions, and presenting complex information clearly.
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Process Copilot Mini",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def ask_question(query: str) -> Dict[str, Any]:
    """Send question to RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {e}")
        return {"answer": "Error connecting to API", "citations": []}

def explain_alarm(tag: str, start: str, end: str) -> Dict[str, Any]:
    """Send alarm analysis request to API"""
    try:
        params = {"tag": tag, "start": start, "end": end}
        response = requests.get(
            f"{API_BASE_URL}/explain_alarm",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {e}")
        return {
            "summary_from_data": "Error connecting to API",
            "answer": "Unable to retrieve alarm analysis",
            "citations": []
        }

def display_citations(citations: list):
    """Display citations in a formatted way"""
    if not citations:
        return
    
    st.subheader("📚 Sources")
    for i, citation in enumerate(citations, 1):
        score_color = "🟢" if citation['score'] > 0.8 else "🟡" if citation['score'] > 0.5 else "🔴"
        st.write(f"{i}. **{citation['title']}** (page {citation['page']}) {score_color} Score: {citation['score']:.3f}")

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🏭 Process Copilot Mini")
    st.markdown("""
    **AI-powered industrial process assistant** with document Q&A and alarm analysis capabilities.
    
    - **Ask Docs**: Get answers from technical manuals with citations
    - **Explain Alarm**: Analyze process data and get procedural guidance
    """)
    
    # Check API health
    api_status = check_api_health()
    if api_status:
        st.success("✅ API is running")
    else:
        st.error("❌ API is not accessible. Please start the FastAPI server first.")
        st.code("uvicorn src.app:app --host 0.0.0.0 --port 8000")
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📖 Ask Docs", "🚨 Explain Alarm"])
    
    # Tab 1: Document Q&A
    with tab1:
        st.header("Document Q&A")
        st.markdown("""
        Ask questions about technical documents and get answers with source citations.
        The system will only answer questions that can be found in the uploaded manuals.
        """)
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What is the normal operating temperature range?",
                "How do you calibrate the pressure sensor?",
                "What are the safety procedures for high alarms?", 
                "What maintenance is required monthly?"
            ]
            for q in sample_questions:
                if st.button(q, key=f"sample_{q}"):
                    st.session_state['current_question'] = q
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What is the maximum operating pressure?",
            key="question_input"
        )
        
        # Ask button
        if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
            with st.spinner("Searching documents and generating answer..."):
                result = ask_question(question)
            
            # Display results
            st.subheader("💬 Answer")
            st.write(result['answer'])
            
            # Display citations
            display_citations(result.get('citations', []))
            
            # Clear the session state
            if 'current_question' in st.session_state:
                del st.session_state['current_question']
    
    # Tab 2: Alarm Analysis  
    with tab2:
        st.header("Alarm & Process Analysis")
        st.markdown("""
        Analyze process alarm data and get procedural guidance based on technical manuals.
        Select a process tag and time window to get data insights combined with operational procedures.
        """)
        
        # Create columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Process tag selection
            tag_options = ["Temp_101", "Pressure_202"]
            selected_tag = st.selectbox(
                "Process Tag:",
                options=tag_options,
                help="Select the process variable to analyze"
            )
            
            # Start time
            start_date = st.date_input("Start Date", value=datetime(2024, 8, 20))
            start_time = st.time_input("Start Time", value=datetime.strptime("15:00", "%H:%M").time())
            
        with col2:
            # Quick time range buttons
            st.write("Quick Select:")
            if st.button("Last Hour"):
                end_datetime = datetime(2024, 8, 20, 16, 30, 0)
                start_datetime = end_datetime - timedelta(hours=1)
                st.session_state['alarm_start'] = start_datetime
                st.session_state['alarm_end'] = end_datetime
                
            if st.button("Temperature Spike (15:30-16:30)"):
                st.session_state['alarm_start'] = datetime(2024, 8, 20, 15, 30, 0)
                st.session_state['alarm_end'] = datetime(2024, 8, 20, 16, 30, 0)
                st.session_state['selected_tag'] = "Temp_101"
            
            if st.button("Pressure Event (15:00-16:00)"):  
                st.session_state['alarm_start'] = datetime(2024, 8, 20, 15, 0, 0)
                st.session_state['alarm_end'] = datetime(2024, 8, 20, 16, 0, 0)
                st.session_state['selected_tag'] = "Pressure_202"
            
            # End time
            end_date = st.date_input("End Date", value=datetime(2024, 8, 20))
            end_time = st.time_input("End Time", value=datetime.strptime("16:00", "%H:%M").time())
        
        # Use session state values if available
        if 'alarm_start' in st.session_state:
            start_datetime = st.session_state['alarm_start']
            end_datetime = st.session_state['alarm_end']
        else:
            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)
            
        if 'selected_tag' in st.session_state:
            selected_tag = st.session_state['selected_tag']
        
        # Format times for API
        start_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Display selected time range
        st.info(f"**Analysis Window:** {selected_tag} from {start_str} to {end_str}")
        
        # Analyze button
        if st.button("🔬 Analyze Alarm", type="primary"):
            with st.spinner("Analyzing process data and retrieving guidance..."):
                result = explain_alarm(selected_tag, start_str, end_str)
            
            # Display results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Data Analysis")
                st.text_area(
                    "Process Statistics:",
                    value=result['summary_from_data'],
                    height=200,
                    disabled=True
                )
            
            with col2:
                st.subheader("📋 Operational Guidance")
                st.text_area(
                    "Recommended Actions:",
                    value=result['answer'],
                    height=200,
                    disabled=True
                )
            
            # Display citations
            display_citations(result.get('citations', []))
            
            # Clear session state
            for key in ['alarm_start', 'alarm_end', 'selected_tag']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Process Copilot Mini v1.0 | Built for industrial automation learning</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()