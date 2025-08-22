"""
Alarm and process data analysis module for Process Copilot Mini
What to learn here: Time-series data processing, alarm pattern analysis,
combining data analysis with document retrieval for industrial applications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .config import Config
from .rag import RAGSystem
from .utils import setup_logging


logger = setup_logging()

class AlarmAnalyzer:
    """
    Analyzes process alarm data and provides data-driven insights
    combined with document-based guidance.
    
    Industrial Context:
    - Process tags represent sensors (temperature, pressure, flow, etc.)
    - Alarm states indicate process conditions (OK, High, HighHigh)
    - Trend analysis helps predict equipment issues
    - Document retrieval provides procedural guidance
    """
    
    def __init__(self):
        self.rag = RAGSystem()
        
    def load_alarm_data(self, csv_path: Path = None) -> pd.DataFrame:
        """
        Load alarm data from CSV file.
        
        Expected format:
        - timestamp: ISO format datetime
        - tag: Process tag identifier  
        - value: Sensor reading
        - alarm_state: OK, High, HighHigh, etc.
        
        Args:
            csv_path: Path to alarm data CSV
            
        Returns:
            DataFrame with alarm data
        """
        csv_path = csv_path or (Config.DATA_DIR / "samples" / "alarms.csv")
        
        try:
            if not csv_path.exists():
                logger.error(f"Alarm data file not found: {csv_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(csv_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp and tag for consistent processing
            df = df.sort_values(['timestamp', 'tag']).reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} alarm records from {csv_path}")
            logger.info(f"Tags: {df['tag'].unique().tolist()}")
            logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load alarm data: {e}")
            return pd.DataFrame()
    
    def slice_by_time(self, df: pd.DataFrame, tag: str, start: str, end: str) -> pd.DataFrame:
        """
        Extract data for specific tag and time window.
        
        Args:
            df: Full alarm dataset
            tag: Process tag to filter
            start: Start time (ISO format)
            end: End time (ISO format)
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Parse time strings
            start_time = pd.to_datetime(start)
            end_time = pd.to_datetime(end)
            
            # Filter by tag and time range
            mask = (
                (df['tag'] == tag) & 
                (df['timestamp'] >= start_time) & 
                (df['timestamp'] <= end_time)
            )
            
            filtered_df = df[mask].copy()
            
            logger.info(f"Filtered to {len(filtered_df)} records for {tag} from {start} to {end}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Failed to slice data: {e}")
            return pd.DataFrame()
    
    def compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for time series data.
        
        Industrial relevance:
        - Min/max values indicate operating range
        - Mean shows typical operating point
        - Trends indicate equipment degradation or process drift
        - Alarm transitions show process stability
        
        Args:
            df: Time series data for a single tag
            
        Returns:
            Dictionary of summary statistics
        """
        if df.empty:
            return {}
        
        try:
            values = df['value'].values
            timestamps = df['timestamp'].values
            
            # Basic statistics
            stats = {
                'count': len(values),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)), 
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'time_span_hours': (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 'h')
            }
            
            # Trend analysis (simple linear regression)
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                stats['trend_slope'] = float(slope)
                stats['trend_direction'] = 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
            else:
                stats['trend_slope'] = 0.0
                stats['trend_direction'] = 'stable'
            
            # Alarm state analysis
            alarm_counts = df['alarm_state'].value_counts().to_dict()
            stats['alarm_states'] = alarm_counts
            
            # Alarm transitions
            alarm_transitions = self._find_alarm_transitions(df)
            stats['alarm_transitions'] = alarm_transitions
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute summary stats: {e}")
            return {}
    
    def _find_alarm_transitions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find alarm state transitions (e.g., OK -> High -> HighHigh).
        
        Why this matters:
        - Alarm floods indicate process upsets
        - Transition patterns show failure modes
        - Timing helps assess response requirements
        """
        if len(df) < 2:
            return []
        
        transitions = []
        prev_state = None
        
        for _, row in df.iterrows():
            current_state = row['alarm_state']
            if prev_state is not None and prev_state != current_state:
                transitions.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'from_state': prev_state,
                    'to_state': current_state,
                    'value': row['value']
                })
            prev_state = current_state
        
        return transitions
    
    def format_data_summary(self, stats: Dict[str, Any], tag: str) -> str:
        """
        Format summary statistics into readable text.
        
        Args:
            stats: Summary statistics dictionary
            tag: Process tag name
            
        Returns:
            Formatted summary string
        """
        if not stats:
            return f"No data available for {tag}."
        
        # Build summary text
        summary_parts = [
            f"Process tag {tag} analysis:",
            f"• Data points: {stats['count']} over {stats['time_span_hours']:.1f} hours",
            f"• Value range: {stats['min_value']:.2f} to {stats['max_value']:.2f} (mean: {stats['mean_value']:.2f})",
            f"• Trend: {stats['trend_direction']} (slope: {stats['trend_slope']:.3f})"
        ]
        
        # Add alarm state summary
        alarm_states = stats.get('alarm_states', {})
        if alarm_states:
            state_summary = ", ".join([f"{state}: {count}" for state, count in alarm_states.items()])
            summary_parts.append(f"• Alarm states: {state_summary}")
        
        # Add transition summary
        transitions = stats.get('alarm_transitions', [])
        if transitions:
            summary_parts.append(f"• Alarm transitions: {len(transitions)} state changes detected")
            
            # Highlight critical transitions
            critical_transitions = [t for t in transitions if 'High' in t['to_state']]
            if critical_transitions:
                summary_parts.append(f"• Critical: {len(critical_transitions)} transitions to alarm states")
        
        return "\n".join(summary_parts)
    
    def get_tag_guidance(self, tag: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get document-based guidance for process tag using RAG.
        
        Strategy:
        - Query documents for tag-specific procedures
        - Look for alarm response procedures  
        - Search for troubleshooting guides
        
        Args:
            tag: Process tag name
            stats: Summary statistics to inform queries
            
        Returns:
            RAG response with guidance and citations
        """
        
        # Build context-aware query based on data analysis
        query_parts = [tag]
        
        # Add context based on alarm states
        alarm_states = stats.get('alarm_states', {})
        if 'HighHigh' in alarm_states:
            query_parts.append("high high alarm response procedure")
        elif 'High' in alarm_states:
            query_parts.append("high alarm response procedure") 
        
        # Add context based on trend
        trend = stats.get('trend_direction', 'stable')
        if trend == 'increasing':
            query_parts.append("rising trend troubleshooting")
        elif trend == 'decreasing': 
            query_parts.append("falling trend troubleshooting")
        
        # Construct query
        query = " ".join(query_parts)
        
        logger.info(f"Querying documents for guidance: '{query}'")
        
        # Get guidance from RAG system
        return self.rag.ask(query)
    
    def explain_alarm(self, tag: str, start: str, end: str) -> Dict[str, Any]:
        """
        Main alarm explanation pipeline.
        
        Process:
        1. Load alarm data
        2. Filter by tag and time window
        3. Compute data summary statistics
        4. Get document-based guidance
        5. Combine data insights with procedural guidance
        
        Args:
            tag: Process tag to analyze
            start: Start time (ISO format)
            end: End time (ISO format)  
            
        Returns:
            Combined analysis with data summary and guidance
        """
        
        logger.info(f"Explaining alarm for {tag} from {start} to {end}")
        
        try:
            # Load data
            df = self.load_alarm_data()
            if df.empty:
                return {
                    "summary_from_data": "No alarm data available.",
                    "answer": "Cannot provide guidance without alarm data.",
                    "citations": []
                }
            
            # Filter data  
            filtered_df = self.slice_by_time(df, tag, start, end)
            if filtered_df.empty:
                return {
                    "summary_from_data": f"No data found for {tag} in the specified time range.",
                    "answer": "Cannot analyze - no data available for this tag and time period.",
                    "citations": []
                }
            
            # Compute summary
            stats = self.compute_summary_stats(filtered_df)
            data_summary = self.format_data_summary(stats, tag)
            
            # Get guidance
            guidance = self.get_tag_guidance(tag, stats)
            
            return {
                "summary_from_data": data_summary,
                "answer": guidance.get("answer", "No specific guidance found."),
                "citations": guidance.get("citations", [])
            }
            
        except Exception as e:
            logger.error(f"Alarm explanation failed: {e}")
            return {
                "summary_from_data": f"Error analyzing {tag}: {str(e)}",
                "answer": "Unable to provide guidance due to analysis error.",
                "citations": []
            }

def main():
    """
    Test the alarm analyzer with sample data.
    Run: python -m src.alarms
    """
    print("🔔 Testing Alarm Analyzer")
    print("=" * 50)
    
    analyzer = AlarmAnalyzer()
    
    # Test with sample time windows
    test_cases = [
        {
            "tag": "Temp_101",
            "start": "2024-08-20 15:30:00", 
            "end": "2024-08-20 16:30:00",
            "description": "Temperature trend analysis"
        },
        {
            "tag": "Pressure_202", 
            "start": "2024-08-20 15:00:00",
            "end": "2024-08-20 16:00:00",
            "description": "Pressure spike analysis"  
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Tag: {case['tag']}, Window: {case['start']} to {case['end']}")
        print("-" * 60)
        
        result = analyzer.explain_alarm(case['tag'], case['start'], case['end'])
        
        print("Data Summary:")
        print(result['summary_from_data'])
        
        print(f"\nGuidance:")
        print(result['answer'])
        
        if result['citations']:
            print(f"\nCitations ({len(result['citations'])}):")
            for j, citation in enumerate(result['citations'], 1):
                print(f"  {j}. {citation['title']}, page {citation['page']}")
        
        print()

if __name__ == "__main__":
    main()