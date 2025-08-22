"""
Tests for alarm analysis module
What to learn here: Testing time-series data processing, pandas operations,
statistical analysis functions, and integration with RAG systems.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from src.alarms import AlarmAnalyzer
from src.config import Config


class TestAlarmAnalyzer:
    """Test AlarmAnalyzer functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.analyzer = AlarmAnalyzer()
        
        # Create sample test data
        self.sample_data = self._create_test_data()
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create sample alarm data for testing"""
        timestamps = pd.date_range('2024-08-20 14:00:00', periods=60, freq='1T')
        data = []
        
        # Temperature data with rising trend and alarms
        for i, ts in enumerate(timestamps):
            temp_value = 85 + (i * 0.5)  # Rising temperature
            if temp_value < 100:
                temp_alarm = "OK"
            elif temp_value < 110:
                temp_alarm = "High"
            else:
                temp_alarm = "HighHigh"
            
            data.append({
                'timestamp': ts,
                'tag': 'Temp_101',
                'value': temp_value,
                'alarm_state': temp_alarm
            })
            
            # Pressure data - stable with one spike
            if 20 <= i <= 25:  # Spike between minutes 20-25
                pressure_value = 60.0  # High pressure spike
                pressure_alarm = "High"
            else:
                pressure_value = 45.0  # Normal pressure
                pressure_alarm = "OK"
            
            data.append({
                'timestamp': ts,
                'tag': 'Pressure_202', 
                'value': pressure_value,
                'alarm_state': pressure_alarm
            })
        
        return pd.DataFrame(data)
    
    def test_load_alarm_data_success(self):
        """Test successful alarm data loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            csv_path = Path(f.name)
        
        try:
            df = self.analyzer.load_alarm_data(csv_path)
            
            assert not df.empty
            assert len(df) == len(self.sample_data)
            assert 'timestamp' in df.columns
            assert 'tag' in df.columns
            assert 'value' in df.columns
            assert 'alarm_state' in df.columns
            
            # Check timestamp conversion
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
            
        finally:
            csv_path.unlink()  # Clean up temp file
    
    def test_load_alarm_data_file_not_found(self):
        """Test handling of missing alarm data file"""
        nonexistent_path = Path("nonexistent_file.csv")
        df = self.analyzer.load_alarm_data(nonexistent_path)
        
        assert df.empty
    
    def test_slice_by_time(self):
        """Test time-based data filtering"""
        # Filter temperature data for a specific window
        start_time = "2024-08-20T14:10:00"
        end_time = "2024-08-20T14:20:00"
        
        filtered_df = self.analyzer.slice_by_time(
            self.sample_data, 
            "Temp_101", 
            start_time, 
            end_time
        )
        
        assert not filtered_df.empty
        assert (filtered_df['tag'] == 'Temp_101').all()
        
        # Check time bounds
        assert filtered_df['timestamp'].min() >= pd.to_datetime(start_time)
        assert filtered_df['timestamp'].max() <= pd.to_datetime(end_time)
    
    def test_slice_by_time_no_data(self):
        """Test time filtering with no matching data"""
        # Use future time range
        start_time = "2025-08-20T14:00:00"
        end_time = "2025-08-20T15:00:00"
        
        filtered_df = self.analyzer.slice_by_time(
            self.sample_data,
            "Temp_101",
            start_time,
            end_time
        )
        
        assert filtered_df.empty
    
    def test_compute_summary_stats_basic(self):
        """Test basic statistical calculations"""
        # Filter temperature data 
        temp_data = self.sample_data[self.sample_data['tag'] == 'Temp_101'].copy()
        
        stats = self.analyzer.compute_summary_stats(temp_data)
        
        # Check required fields
        assert 'count' in stats
        assert 'min_value' in stats
        assert 'max_value' in stats
        assert 'mean_value' in stats
        assert 'trend_slope' in stats
        assert 'trend_direction' in stats
        assert 'alarm_states' in stats
        
        # Validate calculations
        assert stats['count'] == len(temp_data)
        assert stats['min_value'] == temp_data['value'].min()
        assert stats['max_value'] == temp_data['value'].max()
        assert abs(stats['mean_value'] - temp_data['value'].mean()) < 0.01
        
        # Check trend direction (should be increasing for rising temp)
        assert stats['trend_direction'] == 'increasing'
        assert stats['trend_slope'] > 0
    
    def test_compute_summary_stats_empty(self):
        """Test summary stats with empty data"""
        empty_df = pd.DataFrame()
        stats = self.analyzer.compute_summary_stats(empty_df)
        
        assert stats == {}
    
    def test_find_alarm_transitions(self):
        """Test alarm state transition detection"""
        # Create data with known transitions
        test_data = pd.DataFrame([
            {'timestamp': pd.Timestamp('2024-08-20 14:00'), 'alarm_state': 'OK', 'value': 85},
            {'timestamp': pd.Timestamp('2024-08-20 14:01'), 'alarm_state': 'OK', 'value': 90},
            {'timestamp': pd.Timestamp('2024-08-20 14:02'), 'alarm_state': 'High', 'value': 105},
            {'timestamp': pd.Timestamp('2024-08-20 14:03'), 'alarm_state': 'HighHigh', 'value': 115},
            {'timestamp': pd.Timestamp('2024-08-20 14:04'), 'alarm_state': 'HighHigh', 'value': 120}
        ])
        
        transitions = self.analyzer._find_alarm_transitions(test_data)
        
        assert len(transitions) == 2  # OK->High, High->HighHigh
        assert transitions[0]['from_state'] == 'OK'
        assert transitions[0]['to_state'] == 'High'
        assert transitions[1]['from_state'] == 'High'
        assert transitions[1]['to_state'] == 'HighHigh'
    
    def test_format_data_summary(self):
        """Test data summary formatting"""
        sample_stats = {
            'count': 60,
            'min_value': 85.0,
            'max_value': 115.0,
            'mean_value': 100.0,
            'trend_direction': 'increasing',
            'trend_slope': 0.5,
            'time_span_hours': 1.0,
            'alarm_states': {'OK': 30, 'High': 20, 'HighHigh': 10},
            'alarm_transitions': [
                {'from_state': 'OK', 'to_state': 'High'},
                {'from_state': 'High', 'to_state': 'HighHigh'}
            ]
        }
        
        summary = self.analyzer.format_data_summary(sample_stats, "Temp_101")
        
        assert "Temp_101" in summary
        assert "60" in summary  # Count
        assert "85.00 to 115.00" in summary  # Range
        assert "increasing" in summary  # Trend
        assert "OK: 30" in summary  # Alarm states
        assert "2 state changes" in summary  # Transitions
    
    def test_format_data_summary_empty(self):
        """Test summary formatting with empty stats"""
        summary = self.analyzer.format_data_summary({}, "Test_Tag")
        
        assert "No data available for Test_Tag" in summary
    
    @patch('src.alarms.RAGSystem')
    def test_get_tag_guidance(self, mock_rag_system):
        """Test document guidance retrieval"""
        # Mock RAG response
        mock_rag_instance = MagicMock()
        mock_rag_instance.ask.return_value = {
            'answer': 'For high temperature alarms, check cooling system.',
            'citations': [{'title': 'Safety Manual', 'page': 15, 'score': 0.85}]
        }
        mock_rag_system.return_value = mock_rag_instance
        
        analyzer = AlarmAnalyzer()
        
        # Test with alarm states in stats
        stats = {
            'alarm_states': {'HighHigh': 5},
            'trend_direction': 'increasing'
        }
        
        result = analyzer.get_tag_guidance("Temp_101", stats)
        
        assert 'answer' in result
        assert 'citations' in result
        assert 'high temperature alarms' in result['answer'].lower()
        
        # Verify RAG was called with appropriate query
        mock_rag_instance.ask.assert_called_once()
        called_query = mock_rag_instance.ask.call_args[0][0]
        assert 'Temp_101' in called_query
        assert 'high high alarm' in called_query.lower()
    
    @patch('src.alarms.RAGSystem')
    def test_explain_alarm_complete_flow(self, mock_rag_system):
        """Test complete alarm explanation pipeline"""
        # Mock RAG response
        mock_rag_instance = MagicMock()
        mock_rag_instance.ask.return_value = {
            'answer': 'Reduce feed rate and check cooling system.',
            'citations': [{'title': 'Operations Manual', 'page': 23, 'score': 0.78}]
        }
        mock_rag_system.return_value = mock_rag_instance
        
        # Mock data loading
        with patch.object(self.analyzer, 'load_alarm_data', return_value=self.sample_data):
            result = self.analyzer.explain_alarm(
                "Temp_101",
                "2024-08-20 14:00:00", 
                "2024-08-20 14:30:00"
            )
        
        assert 'summary_from_data' in result
        assert 'answer' in result
        assert 'citations' in result
        
        # Check data summary content
        assert 'Temp_101' in result['summary_from_data']
        assert 'Data points:' in result['summary_from_data']
        
        # Check guidance
        assert 'feed rate' in result['answer']
        assert len(result['citations']) == 1
    
    @patch('src.alarms.RAGSystem')
    def test_explain_alarm_no_data(self, mock_rag_system):
        """Test alarm explanation with no available data"""
        mock_rag_instance = MagicMock()
        mock_rag_system.return_value = mock_rag_instance
        
        # Mock empty data loading
        with patch.object(self.analyzer, 'load_alarm_data', return_value=pd.DataFrame()):
            result = self.analyzer.explain_alarm(
                "Nonexistent_Tag",
                "2024-08-20 14:00:00",
                "2024-08-20 15:00:00" 
            )
        
        assert "No alarm data available" in result['summary_from_data']
        assert "Cannot provide guidance" in result['answer']
        assert len(result['citations']) == 0
    
    def test_explain_alarm_no_matching_data(self):
        """Test alarm explanation with no matching time/tag data"""
        # Mock data loading but no matching slice
        with patch.object(self.analyzer, 'load_alarm_data', return_value=self.sample_data):
            with patch.object(self.analyzer, 'slice_by_time', return_value=pd.DataFrame()):
                result = self.analyzer.explain_alarm(
                    "Unknown_Tag",
                    "2024-08-20 14:00:00",
                    "2024-08-20 15:00:00"
                )
        
        assert "No data found for Unknown_Tag" in result['summary_from_data']
        assert "Cannot analyze" in result['answer']


if __name__ == "__main__":
    pytest.main([__file__])