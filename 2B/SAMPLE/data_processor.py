"""
Traffic Data Processor for TBRGS Assignment
Processes the VicRoads SCATS data for traffic flow prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TrafficDataProcessor:
    def __init__(self, excel_file_path):
        """
        Initialize the traffic data processor
        
        Args:
            excel_file_path (str): Path to the SCATS data Excel file
        """
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.processed_data = None
        self.time_series_data = {}  # Dictionary to store time series for each SCATS site
        
    def load_data(self):
        """
        Load and parse the Excel file containing traffic data
        """
        print("Loading traffic data from Excel file...")
        
        # Read the Excel file
        self.raw_data = pd.read_excel(self.excel_file_path, sheet_name='Data')
        
        # Skip header rows (first 20 rows contain metadata)
        # Actual data starts from row 20 (0-indexed)
        self.data_rows = self.raw_data.iloc[20:].copy()
        
        # Rename columns for clarity
        # First 9 columns are metadata, then we have time columns
        new_column_names = [
            'SCATS_Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 
            'NB_LONGITUDE', 'HF_VicRoads_Internal', 'VR_Internal_Stat', 
            'VR_Internal_Loc', 'NB_TYPE_SURVEY', 'Start_Time'
        ]
        
        # Add time column names (96 time intervals per day, 15-min intervals)
        time_columns = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                time_str = f"{hour:02d}:{minute:02d}:00"
                time_columns.append(time_str)
        
        # Combine metadata columns with time columns
        expected_columns = new_column_names + time_columns
        
        # Ensure we have the right number of columns
        if len(self.data_rows.columns) >= len(expected_columns):
            self.data_rows.columns = expected_columns[:len(self.data_rows.columns)]
        else:
            # If we don't have enough columns, use what we have
            self.data_rows.columns = list(self.data_rows.columns)
            
        print(f"Loaded {len(self.data_rows)} data rows")
        print(f"Columns: {list(self.data_rows.columns)}")
        
        return self.data_rows
    
    def parse_timestamps(self):
        """
        Parse the Start_Time column to extract date information
        """
        print("Parsing timestamps...")
        
        # Convert Start_Time to datetime
        self.data_rows['Start_Time'] = pd.to_datetime(self.data_rows['Start_Time'])
        
        # Extract date components
        self.data_rows['Date'] = self.data_rows['Start_Time'].dt.date
        self.data_rows['Time_of_Day'] = self.data_rows['Start_Time'].dt.time
        
        print(f"Date range: {self.data_rows['Date'].min()} to {self.data_rows['Date'].max()}")
        print(f"Unique dates: {self.data_rows['Date'].nunique()}")
        
        return self.data_rows
    
    def extract_time_series(self, scats_id=None):
        """
        Extract time series data for a specific SCATS site or all sites
        
        Args:
            scats_id (str, optional): Specific SCATS site ID. If None, process all sites.
            
        Returns:
            dict: Dictionary containing time series data
        """
        print("Extracting time series data...")
        
        # Filter data if specific SCATS ID is requested
        if scats_id:
            filtered_data = self.data_rows[self.data_rows['SCATS_Number'] == str(scats_id)].copy()
            print(f"Extracting data for SCATS ID: {scats_id}")
            print(f"Found {len(filtered_data)} records")
        else:
            filtered_data = self.data_rows.copy()
            print(f"Extracting data for all SCATS sites")
            print(f"Found {len(filtered_data)} total records")
        
        # Get time columns (excluding metadata columns)
        time_cols = [col for col in filtered_data.columns if col not in 
                    ['SCATS_Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 
                     'NB_LONGITUDE', 'HF_VicRoads_Internal', 'VR_Internal_Stat', 
                     'VR_Internal_Loc', 'NB_TYPE_SURVEY', 'Start_Time', 'Date', 'Time_of_Day']]
        
        # Sort time columns chronologically
        time_cols_sorted = sorted(time_cols, key=lambda x: datetime.strptime(x, '%H:%M:%S'))
        
        # Process each unique SCATS site
        unique_scats = filtered_data['SCATS_Number'].unique()
        
        for scat in unique_scats:
            site_data = filtered_data[filtered_data['SCATS_Number'] == scat].copy()
            
            # Sort by date and time to ensure chronological order
            site_data = site_data.sort_values(['Date', 'Start_Time'])
            
            # Extract time series values
            time_series = []
            timestamps = []
            
            for _, row in site_data.iterrows():
                # Get values for all time intervals in this day
                day_values = [row[col] for col in time_cols_sorted if pd.notnull(row[col])]
                time_series.extend(day_values)
                
                # Create timestamps for each 15-minute interval
                base_date = row['Date']
                for i, time_str in enumerate(time_cols_sorted):
                    if pd.notnull(row[time_str]):
                        hour, minute, second = map(int, time_str.split(':'))
                        timestamp = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute, seconds=second)
                        timestamps.append(timestamp)
            
            # Store the time series data
            self.time_series_data[scat] = {
                'values': np.array(time_series, dtype=float),
                'timestamps': timestamps,
                'location': site_data.iloc[0]['Location'] if len(site_data) > 0 else 'Unknown',
                'latitude': site_data.iloc[0]['NB_LATITUDE'] if len(site_data) > 0 else None,
                'longitude': site_data.iloc[0]['NB_LONGITUDE'] if len(site_data) > 0 else None
            }
            
            print(f"SCATS {scat}: {len(time_series)} data points from {timestamps[0] if timestamps else 'N/A'} to {timestamps[-1] if timestamps else 'N/A'}")
        
        return self.time_series_data
    
    def get_training_data(self, scats_id, sequence_length=24, prediction_horizon=1):
        """
        Prepare training data for ML models
        
        Args:
            scats_id (str): SCATS site ID
            sequence_length (int): Number of past time steps to use for prediction
            prediction_horizon (int): Number of future time steps to predict
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        if scats_id not in self.time_series_data:
            raise ValueError(f"SCATS ID {scats_id} not found in processed data")
        
        data = self.time_series_data[scats_id]['values']
        
        # Remove any NaN values
        data = data[~np.isnan(data)]
        
        X, y = [], []
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[(i + sequence_length):(i + sequence_length + prediction_horizon)])
        
        return np.array(X), np.array(y)
    
    def save_processed_data(self, output_dir='processed_data'):
        """
        Save processed data to files for later use
        
        Args:
            output_dir (str): Directory to save processed data
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save time series data for each SCATS site
        for scat_id, data in self.time_series_data.items():
            # Save as CSV
            df = pd.DataFrame({
                'timestamp': data['timestamps'],
                'traffic_flow': data['values']
            })
            df.to_csv(os.path.join(output_dir, f'scats_{scat_id}_timeseries.csv'), index=False)
        
        # Save metadata
        metadata = []
        for scat_id, data in self.time_series_data.items():
            metadata.append({
                'SCATS_ID': scat_id,
                'Location': data['location'],
                'Latitude': data['latitude'],
                'Longitude': data['longitude'],
                'Data_Points': len(data['values']),
                'Start_Time': data['timestamps'][0] if data['timestamps'] else None,
                'End_Time': data['timestamps'][-1] if data['timestamps'] else None
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'scats_metadata.csv'), index=False)
        
        print(f"Processed data saved to {output_dir}/")
    
    def get_summary_stats(self):
        """
        Get summary statistics of the processed data
        
        Returns:
            dict: Summary statistics
        """
        stats = {}
        for scat_id, data in self.time_series_data.items():
            values = data['values']
            stats[scat_id] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'location': data['location']
            }
        return stats

def main():
    """
    Main function to demonstrate the data processor
    """
    # Initialize processor
    processor = TrafficDataProcessor('Scats Data October 2006.xls')
    
    # Load and process data
    processor.load_data()
    processor.parse_timestamps()
    time_series_data = processor.extract_time_series()
    
    # Show summary statistics
    stats = processor.get_summary_stats()
    print("\nSummary Statistics:")
    for scat_id, stat in list(stats.items())[:5]:  # Show first 5 sites
        print(f"SCATS {scat_id} ({stat['location']}):")
        print(f"  Data points: {stat['count']}")
        print(f"  Mean traffic flow: {stat['mean']:.2f}")
        print(f"  Std dev: {stat['std']:.2f}")
        print(f"  Range: [{stat['min']:.2f}, {stat['max']:.2f}]")
        print()
    
    # Save processed data
    processor.save_processed_data()
    
    # Example: Get training data for a specific site
    if '0970' in processor.time_series_data:
        X, y = processor.get_training_data('0970', sequence_length=24, prediction_horizon=1)
        print(f"Training data shape for SCATS 0970: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    main()