import pandas as pd
from typing import List, Optional
from data_sources import DataSourceAdapter
from config import Configuration

class DataProcessor:
    """
    Processes raw business data for revenue analysis.
    Implements type hinting and error handling for robustness.
    """

    def __init__(self):
        self.config = Configuration().get_config()
        self.data_sources = [DataSourceAdapter(source) for source in self.config['data_sources']]

    def fetch_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """
        Fetches data from a specified source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            for ds in self.data_sources:
                if ds.name == source_name:
                    return ds.get_data()
            raise ValueError(f"Source {source_name} not found.")
        except Exception as e:
            logging.error(f"Failed to fetch data from {source_name}: {str(e)}")
            return None

    def process_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Processes multiple DataFrames into a unified DataFrame.
        
        Args:
            dataframes: List of DataFrames
            
        Returns:
            Processed DataFrame
        """
        try:
            combined_df = pd.concat(dataframes)
            # Basic cleaning
            combined_df.dropna(inplace=True)
            combined_df['Revenue'] = pd.to_numeric(combined_df['Revenue'], errors='coerce')
            
            return combined_df
        except Exception as e:
            logging.error(f"Data processing failed: {str(e)}")
            return None