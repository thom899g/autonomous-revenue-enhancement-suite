import logging
from typing import Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('revenue Enhancement.log'),
        logging.StreamHandler()
    ]
)

class Configuration:
    """
    Manages configuration parameters for the Revenue Enhancement Suite.
    Implements singleton pattern to ensure global consistency.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.configurations = {
            'data_sources': ['CRM', 'Financials', 'Inventory'],
            'optimization_models': ['price_prediction', 'customer_segmentation'],
            'logging_level': logging.INFO,
            'api_keys': {}
        }

    def get_config(self) -> Dict[str, Any]:
        return self.configurations

    def update_config(self, updates: Dict[str, Any]) -> None:
        self.configurations.update(updates)