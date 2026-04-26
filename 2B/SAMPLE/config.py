"""
Configuration for TBRGS System
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    sequence_length: int = 24  # 6 hours of 15-min intervals
    prediction_horizon: int = 1  # Predict next 15 minutes
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    validation_split: float = 0.2

@dataclass
class TrafficConfig:
    """Traffic flow to travel time conversion"""
    speed_limit_kmh: float = 60.0
    intersection_delay_seconds: float = 30.0

@dataclass
class GraphConfig:
    """Graph/road network configuration"""
    # Boroondara area approximate bounds (for graph construction)
    BOROONDARA_SITES: List[str] = None  # Will be populated from metadata

    def __post_init__(self):
        # Key SCATS sites in Boroondara area from the metadata
        self.BOROONDARA_SITES = [
            '0970', '2000', '2200', '2820', '2825', '2827', '2846',
            '3001', '3002', '3120', '3122', '3126', '3127', '3180',
            '3662', '3682', '3685', '3804', '3812', '4030', '4032',
            '4034', '4035', '4040', '4043', '4051', '4057', '4063',
            '4262', '4263', '4264', '4266', '4270', '4272', '4273',
            '4321', '4324', '4335', '4812', '4821'
        ]

class PathConfig:
    """Path finding configuration"""
    TOP_K_PATHS: int = 5
    DEFAULT_ORIGIN: str = '2000'  # WARRIGAL_RD/TOORAK_RD
    DEFAULT_DESTINATION: str = '3002'  # DENMARK_ST/BARKERS_RD

# Data paths
DATA_DIR = 'processed_data'
SCATS_METADATA_FILE = os.path.join(DATA_DIR, 'scats_metadata.csv')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Ensure directories exist
for dir_path in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
