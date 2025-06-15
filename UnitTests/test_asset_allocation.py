import os
import sys
import yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from AssetAllocation import AssetAllocation
from ConfigManager import ConfigManager


def test_generate_allocations(tmp_path):
    Ranking = pd.DataFrame({'CompositeRank': [1, 2, 3]}, index=['A', 'B', 'C'])
    ConfigFile = tmp_path / 'config.yaml'
    with open(ConfigFile, 'w') as File:
        yaml.dump({'TopN': 50, 'Alpha': 1}, File)
    Config = ConfigManager(ConfigPath=str(ConfigFile))
    Allocation = AssetAllocation(Ranking, Config).GenerateAllocations()
    Weights = np.exp(-1 * np.arange(2))
    Weights /= Weights.sum()
    Expected = pd.DataFrame({'Allocation': Weights}, index=['A', 'B'])
    pd.testing.assert_frame_equal(Allocation, Expected)
