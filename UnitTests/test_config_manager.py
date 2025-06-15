import os
import yaml
from ConfigManager import ConfigManager

def test_config_loading(tmp_path):
    ConfigFile = tmp_path / "TestConfig.yaml"
    Content = {
        "Ticker": "MSFT",
        "StartDate": "2023-01-01",
        "EndDate": "2023-01-10",
        "Interval": "1d",
        "CacheDir": "Cache"
    }
    with open(ConfigFile, "w") as File:
        yaml.dump(Content, File)

    Manager = ConfigManager(ConfigPath=str(ConfigFile))
    for Key, Value in Content.items():
        assert Manager.GetParameter(Key) == Value
