import logging
import yaml

class ConfigManager:
    def __init__(self, ConfigPath="ConfigManager.yaml"):
        self.ConfigPath = ConfigPath
        self._LoadConfig()
        logging.getLogger(__name__).info("Loaded configuration from %s", self.ConfigPath)

    def _LoadConfig(self):
        with open(self.ConfigPath, "r") as File:
            self.ConfigData = yaml.safe_load(File)
        logging.getLogger(__name__).info("Config file %s loaded", self.ConfigPath)

    def GetParameter(self, Key, Default=None):
        Value = self.ConfigData.get(Key, Default)
        logging.getLogger(__name__).info("Retrieved parameter %s: %s", Key, Value)
        return Value
