import yaml

class ConfigManager:
    def __init__(self, ConfigPath="ConfigManager.yaml"):
        self.ConfigPath = ConfigPath
        self._LoadConfig()

    def _LoadConfig(self):
        with open(self.ConfigPath, "r") as File:
            self.ConfigData = yaml.safe_load(File)

    def GetParameter(self, Key, Default=None):
        return self.ConfigData.get(Key, Default)
