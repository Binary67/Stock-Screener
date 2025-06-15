import logging
import os
from datetime import datetime
from glob import glob

class LoggingManager:
    def __init__(self, LogDir="Logs", LogPrefix="application", BackupCount=5):
        self.LogDir = LogDir
        self.LogPrefix = LogPrefix
        self.BackupCount = BackupCount
        os.makedirs(self.LogDir, exist_ok=True)
        UniqueStamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        LogFileName = f"{self.LogPrefix}_{UniqueStamp}.log"
        LogFilePath = os.path.join(self.LogDir, LogFileName)

        Handler = logging.FileHandler(LogFilePath)
        Formatter = logging.Formatter(
            "%(asctime)s [%(module)s.%(funcName)s] %(levelname)s: %(message)s"
        )
        Handler.setFormatter(Formatter)

        self.Logger = logging.getLogger()
        self.Logger.setLevel(logging.INFO)

        for Existing in list(self.Logger.handlers):
            self.Logger.removeHandler(Existing)
        self.Logger.addHandler(Handler)
        self._CleanOldLogs()

    def _CleanOldLogs(self):
        Pattern = os.path.join(self.LogDir, f"{self.LogPrefix}_*.log")
        LogFiles = sorted(glob(Pattern), key=os.path.getmtime, reverse=True)
        for OldFile in LogFiles[self.BackupCount:]:
            try:
                os.remove(OldFile)
            except FileNotFoundError:
                pass
