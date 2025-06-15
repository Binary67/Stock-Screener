import logging
from LoggingManager import LoggingManager


def test_logging_rotation(tmp_path):
    LogDir = tmp_path / "Logs"
    for i in range(7):
        LoggingManager(LogDir=str(LogDir), LogPrefix="test", BackupCount=5)
        logging.getLogger().info("Entry %d", i)
    LogFiles = list(LogDir.glob("test_*.log"))
    assert len(LogFiles) == 5
