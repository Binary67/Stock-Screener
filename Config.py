"""YAML-backed configuration loader for the stock-screener pipeline.

Per AGENTS Coding Contract:
- Keep `main.py` as an orchestrator and move config logic here.
- Use PascalCase for newly declared variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml as YamlLib  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - graceful fallback if dependency missing
    YamlLib = None  # type: ignore[assignment]


# Default location next to repo root `main.py`
DefaultConfigPath = Path(__file__).resolve().parent / "config.yaml"


def LoadConfig(ConfigPath: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    ConfigPath: str | Path | None
        Optional path to a YAML file. Defaults to repository-level `config.yaml`.

    Returns
    -------
    dict
        A dictionary of configuration values from YAML.
    """
    TargetPath = Path(ConfigPath) if ConfigPath is not None else DefaultConfigPath

    if not TargetPath.exists():
        raise FileNotFoundError(
            f"Config file not found at {TargetPath}. Create it or supply a path to LoadConfig()."
        )

    if YamlLib is None:
        raise ImportError(
            "PyYAML is required to read YAML config. Install with `pip install pyyaml`."
        )

    with TargetPath.open("r", encoding="utf-8") as File:
        Data = YamlLib.safe_load(File) or {}

    if not isinstance(Data, dict):
        raise ValueError("Top-level YAML content must be a mapping/dict.")

    return Data

