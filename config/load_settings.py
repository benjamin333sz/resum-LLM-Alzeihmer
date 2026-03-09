import yaml
from pathlib import Path
from langfuse import observe


@observe(name="load_settings", as_type="tool")
def load_settings(path: str = "config/settings.yaml") -> dict:
    """
    Load project settings from a YAML file.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
