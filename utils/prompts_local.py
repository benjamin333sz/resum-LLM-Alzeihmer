from pathlib import Path
from langfuse import observe


@observe(name="load_local_prompt", as_type="tool")
def load_local_prompt(path: str) -> str:
    """
    Load a prompt template from a local text file.

    Args:
        path (str): Path to the prompt file.
    Returns:
        str: Prompt template as a string.
    """
    return Path(path).read_text(encoding="utf-8")
