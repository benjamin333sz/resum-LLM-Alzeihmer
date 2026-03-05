import re
from langfuse import observe
from typing import Any
import json


@observe(name="clean_bib_token", as_type="tool")
def clean_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@observe(name="safe_json_load", as_type="tool")
def safe_json_load(output: str):
    if not output:
        return {}
    start = output.find("{")
    end = output.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return {}


@observe(name="safe_format", as_type="tool")
def safe_format(prompt: str, **kwargs):
    try:
        return prompt.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing placeholder in prompt: {e}")


@observe(name="format_user_modalities", as_type="tool")
def format_user_modalities(modalities: dict) -> dict:
    """
    Vérifie et formate les modalités sous le format :

    USER_MODALITIES = {
        "Modality_A": ("description"),
        ...
    }
    """

    if not isinstance(modalities, dict):
        raise ValueError("Modalities must be a dict")

    formatted = {}

    for key, value in modalities.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Invalid modality key: {key}")

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid description for modality: {key}")

        formatted[key] = (value.strip(),)

    if len(formatted) < 3:
        raise ValueError("Too few valid modalities")

    return formatted
