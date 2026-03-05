import json
from typing import List
from langfuse import observe
from entities.paper import Paper
from typing import Any


@observe(name="pipeline_crash", as_type="span")
def log_crash(error: str):
    return {"error": error}


@observe(name="save_json", as_type="tool")
def save_json(objects: list[Paper], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([o.to_dict() for o in objects], f, indent=2, ensure_ascii=False)


@observe(name="save_raw_json", as_type="tool")
def save_raw_json(data: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@observe(name="load_papers", as_type="tool")
def load_papers(path: str) -> list[Paper]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


@observe(name="load_prompt", as_type="tool")
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
