import json
import subprocess
import platform
from langfuse import observe
from entities.paper import Paper
from typing import Any
from datetime import datetime
from pathlib import Path


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


@observe(name="save_results", as_type="tool")
def save_results(latex: str, subject: str, pdf: bool = True):
    day = datetime.now().strftime("%d_%m_%Y")
    filename = f"State_of_the_art_{subject}_{day}"
    tex_path = f"results/{filename}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    if pdf:
        compile_pdf_latex(tex_path, filename)


@observe(name="compile_pdf_Latex", as_type="tool")
def compile_pdf_latex(tex_path: str, filename: str):
    project_root = Path(__file__).resolve().parent
    if platform.system() == "Windows":
        tectonic_path = project_root / "tectonic" / "Windows" / "tectonic.exe"
    else:
        tectonic_path = project_root / "tectonic" / "Linux_MacOS" / "tectonic"
    result = subprocess.run(
        [str(tectonic_path), "--outdir", "results", tex_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"PDF generated: results/{filename}.pdf")
    else:
        print("Error compilation:")
        print(result.stderr)


@observe(name="load_papers", as_type="tool")
def load_papers(path: str) -> list[Paper]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


@observe(name="load_prompt", as_type="tool")
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
