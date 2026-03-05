import json
import re
from langfuse import observe
from utils.utils import safe_format
from llm.base import LLMClient
from utils.io import save_json
from utils.utils import format_user_modalities


@observe(name="creation_modality", as_type="generation")
def creation_modality(
    subject: str,
    llm: LLMClient,
    prompts: dict[str, str],
):

    creation_prompt = prompts["creation_modality"]

    prompt = safe_format(creation_prompt, subject=subject)

    response = llm.complete(prompt, temperature=0.1)
    cleaned = re.sub(r"```json|```", "", response).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        raise ValueError("Invalid JSON returned during modality creation")

    if "modalities" not in parsed or not isinstance(parsed["modalities"], list):
        raise ValueError("Invalid modality structure")

    modalities = {}

    for item in parsed["modalities"]:
        modality_id = item.get("modality_id")
        description = item.get("description")

        if not modality_id or not description:
            continue

        modalities[modality_id] = description

    modalities = format_user_modalities(modalities)

    with open("data/artifacts/modalities_create.json", "w", encoding="utf-8") as f:
        json.dump(modalities, f, ensure_ascii=False, indent=2)

    return modalities
