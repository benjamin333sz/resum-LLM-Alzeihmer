import json
import re
from tqdm import tqdm
from entities.paper import Paper
from llm.base import LLMClient
from langfuse import observe
from utils.utils import safe_format
from collections import defaultdict


@observe(name="clustering", as_type="generation")
def clustering(
    papers: list[Paper],
    llm: LLMClient,
    prompts: dict[str, str],
    user_modalities: dict[str, str],
):

    clusters = defaultdict(list)

    prompt_template = prompts["clustering"]

    for paper in tqdm(papers, desc="[Step_2] Attribution clusters following modalities"):

        modalities_block = "\n".join(f"{k}: {v}" for k, v in user_modalities.items())

        prompt = safe_format(
            prompt_template,
            title=paper.title,
            abstract=paper.abstract,
            modalities_block=modalities_block,
        )

        response = llm.complete(prompt, temperature=0.1)
        cleaned = re.sub(r"```json|```", "", response).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:
            clusters["UNRESOLVED"].append(paper)
            continue

        action = parsed.get("action")
        modality = parsed.get("modality_id")

        if action == "USE_EXISTING" and modality in user_modalities:
            clusters[modality].append(paper)
        else:
            clusters["UNRESOLVED"].append(paper)

    return clusters
