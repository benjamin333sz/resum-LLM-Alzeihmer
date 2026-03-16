import json
import re
from tqdm import tqdm
from entities.paper import Paper
from llm.base import LLMClient
from langfuse import observe
from utils.utils import safe_format, safe_json_load, count_tokens
from collections import defaultdict


@observe(name="clustering_batch", as_type="generation")
def clustering_batch(
    papers: list[Paper],
    llm: LLMClient,
    prompt_template: str,
    user_modalities: dict[str, str],
    batch_size_start: int = 5,
    max_input_tokens: int = 8000,
):
    clusters = defaultdict(list)
    modalities_json = json.dumps(user_modalities)
    i = 0
    pbar = tqdm(total=len(papers), desc="[Step_2] Clustering batch", unit="paper")

    while i < len(papers):
        current_batch_size = min(batch_size_start, len(papers) - i)

        while current_batch_size > 0:
            batch_papers = papers[i : i + current_batch_size]
            articles_json = json.dumps(
                [{"title": p.title, "abstract": p.abstract} for p in batch_papers]
            )
            prompt = safe_format(
                prompt_template,
                modalities_json=modalities_json,
                articles_json=articles_json,
            )

            tokens = count_tokens(prompt)
            if tokens <= max_input_tokens:
                break
            current_batch_size -= 1

        if current_batch_size == 0:
            raise RuntimeError("Single article exceeds token limit.")

        response = llm.complete(prompt, temperature=0.1)
        results = safe_json_load(response)

        for idx, paper in enumerate(batch_papers, 1):
            item = results.get(str(idx), {"action": "UNRESOLVED"})
            action = item.get("action")
            modality = item.get("modality_id")
            if action == "USE_EXISTING" and modality in user_modalities:
                clusters[modality].append(paper)
            else:
                clusters["UNRESOLVED"].append(paper)
        i += current_batch_size
        pbar.update(current_batch_size)

    pbar.close()
    return clusters
