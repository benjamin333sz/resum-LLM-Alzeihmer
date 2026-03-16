from entities.paper import Paper
from llm.base import LLMClient
from langfuse import observe
from utils.utils import safe_format, safe_json_load, first_n_sentences, count_tokens
from tqdm import tqdm
import re
import json


@observe(name="filter_subject", as_type="evaluator")
def filter_batch(
    papers: list[Paper],
    llm: LLMClient,
    prompt_template: str,
    subject: str = "Alzheimer",
    batch_size: int = 5,
    max_input_tokens: int = 8000,
    sentance_max: int = 5,
) -> list[Paper]:

    i = 0
    pbar = tqdm(
        total=len(papers),
        desc=f"[Step_1] Filtering papers about {subject}",
        unit="paper",
    )

    while i < len(papers):

        current_batch_size = min(batch_size, len(papers) - i)

        while current_batch_size > 0:

            batch_papers = papers[i : i + current_batch_size]

            articles_json = json.dumps(
                [
                    {
                        "title": first_n_sentences(p.title, sentance_max),
                        "abstract": first_n_sentences(p.abstract, sentance_max),
                    }
                    for p in batch_papers
                ]
            )

            prompt = safe_format(
                prompt_template, subject=subject, articles_json=articles_json
            )

            tokens = count_tokens(prompt)

            if tokens <= max_input_tokens:
                break

            current_batch_size -= 1

        if current_batch_size == 0:
            raise RuntimeError("Single article exceeds token limit.")

        response = llm.complete(prompt)

        try:
            results = safe_json_load(response)
        except Exception:
            results = {
                str(idx + 1): "YES" if "yes" in response.lower() else "NO"
                for idx in range(len(batch_papers))
            }

        for idx, paper in enumerate(batch_papers, 1):
            paper.is_about_subject = results.get(str(idx), "NO") == "YES"

        i += current_batch_size
        pbar.update(current_batch_size)

    pbar.close()

    return papers
