from langfuse import observe
from utils.utils import safe_format
from llm.base import LLMClient


@observe(name="generate_axis_paragraph", as_type="generation")
def generate_paragraph(
    cluster_name: str,
    axis: str,
    summaries: list[str],
    arxiv_ids: list[str],
    llm: LLMClient,
    prompt: str,
) -> str:

    summaries_block = "\n\n".join(
        f"[{arxiv_ids[i]}]\n{summaries[i]}" for i in range(len(summaries))
    )

    allowed_citations_safe = ", ".join(
        [c.replace("{", "{{").replace("}", "}}") for c in arxiv_ids]
    )

    formatted_prompt = safe_format(
        prompt,
        cluster_name=cluster_name,
        axis=axis,
        summaries_block=summaries_block,
        allowed_citations=allowed_citations_safe,
    )

    return llm.complete(formatted_prompt, temperature=0.1)
