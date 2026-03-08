from langfuse import observe
from utils.utils import safe_format
from llm.base import LLMClient
import json
from utils.utils import safe_json_load


@observe(name="extract_axes", as_type="generation")
def extract_axes(
    cluster_name: str, summaries: list[str], llm: LLMClient, prompt: str
) -> list[str]:

    summaries_block = "\n\n".join(
        f"[Paper {i+1}]\n{summary}" for i, summary in enumerate(summaries)
    )

    formatted_prompt = safe_format(
        prompt, cluster_name=cluster_name, summaries_block=summaries_block
    )

    response = llm.complete(formatted_prompt, temperature=0.1)

    data = safe_json_load(response)
    axes = data.get("axes")

    if not isinstance(axes, list):
        raise ValueError(
            f"Invalid axes format: {data}. Format output is maybe not a json"
        )

    return axes
