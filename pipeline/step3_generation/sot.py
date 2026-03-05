from langfuse import observe
from utils.utils import safe_format
from llm.base import LLMClient


@observe(name="generate_sot", as_type="generation")
def generate_sot(
    cluster_name: str, paragraphs_dict, llm: LLMClient, prompt: str
) -> str:

    joined = "\n\n".join(
        f"### {axis}\n{paragraph}" for axis, paragraph in paragraphs_dict.items()
    )

    formatted_prompt = safe_format(prompt, cluster_name=cluster_name, joined=joined)

    return llm.complete(formatted_prompt, temperature=0.8)
