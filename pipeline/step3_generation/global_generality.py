from langfuse import observe
from utils.utils import safe_format
from llm.base import LLMClient
import json
from utils.utils import safe_json_load


@observe(name="conclusion", as_type="generation")
def conclusion(
    prompt: str,
    llm: LLMClient,
    subject: str,
    joined: list[str],
) -> str:

    formatted_prompt = safe_format(prompt, subject=subject, joined=joined)

    response = llm.complete(formatted_prompt, temperature=0.8)

    return response


@observe(name="introduction", as_type="generation")
def introduction(
    prompt: str,
    llm: LLMClient,
    subject: str,
    joined: list[str],
) -> str:

    formatted_prompt = safe_format(prompt, subject=subject, joined=joined)

    response = llm.complete(formatted_prompt, temperature=0.8)

    return response
