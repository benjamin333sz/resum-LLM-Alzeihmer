from langfuse import observe
from utils.utils import safe_format, safe_json_load
from llm.base import LLMClient


def validate_and_fix_assignments(assignments, arxiv_ids, axes):
    """
    Ensures:
    - All arxiv_ids are present
    - No extra IDs
    - Axes are valid
    - Format is correct
    """
    if not isinstance(assignments, dict):
        return {arxiv_id: [] for arxiv_id in arxiv_ids}

    fixed = {}
    for arxiv_id in arxiv_ids:
        value = assignments.get(arxiv_id, [])
        if not isinstance(value, list):
            value = []
        valid_axes = [a for a in value if a in axes]
        fixed[arxiv_id] = valid_axes[:2]

    return fixed


@observe(name="assign_articles_to_axes", as_type="generation")
def assign_articles_to_axes(
    cluster_name: str,
    summaries: list[str],
    axes: list[str],
    arxiv_ids: list[str],
    llm: LLMClient,
    prompt: str,
):
    axes_block = "\n".join(f"- {a}" for a in axes)

    joined_articles = "\n\n".join(
        f"[{arxiv_ids[i]}]\n{summaries[i]}" for i in range(len(summaries))
    )

    formatted_prompt = safe_format(
        prompt,
        cluster_name=cluster_name,
        axes_block=axes_block,
        joined_articles=joined_articles,
    )

    response = llm.complete(formatted_prompt)

    data = safe_json_load(response)

    assignments = data.get("assignments", {})
    assignments = validate_and_fix_assignments(assignments, arxiv_ids, axes)

    return assignments
