from entities.paper import Paper
from llm.base import LLMClient
from tqdm import tqdm
from tracking.client import get_langfuse
from langfuse import observe
from utils.utils import safe_format


@observe(name="filter_subject", as_type="evaluator")
def filter_subject(
    papers: list[Paper],
    llm: LLMClient,
    prompt_template: str,
    subject: str = "Alzheimer",
) -> list[Paper]:
    """
    Filter papers using an LLM to determine whether they are about Alzheimer's disease.

    Args:
        papers (list[Paper]): Papers to evaluate.
        llm (LLMClient): LLM client used for inference.
        prompt_template (str): Compiled prompt template from Langfuse.

    Returns:
        list[Paper]: Papers with is_about_subject field populated.
    """

    langfuse = get_langfuse()

    for paper in tqdm(papers, desc=f"[Step_1] Filtering papers iteration about {subject}"):
        prompt = safe_format(
            prompt_template,
            title=paper.title,
            abstract=paper.abstract,
            subject=subject,
        )

        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"{subject}_classification_llm",
            model=llm.model,
            input=prompt,
            metadata={
                "arxiv_id": paper.arxiv_id,
                "paper_title": paper.title,
            },
        ) as generation:

            response = llm.complete(prompt, temperature=0.1)
            generation.update(output=response)

        response_lower = response.lower()
        paper.is_about_subject = "yes" in response_lower

    return papers
