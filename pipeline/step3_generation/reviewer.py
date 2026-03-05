from langfuse import observe
from utils.utils import safe_format


@observe(name="review_state_of_the_art", as_type="evaluator")
def review_sot(cluster_name, sot_text, llm, prompt):

    formatted_prompt = safe_format(prompt, cluster_name=cluster_name, sot_text=sot_text)

    return llm.complete(formatted_prompt, temperature=0.1)


@observe(name="revise_state_of_the_art", as_type="generation")
def revise_sot(cluster_name, sot_text, review_text, llm, prompt):

    formatted_prompt = safe_format(
        prompt, cluster_name=cluster_name, sot_text=sot_text, review_text=review_text
    )

    return llm.complete(formatted_prompt, temperature=0.1)
