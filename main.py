# Load key of the project
from dotenv import load_dotenv

load_dotenv()
import os
from utils.io import log_crash, save_results
from langfuse import get_client, observe
from tracking.prompt_registry import sync_and_load_prompts
from config.load_settings import load_settings
from pipeline.step1_collect.run import run_step1
from pipeline.step2_clustering.run import run_step2
from pipeline.step3_generation.run import run_step3
from llm.base import LLMClient
from llm.LLMFactory import LLMFactory
from config.modalities.user_modalities import USER_MODALITIES
from config.modalities.alzheimer_modalities import MODALITIES

nb_paper = 10
provider = "ollama"
model = "gemma3:4b"
subject = "Alzheimer"
user_modalities = {}
scholar_citation = False
max_articles_per_generation = None
pdf_compilation = True
batch_size=0

@observe(name="subject_pipeline", as_type="chain")
def main(
    subject: str,
    provider: str = "ollama",
    model: str = "gemma3:4b",
    nb_paper: int = 10,
    max_articles_per_generation: int | None = None,
    scholar_citation: bool = False,
    user_modalities: dict = {},
    pdf_compilation: bool = True,
    batch_size: int = 0,
):
    """
    Full pipeline to generate a LaTeX state-of-the-art document on a given subject.

    Loads settings and synchronizes Langfuse prompts with local ones, then runs:
        - Step 1: Fetches papers from arXiv and processes them (citations, filtering, ...).
        - Step 2: Clusters the papers by modality.
        - Step 3: Generates the LaTeX document.
    Finally, saves the LaTeX document and optionally compiles it into a PDF.

    Args:
        subject (str): The research subject to investigate.
        provider (str, optional): The provider from which the model is served. Defaults to "ollama".
        model (str, optional): The model used for generation. Defaults to "gemma3:4b".
        nb_paper (int, optional): Number of papers to fetch from arXiv (not necessarily all processed). Defaults to 100.
        max_articles_per_generation (int | None, optional): Number of articles the model reads to generate
            each section of the state of the art. If None, all retrieved articles are used. Defaults to None.
        scholar_citation (bool, optional): Whether to retrieve citation counts via Semantic Scholar. Defaults to False.
        user_modalities (dict, optional): Custom modalities for the clustering step. If provided, overrides
            the automatic modality detection. Defaults to {}.
        pdf_compilation (bool, optional): Whether to compile the generated LaTeX document into a PDF. Defaults to True.
        batch_size (int, optional): Number of articles processed per API call batch. Values of 0 or 1 fall back
            to sequential (one-by-one) processing. Batching is recommended for remote API models;
            sequential processing is recommended for local models. Defaults to 0.
    """

    subject = subject.lower()
    print("Begin")
    llm = LLMFactory.create(provider=provider, model=model)
    print(f"Subject : {subject} \n provider : {provider}\n model : {model}")

    settings = load_settings()
    prompts = sync_and_load_prompts(settings["langfuse"]["prompts"])

    papers = run_step1(
        prompts=prompts,
        nb_paper=nb_paper,
        llm=llm,
        subject=subject,
        scholar_citation=scholar_citation,
        batch_size=batch_size,
    )

    clusters = run_step2(
        papers=papers,
        prompts=prompts,
        llm=llm,
        subject=subject,
        user_modalities=user_modalities,
        batch_size=batch_size,
    )

    latex = run_step3(
        clusters=clusters,
        papers=papers,
        prompts=prompts,
        llm=llm,
        reviewer_iterations=1,
        subject=subject,
        provider=provider,
        model=model,
        max_articles_per_generation=max_articles_per_generation,
    )
    save_results(latex, subject, pdf_compilation)

    print("End")


if __name__ == "__main__":
    try:
        main(
            subject=subject,
            nb_paper=nb_paper,
            provider=provider,
            model=model,
            user_modalities=user_modalities,
            scholar_citation=scholar_citation,
            max_articles_per_generation=max_articles_per_generation,
            pdf_compilation=pdf_compilation,
            batch_size=batch_size,
        )
    except Exception as e:
        log_crash(str(e))
        raise
    finally:
        get_client().flush()
