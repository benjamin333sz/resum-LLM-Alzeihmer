from pipeline.step1_collect.arxiv import fetch_arxiv_papers
from pipeline.step1_collect.semantic_scholar import enrich_with_semantic_scholar
from pipeline.step1_collect.llm_filter import filter_subject
from pipeline.step1_collect.llm_batch import filter_batch
from llm.ollama import OllamaClient
from pipeline.step1_collect.arxiv_search import build_arxiv_search
from utils.io import save_json
from langfuse import observe
from entities.paper import Paper
from pipeline.step1_collect.bibtex import generate_bib_file
from llm.base import LLMClient
from datetime import datetime


@observe(name="step1_collect", as_type="chain")
def run_step1(
    prompts: dict[str, str],
    llm: LLMClient,
    nb_paper: int = 50,
    subject: str = "alzheimer",
    scholar_citation: bool = False,
    batch_size: int = 5,
) -> list[Paper]:
    """
    Run the full Step 1 pipeline:
    - Fetch arXiv papers
    - Enrich with Semantic Scholar
    - Filter using LLM
    - Save results
    """
    # Retrieve the article
    feed_url = build_arxiv_search(search_query=subject, max_results=nb_paper)

    papers = fetch_arxiv_papers(feed_url)

    if not papers:
        raise ValueError(
            f"No papers found on arXiv for subject '{subject}'. "
            "Please try a broader or different query."
        )

    save_json(papers, "data/raw/papers.json")

    # data enrichment
    if scholar_citation:
        papers = enrich_with_semantic_scholar(papers)

    # LLM filter
    if batch_size or batch_size == 1:
        papers = filter_batch(
            papers,
            llm,
            prompts["filter_batch"],
            subject=subject,
            batch_size=batch_size,
        )
    else:
        papers = filter_subject(
            papers,
            llm,
            prompts["filter_subject"],
            subject=subject,
        )

    filtered = [p for p in papers if p.is_about_subject]
    print(f"[Step_1] {len(filtered)} articles are about Alzheimer.")

    if not filtered:
        raise ValueError(
            f"Papers were found for '{subject}' but none passed the LLM filter."
        )

    # save
    save_json(filtered, "data/processed/step1_papers.json")

    day = datetime.now().strftime("%Y_%m_%d")
    generate_bib_file(filtered, f"data/processed/{subject}_state_of_the_art_{day}.bib")

    return filtered
