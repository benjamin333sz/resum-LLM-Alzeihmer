import time
import requests
from entities.paper import Paper
from langfuse import observe
from tqdm import tqdm

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}"


@observe(name="semantic_scholar_enrichment", as_type="retriever")
def enrich_with_semantic_scholar(
    papers: list[Paper],
    sleep_seconds: float = 1.0,
    max_retries: int = 4,
) -> list[Paper]:
    """
    Enrich papers with citation metadata from Semantic Scholar.
    Implements exponential backoff retries for failed requests.

    Args:
        papers (list[Paper]): List of Paper objects to enrich.
        sleep_seconds (float): Base sleep between successful requests.
        max_retries (int): Maximum number of retry attempts with exponential backoff.

    Returns:
        list[Paper]: Enriched papers (with citation metadata).
    """
    failed_ids = []
    success_count = 0

    for paper in tqdm(papers, desc="[Step_1] Enriching with Semantic Scholar"):
        url = SEMANTIC_SCHOLAR_URL.format(paper.arxiv_id)
        params = {"fields": "year,citationCount,referenceCount,fieldsOfStudy"}

        attempt = 0
        while attempt <= max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                paper.year = data.get("year")
                paper.citation_count = data.get("citationCount")
                paper.reference_count = data.get("referenceCount")
                paper.fields_of_study = data.get("fieldsOfStudy")

                success_count += 1
                time.sleep(sleep_seconds)  # sleep to preserve the API
                break  # success

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt > max_retries:
                    failed_ids.append(paper.arxiv_id)
                    # log for debug
                    # print(f"[SemanticScholar] Failed after {max_retries} attempts: {paper.arxiv_id}")
                    break
                # exponential backoff (1,2,4,8,16s)
                backoff = 2 ** (attempt - 1)
                time.sleep(backoff)

    print(f"\n[SemanticScholar] Success: {success_count}/{len(papers)} papers enriched")
    if failed_ids:
        print(
            f"[SemanticScholar] Failed for {len(failed_ids)} papers: {', '.join(failed_ids)}"
        )

    return papers
