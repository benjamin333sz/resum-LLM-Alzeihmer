import arxiv
from entities.paper import Paper
from langfuse import observe
from tqdm import tqdm
import re


@observe(name="fetch_arxiv_papers", as_type="retriever")
def fetch_arxiv_papers(search: arxiv.Search) -> list[Paper]:

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=5,
    )

    papers = []

    for result in tqdm(client.results(search),desc="[Step_1] Fetch papers"):
        arxiv_id = normalize_arxiv_id(result.entry_id)

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.isoformat(),
                categories=result.categories,
            )
        )

    return papers


def normalize_arxiv_id(entry_id: str) -> str:
    """
    Extract arXiv ID without version.
    Works for modern and legacy IDs.
    """
    raw_id = entry_id.split("/abs/")[-1]
    return re.sub(r"v\d+$", "", raw_id)
