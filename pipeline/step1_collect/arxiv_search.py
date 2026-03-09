import arxiv
from langfuse import observe


@observe(name="build_arxiv_search", as_type="tool")
def build_arxiv_search(
    search_query: str,
    max_results: int = 100,
    sort_by: str = "lastUpdatedDate",
    sort_order: str = "descending",
) -> arxiv.Search:
    """
    Build an arXiv Search object.
    """

    sort_map = {
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
    }

    order_map = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending,
    }

    return arxiv.Search(
        query=f"all:{search_query}",
        max_results=max_results,
        sort_by=sort_map.get(sort_by, arxiv.SortCriterion.LastUpdatedDate),
        sort_order=order_map.get(sort_order, arxiv.SortOrder.Descending),
    )
