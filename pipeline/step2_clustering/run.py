from collections import defaultdict
from entities.paper import Paper
from utils.io import save_raw_json
from config.modalities.alzheimer_modalities import MODALITIES
from pipeline.step2_clustering.creation_modality import creation_modality
from .clustering import clustering
from .clustering_batch import clustering_batch
from langfuse import observe
from llm.base import LLMClient


@observe(name="step2_clustering", as_type="chain")
def run_step2(
    papers: list[Paper],
    llm: LLMClient,
    prompts: dict[str, str],
    subject: str,
    user_modalities: dict[str, str] | None = None,
    batch_size: int = 5,
) -> dict[str, object]:

    if user_modalities:
        if batch_size:
            clusters = clustering_batch(
                papers,
                llm,
                prompts["clustering_batch"],
                user_modalities,
                batch_size_start=batch_size,
            )
        else:
            clusters = clustering(papers, llm, prompts["clustering"], user_modalities)
    else:
        user_modalities = creation_modality(subject, llm, prompts["creation_modality"])
        if batch_size:
            clusters = clustering_batch(
                papers,
                llm,
                prompts["clustering_batch"],
                user_modalities,
                batch_size_start=batch_size,
            )
        else:
            clusters = clustering(papers, llm, prompts["clustering"], user_modalities)

    if not clusters:
        raise ValueError("Clustering failed: no clusters created.")

    output = {
        "step": "step2_clustering",
        "method": "LLM_modality_clustering",
        "model": llm.model,
        "n_clusters": len(clusters),
        "n_papers": len(papers),
        "modalities": [
            {
                "modality_id": k,
                "size": len(v),
                "article_ids": [p.arxiv_id for p in v],
            }
            for k, v in clusters.items()
        ],
    }

    save_raw_json(output, "data/artifacts/step2_clusters.json")

    return output
