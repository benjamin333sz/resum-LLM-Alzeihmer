from langfuse import observe
from .axes import extract_axes
from .assignment import assign_articles_to_axes
from .generate_paragraph import generate_paragraph
from .sot import generate_sot
from .reviewer import review_sot, revise_sot
from .latex import build_full_latex_document, arxiv_to_bibkey
from entities.paper import Paper
from llm.base import LLMClient
from tqdm import tqdm
from .global_generality import introduction, conclusion
from .chunk import split_chunks


@observe(name="step3_generation", as_type="chain")
def run_step3(
    clusters: dict[str, list[str]],
    papers: list[Paper],
    llm: LLMClient,
    prompts: dict[str, str],
    provider: str,
    model: str,
    reviewer_iterations: int = 1,
    subject: str = "alzheimer",
    CHUNK_SIZE: int = 10,
) -> str:

    papers_by_id = {p.arxiv_id: p for p in papers}
    clusters_content = {}

    names_clusters = []

    for modality in tqdm(
        clusters["modalities"],
        desc="[Step_3] Generation of the State-of-the-Art",
        position=0,
    ):
        cluster_name = modality["modality_id"]
        names_clusters.append(cluster_name)
        paper_ids = modality["article_ids"]

        chunks = split_chunks(paper_ids, CHUNK_SIZE)
        chunk_sots = []

        for i, chunk_ids in enumerate(tqdm(chunks, desc="split chunk", leave=False)):

            summaries = [
                papers_by_id[pid].abstract for pid in chunk_ids if pid in papers_by_id
            ]

            if not summaries:
                continue

            axes = extract_axes(cluster_name, summaries, llm, prompts["extract_axes"])

            assignments = assign_articles_to_axes(
                cluster_name,
                summaries,
                axes,
                chunk_ids,
                llm,
                prompts["assign_articles"],
            )

            axis_to_arxiv_ids = {axis: [] for axis in axes}

            for arxiv_id, assigned_axes in assignments.items():
                for axis in assigned_axes:
                    if axis in axis_to_arxiv_ids:
                        axis_to_arxiv_ids[axis].append(arxiv_id)

            paragraphs = {}

            for axis in axes:

                arxiv_ids = axis_to_arxiv_ids.get(axis, [])
                filtered_summaries = [
                    papers_by_id[aid].abstract
                    for aid in arxiv_ids
                    if aid in papers_by_id
                ]

                if not filtered_summaries:
                    continue

                bib_keys = [arxiv_to_bibkey(aid) for aid in arxiv_ids]

                paragraph = generate_paragraph(
                    cluster_name,
                    axis,
                    filtered_summaries,
                    bib_keys,
                    llm,
                    prompts["paragraph"],
                )

                paragraphs[axis] = paragraph

            sot = generate_sot(cluster_name, paragraphs, llm, prompts["sot_merge"])

            chunk_sots.append(f"---\nChunk {i+1}\n---\n{sot}")

        # concaténer tous les chunks pour la modalité
        clusters_content[cluster_name] = "\n\n".join(chunk_sots)

    global_intro = introduction(
        prompts["global_intro"],
        llm,
        subject=subject,
        joined=names_clusters,
    )
    global_conclusion = conclusion(
        prompts["global_conclusion"],
        llm,
        subject=subject,
        joined=names_clusters,
    )

    return build_full_latex_document(
        clusters_content=clusters_content,
        global_intro=global_intro,
        global_conclusion=global_conclusion,
        papers=papers,
        subject=subject,
        model=model,
        provider=provider,
    )
