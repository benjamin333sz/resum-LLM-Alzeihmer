from langfuse import observe
from entities.paper import Paper
from utils.utils import clean_token


def paper_to_bibtex(paper: Paper) -> str:
    authors = " and ".join(paper.authors)

    year = paper.year
    if year is None and paper.published:
        year = paper.published[:4]

    primary_class = paper.categories[0] if paper.categories else "unknown"

    key_author = clean_token(paper.authors[0].split()[-1])
    first_word_title = clean_token(paper.title.split()[0])
    bib_key = f"{key_author}{year}{first_word_title}"

    return f"""@misc{{{bib_key},
  title={{ {paper.title} }},
  author={{ {authors} }},
  year={{ {year} }},
  eprint={{ {paper.arxiv_id} }},
  archivePrefix={{arXiv}},
  primaryClass={{ {primary_class} }},
  url={{ https://arxiv.org/abs/{paper.arxiv_id} }}
}}
"""


@observe(name="generate_bibliography", as_type="span")
def generate_bib_file(papers: list[Paper], path: str):
    entries = [paper_to_bibtex(p) for p in papers]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
