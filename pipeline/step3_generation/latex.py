import re
from datetime import datetime
from typing import Dict, List
from langfuse import observe
import bibtexparser

# ==========================
# Citation protection
# ==========================

CITE_PATTERN = re.compile(r"\\cite[a-zA-Z]*\{[^}]+\}")
MATH_PATTERN = re.compile(r"\${1,2}.*?\${1,2}", re.DOTALL)
LATEX_CMD_PATTERN = re.compile(r"\\[a-zA-Z]+\{[^}]*\}")


def arxiv_to_bibkey(arxiv_id: str) -> str:
    return f"arxiv{arxiv_id.replace('.', '')}"


def fix_double_backslashes(text: str) -> str:
    return re.sub(r"\\\\cite", r"\\cite", text)


@observe(name="protect_citations", as_type="tool")
def protect_citations(text: str):
    protected = {}

    def repl(match):
        key = f"@@CITE{len(protected)}@@"
        protected[key] = match.group(0)
        return key

    return CITE_PATTERN.sub(repl, text), protected


@observe(name="restore_citations", as_type="tool")
def restore_citations(text: str, protected: dict):
    for k, v in protected.items():
        text = text.replace(k, v)
    return text


def protect_math(text):
    protected = {}

    def repl(match):
        key = f"@@MATH{len(protected)}@@"
        protected[key] = match.group(0)
        return key

    return MATH_PATTERN.sub(repl, text), protected


def restore_math(text, protected):
    for k, v in protected.items():
        text = text.replace(k, v)
    return text


# ==========================
# LaTeX escaping
# ==========================


@observe(name="latex_escape_text", as_type="tool")
def latex_escape_text(text: str) -> str:
    if not text:
        return text

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "_": r"\_",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


@observe(name="latex_escape", as_type="tool")
def latex_escape(text: str) -> str:
    if not text:
        return text
    text = normalize_unicode(text)
    text = fix_double_backslashes(text)

    text, cites = protect_citations(text)
    text, maths = protect_math(text)

    text = latex_escape_text(text)

    text = restore_math(text, maths)
    text = restore_citations(text, cites)
    return text


# ==========================
# Bibliography builder
# ==========================


def build_bibliography_from_papers(papers, arxiv_url_map: dict[str, str]) -> str:
    lines = ["\\begin{thebibliography}\n"]
    lines[0] = lines[0] + "{" + f"{len(papers)}" + "}\n"

    for p in papers:
        key = f"arxiv{p.arxiv_id.replace('.', '')}"

        authors = latex_escape(", ".join(p.authors))
        title = latex_escape(p.title)

        entry = f"\\bibitem{{{key}}}\n{authors}. \\textit{{{title}}}. arXiv:{p.arxiv_id} ({p.published})"

        # Lookup URL in the parsed bib
        url = arxiv_url_map.get(p.arxiv_id)
        if url:
            entry += f". Available at: \\url{{{latex_escape(url)}}}"

        entry += "\n"
        lines.append(entry)

    lines.append("\\end{thebibliography}\n")
    return "".join(lines)


# ==========================
# Utility
# ==========================


def remove_markdown_fences(text: str) -> str:
    return re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")


import unicodedata


def normalize_unicode(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# ==========================
# Url recuperation
# ==========================


def parse_bib_urls(bib_path: str) -> dict[str, str]:
    with open(bib_path, encoding="utf-8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    arxiv_url_map = {}

    for entry in bib_database.entries:
        if "eprint" in entry and "url" in entry:
            arxiv_url_map[entry["eprint"].strip()] = entry["url"].strip()

    return arxiv_url_map


# ==========================
# Document builder
# ==========================


@observe(name="build_full_latex_document", as_type="tool")
def build_full_latex_document(
    clusters_content: Dict[str, str],
    global_intro: str,
    global_conclusion: str,
    papers: List,
    provider: str,
    model: str,
    subject: str = "Alzheimer",
) -> str:

    day = datetime.now().strftime("%Y_%m_%d")
    title = f"STATE OF THE ART : {subject.upper()} Research ({day}) \\ Generate by {provider} with {model}"

    # 🔹 Parse the .bib generated in step 1
    arxiv_url_map = parse_bib_urls(
        f"data/processed/{subject}_state_of_the_art_{day}.bib"
    )

    preamble = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{textcomp}}
\usepackage{{textgreek}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{geometry}}
\usepackage{{setspace}}
\usepackage{{hyperref}}
\geometry{{margin=2.5cm}}
\onehalfspacing
\title{{{latex_escape(title)}}}
\date{{\today}}
\begin{{document}}
\maketitle

"""

    doc = preamble

    # Introduction
    doc += "\\newpage \\section*  {Introduction}\n \n"
    global_intro = remove_markdown_fences(global_intro)
    doc += latex_escape(global_intro) + "\n\n"

    # Clusters
    for cluster_name, sot in clusters_content.items():
        doc += f"\\newpage\n \\section{{{latex_escape(cluster_name)}}}\n"
        sot = remove_markdown_fences(sot)
        doc += latex_escape(sot) + "\n\n"

    # Conclusion
    doc += "\\newpage\n \\section*{Conclusion}\n"
    global_conclusion = remove_markdown_fences(global_conclusion)
    doc += latex_escape(global_conclusion) + "\n\n \\newpage"

    # References with URLs
    doc += build_bibliography_from_papers(papers, arxiv_url_map)

    doc += "\n\\end{document}"

    return doc
