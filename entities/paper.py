from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    categories: List[str]

    year: Optional[int] = None
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None
    fields_of_study: Optional[List[str]] = None

    is_about_subject: Optional[bool] = None

    def to_dict(self):
        return asdict(self)
