from dataclasses import dataclass


@dataclass
class Protein:
    protein_id: str
    sequence: str
    go_terms: list[str]
