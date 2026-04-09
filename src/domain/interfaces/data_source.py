from abc import ABC, abstractmethod

import pandas as pd


class ProteinDataSource(ABC):
    @abstractmethod
    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        """Retorna DataFrame com colunas: protein_id, sequence, go_terms."""
        ...

    @abstractmethod
    def verify_conformity(self, data: pd.DataFrame) -> bool:
        """Verifica colunas obrigatórias e ausência de nulos críticos."""
        ...
