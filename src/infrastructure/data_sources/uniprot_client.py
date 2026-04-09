import pandas as pd

from src.domain.interfaces.data_source import ProteinDataSource


class UniProtClient(ProteinDataSource):
    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        raise NotImplementedError

    def verify_conformity(self, data: pd.DataFrame) -> bool:
        raise NotImplementedError
