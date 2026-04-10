import pandas as pd
import pytest

from src.application.use_cases.fetch_proteins import FetchProteinsUseCase
from src.domain.interfaces.data_source import ProteinDataSource


class FakeDataSource(ProteinDataSource):
    def __init__(self, df: pd.DataFrame, conformity: bool = True):
        self._df = df
        self._conformity = conformity

    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        return self._df.head(limit)

    def verify_conformity(self, data: pd.DataFrame) -> bool:
        return self._conformity


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "protein_id": ["P001", "P002", "P003"],
            "sequence": ["ACDEFG", "HIKLMN", "PQRSTV"],
            "go_terms": ["GO:0001", "GO:0002", "GO:0003"],
        }
    )


class TestFetchProteinsUseCase:
    def test_returns_dataframe(self, valid_df):
        source = FakeDataSource(valid_df)
        use_case = FetchProteinsUseCase(source)

        result = use_case.execute(limit=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_respects_limit(self, valid_df):
        source = FakeDataSource(valid_df)
        use_case = FetchProteinsUseCase(source)

        result = use_case.execute(limit=2)
        assert len(result) == 2

    def test_raises_on_failed_conformity(self, valid_df):
        source = FakeDataSource(valid_df, conformity=False)
        use_case = FetchProteinsUseCase(source)

        with pytest.raises(ValueError, match="conformidade"):
            use_case.execute(limit=10)

    def test_passes_conformity_check(self, valid_df):
        source = FakeDataSource(valid_df, conformity=True)
        use_case = FetchProteinsUseCase(source)

        result = use_case.execute(limit=10)
        assert len(result) == 3
