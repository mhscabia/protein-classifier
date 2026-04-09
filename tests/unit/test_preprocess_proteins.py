import pandas as pd
import pytest

from src.application.use_cases.preprocess_proteins import PreprocessProteinsUseCase
from src.domain.interfaces.preprocessor import ProteinPreprocessor


class FakePreprocessor(ProteinPreprocessor):
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna().reset_index(drop=True)

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["seq_length"] = df["sequence"].apply(len)
        return df


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "protein_id": ["P001", "P002"],
            "sequence": ["ACDEFG", "HIKLMN"],
            "go_terms": ["GO:0001", "GO:0002"],
        }
    )


class TestPreprocessProteinsUseCase:
    def test_returns_normalized_dataframe(self, valid_df):
        preprocessor = FakePreprocessor()
        use_case = PreprocessProteinsUseCase(preprocessor)

        result = use_case.execute(valid_df)
        assert "seq_length" in result.columns
        assert len(result) == 2

    def test_clean_then_normalize(self):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", None, "P003"],
                "sequence": ["ACDE", None, "FGHI"],
                "go_terms": ["GO:0001", "GO:0002", "GO:0003"],
            }
        )
        preprocessor = FakePreprocessor()
        use_case = PreprocessProteinsUseCase(preprocessor)

        result = use_case.execute(df)
        assert len(result) == 2
        assert "seq_length" in result.columns
