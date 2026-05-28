import pandas as pd
import pytest

from src.infrastructure.preprocessing.pandas_preprocessor import PandasPreprocessor


@pytest.fixture
def config(tmp_path):
    return {
        "data": {
            "raw_path": str(tmp_path / "raw"),
            "processed_path": str(tmp_path / "processed"),
            "uniprot_limit": 10,
            "go_namespace": "molecular_function",
        },
        "model": {"random_seed": 42, "test_size": 0.2, "validation_size": 0.1},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def preprocessor(config):
    return PandasPreprocessor(config=config)


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "protein_id": ["P001", "P002", "P003"],
            "sequence": ["ACDEFG", "HIKLMN", "PQRSTV"],
            "go_terms": ["GO:0001;GO:0002", "GO:0003", "GO:0004;GO:0005"],
        }
    )


# ---------- clean ----------


class TestClean:
    def test_removes_null_rows(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", None, "P003"],
                "sequence": ["ACDE", "FGHI", None],
                "go_terms": ["GO:0001", "GO:0002", "GO:0003"],
            }
        )
        result = preprocessor.clean(df)
        assert len(result) == 1
        assert result.iloc[0]["protein_id"] == "P001"

    def test_removes_empty_strings(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", "P002"],
                "sequence": ["ACDE", ""],
                "go_terms": ["GO:0001", "GO:0002"],
            }
        )
        result = preprocessor.clean(df)
        assert len(result) == 1

    def test_removes_duplicates(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", "P001", "P002"],
                "sequence": ["ACDE", "FGHI", "KLMN"],
                "go_terms": ["GO:0001", "GO:0002", "GO:0003"],
            }
        )
        result = preprocessor.clean(df)
        assert len(result) == 2
        assert result.iloc[0]["sequence"] == "ACDE"

    def test_removes_invalid_sequences(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", "P002", "P003"],
                "sequence": ["ACDE", "AC1E", "XBZJ"],
                "go_terms": ["GO:0001", "GO:0002", "GO:0003"],
            }
        )
        result = preprocessor.clean(df)
        assert len(result) == 1
        assert result.iloc[0]["protein_id"] == "P001"

    def test_valid_data_unchanged(self, preprocessor, valid_df):
        result = preprocessor.clean(valid_df)
        assert len(result) == len(valid_df)

    def test_resets_index(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001", "P002"],
                "sequence": ["ACDE", "FGHI"],
                "go_terms": ["GO:0001", ""],
            }
        )
        result = preprocessor.clean(df)
        assert list(result.index) == list(range(len(result)))
