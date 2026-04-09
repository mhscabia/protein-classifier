import numpy as np
import pandas as pd
import pytest

from src.infrastructure.preprocessing.pandas_preprocessor import (
    PandasPreprocessor,
    _composition,
    _molecular_weight,
    AMINO_ACIDS,
)


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


# ---------- _composition ----------


class TestComposition:
    def test_uniform_sequence(self):
        result = _composition("AAAA")
        assert result["aa_A"] == pytest.approx(1.0)
        assert result["aa_C"] == pytest.approx(0.0)

    def test_mixed_sequence(self):
        result = _composition("AC")
        assert result["aa_A"] == pytest.approx(0.5)
        assert result["aa_C"] == pytest.approx(0.5)

    def test_case_insensitive(self):
        result = _composition("aacC")
        assert result["aa_A"] == pytest.approx(0.5)
        assert result["aa_C"] == pytest.approx(0.5)

    def test_empty_sequence(self):
        result = _composition("")
        assert all(v == 0.0 for v in result.values())

    def test_all_amino_acids_present(self):
        result = _composition("A")
        assert len(result) == len(AMINO_ACIDS)


# ---------- _molecular_weight ----------


class TestMolecularWeight:
    def test_single_residue(self):
        weight = _molecular_weight("A")
        assert weight == pytest.approx(89.09, abs=0.01)

    def test_dipeptide(self):
        expected = 89.09 + 121.16 - 18.015
        weight = _molecular_weight("AC")
        assert weight == pytest.approx(expected, abs=0.01)

    def test_empty_sequence(self):
        assert _molecular_weight("") == 0.0

    def test_case_insensitive(self):
        assert _molecular_weight("ac") == _molecular_weight("AC")


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


# ---------- normalize ----------


class TestNormalize:
    def test_adds_feature_columns(self, preprocessor, valid_df):
        result = preprocessor.normalize(valid_df)
        assert "seq_length" in result.columns
        assert "molecular_weight" in result.columns
        for aa in AMINO_ACIDS:
            assert f"aa_{aa}" in result.columns

    def test_preserves_original_columns(self, preprocessor, valid_df):
        result = preprocessor.normalize(valid_df)
        assert "protein_id" in result.columns
        assert "sequence" in result.columns
        assert "go_terms" in result.columns

    def test_features_are_standardized(self, preprocessor, valid_df):
        result = preprocessor.normalize(valid_df)
        feature_cols = ["seq_length", "molecular_weight"] + [
            f"aa_{aa}" for aa in AMINO_ACIDS
        ]
        for col in feature_cols:
            values = result[col].values
            if np.std(values) > 0:
                assert np.mean(values) == pytest.approx(0.0, abs=1e-10)
                assert np.std(values, ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_saves_csv(self, preprocessor, valid_df):
        preprocessor.normalize(valid_df)
        output = (
            preprocessor._processed_path / "proteins_clean.csv"
        )
        assert output.exists()
        saved = pd.read_csv(output)
        assert len(saved) == len(valid_df)

    def test_single_row(self, preprocessor):
        df = pd.DataFrame(
            {
                "protein_id": ["P001"],
                "sequence": ["ACDE"],
                "go_terms": ["GO:0001"],
            }
        )
        result = preprocessor.normalize(df)
        assert len(result) == 1
        assert "seq_length" in result.columns
