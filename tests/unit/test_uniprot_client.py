from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.data_sources.uniprot_client import UniProtClient, _extract_go_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/",
            "uniprot_limit": 500,
            "go_namespace": "molecular_function",
        },
        "model": {"random_seed": 42},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def client(config):
    return UniProtClient(config=config)


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "protein_id": ["P12345", "Q67890", "A11111"],
            "sequence": ["MAAAA", "MBBBB", "MCCCC"],
            "go_terms": [
                "GO:0003674;GO:0005524",
                "GO:0003674",
                "GO:0004672;GO:0005524",
            ],
        }
    )


# ---------------------------------------------------------------------------
# _extract_go_ids
# ---------------------------------------------------------------------------

class TestExtractGoIds:
    def test_single_term(self):
        text = "molecular_function [GO:0003674]"
        assert _extract_go_ids(text) == ["GO:0003674"]

    def test_multiple_terms(self):
        text = "ATP binding [GO:0005524]; protein kinase activity [GO:0004672]"
        assert _extract_go_ids(text) == ["GO:0005524", "GO:0004672"]

    def test_empty_string(self):
        assert _extract_go_ids("") == []

    def test_nan_value(self):
        assert _extract_go_ids(float("nan")) == []

    def test_no_go_terms(self):
        assert _extract_go_ids("some random text") == []


# ---------------------------------------------------------------------------
# verify_conformity
# ---------------------------------------------------------------------------

class TestVerifyConformity:
    def test_valid_data(self, client, valid_df):
        assert client.verify_conformity(valid_df) is True

    def test_missing_column(self, client):
        df = pd.DataFrame({"protein_id": ["P1"], "sequence": ["M"]})
        assert client.verify_conformity(df) is False

    def test_empty_dataframe(self, client):
        df = pd.DataFrame(columns=["protein_id", "sequence", "go_terms"])
        assert client.verify_conformity(df) is False

    def test_null_protein_id(self, client):
        df = pd.DataFrame(
            {
                "protein_id": [None, "Q1"],
                "sequence": ["MAAAA", "MBBBB"],
                "go_terms": ["GO:0003674", "GO:0003674"],
            }
        )
        assert client.verify_conformity(df) is False

    def test_null_sequence(self, client):
        df = pd.DataFrame(
            {
                "protein_id": ["P1", "Q1"],
                "sequence": ["MAAAA", None],
                "go_terms": ["GO:0003674", "GO:0003674"],
            }
        )
        assert client.verify_conformity(df) is False

    def test_null_go_terms(self, client):
        df = pd.DataFrame(
            {
                "protein_id": ["P1"],
                "sequence": ["MAAAA"],
                "go_terms": [None],
            }
        )
        assert client.verify_conformity(df) is False

    def test_empty_string_sequence(self, client):
        df = pd.DataFrame(
            {
                "protein_id": ["P1"],
                "sequence": ["  "],
                "go_terms": ["GO:0003674"],
            }
        )
        assert client.verify_conformity(df) is False

    def test_empty_string_go_terms(self, client):
        df = pd.DataFrame(
            {
                "protein_id": ["P1"],
                "sequence": ["MAAAA"],
                "go_terms": [""],
            }
        )
        assert client.verify_conformity(df) is False

    def test_duplicate_ids_still_valid(self, client):
        """Duplicatas geram warning mas não invalidam o dataset."""
        df = pd.DataFrame(
            {
                "protein_id": ["P1", "P1"],
                "sequence": ["MAAAA", "MBBBB"],
                "go_terms": ["GO:0003674", "GO:0005524"],
            }
        )
        assert client.verify_conformity(df) is True


# ---------------------------------------------------------------------------
# _parse_next_link
# ---------------------------------------------------------------------------

class TestParseNextLink:
    def test_valid_link_header(self):
        header = '<https://rest.uniprot.org/uniprotkb/search?cursor=abc>; rel="next"'
        assert (
            UniProtClient._parse_next_link(header)
            == "https://rest.uniprot.org/uniprotkb/search?cursor=abc"
        )

    def test_empty_header(self):
        assert UniProtClient._parse_next_link("") is None

    def test_no_next_rel(self):
        header = '<https://example.com>; rel="prev"'
        assert UniProtClient._parse_next_link(header) is None


# ---------------------------------------------------------------------------
# _normalize_dataframe
# ---------------------------------------------------------------------------

class TestNormalizeDataframe:
    def test_normalizes_columns(self, client):
        raw = pd.DataFrame(
            {
                "Entry": ["P1", "Q2"],
                "Sequence": ["MAAAA", "MBBBB"],
                "Gene Ontology (molecular function)": [
                    "ATP binding [GO:0005524]",
                    "molecular_function [GO:0003674]; kinase [GO:0004672]",
                ],
            }
        )
        df = client._normalize_dataframe(raw)

        assert list(df.columns) == ["protein_id", "sequence", "go_terms"]
        assert df.iloc[0]["protein_id"] == "P1"
        assert df.iloc[0]["go_terms"] == "GO:0005524"
        assert df.iloc[1]["go_terms"] == "GO:0003674;GO:0004672"

    def test_filters_rows_without_go_terms(self, client):
        raw = pd.DataFrame(
            {
                "Entry": ["P1", "Q2"],
                "Sequence": ["MAAAA", "MBBBB"],
                "Gene Ontology (molecular function)": [
                    "ATP binding [GO:0005524]",
                    "",
                ],
            }
        )
        df = client._normalize_dataframe(raw)
        assert len(df) == 1
        assert df.iloc[0]["protein_id"] == "P1"


# ---------------------------------------------------------------------------
# fetch_proteins (mocked HTTP)
# ---------------------------------------------------------------------------

MOCK_TSV_RESPONSE = (
    "Entry\tSequence\tGene Ontology (molecular function)\n"
    "P12345\tMAAAA\tATP binding [GO:0005524]; kinase [GO:0004672]\n"
    "Q67890\tMBBBB\tmolecular_function [GO:0003674]\n"
)


class TestFetchProteins:
    @patch("src.infrastructure.data_sources.uniprot_client.requests.get")
    def test_fetch_and_save(self, mock_get, client, tmp_path):
        client._raw_path = tmp_path

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_TSV_RESPONSE
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = client.fetch_proteins(limit=10)

        assert len(df) == 2
        assert list(df.columns) == ["protein_id", "sequence", "go_terms"]
        assert df.iloc[0]["protein_id"] == "P12345"
        assert "GO:0005524" in df.iloc[0]["go_terms"]

        csv_path = tmp_path / "proteins.csv"
        assert csv_path.exists()

    @patch("src.infrastructure.data_sources.uniprot_client.requests.get")
    def test_empty_response(self, mock_get, client, tmp_path):
        client._raw_path = tmp_path

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Entry\tSequence\tGene Ontology (molecular function)\n"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = client.fetch_proteins(limit=10)
        assert len(df) == 0

    @patch("src.infrastructure.data_sources.uniprot_client.requests.get")
    def test_respects_limit(self, mock_get, client, tmp_path):
        client._raw_path = tmp_path

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_TSV_RESPONSE
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = client.fetch_proteins(limit=1)
        assert len(df) <= 1

    @patch("src.infrastructure.data_sources.uniprot_client.requests.get")
    def test_pagination(self, mock_get, client, tmp_path):
        client._raw_path = tmp_path

        page1 = MagicMock()
        page1.status_code = 200
        page1.text = MOCK_TSV_RESPONSE
        page1.headers = {
            "Link": '<https://rest.uniprot.org/next>; rel="next"'
        }
        page1.raise_for_status = MagicMock()

        page2_tsv = (
            "Entry\tSequence\tGene Ontology (molecular function)\n"
            "A11111\tMCCCC\tbinding [GO:0005488]\n"
        )
        page2 = MagicMock()
        page2.status_code = 200
        page2.text = page2_tsv
        page2.headers = {}
        page2.raise_for_status = MagicMock()

        mock_get.side_effect = [page1, page2]

        df = client.fetch_proteins(limit=10)
        assert len(df) == 3
        assert mock_get.call_count == 2
