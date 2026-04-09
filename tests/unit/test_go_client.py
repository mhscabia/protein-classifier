import json
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.data_sources.go_client import GOClient


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
    return GOClient(config=config)


# ---------------------------------------------------------------------------
# _fill_parent_ids
# ---------------------------------------------------------------------------

class TestFillParentIds:
    def test_derives_parents_from_children(self):
        term_details = {
            "GO:0003674": {
                "term_id": "GO:0003674",
                "name": "molecular_function",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [{"id": "GO:0005488", "relation": "is_a"}],
                "parent_ids": [],
            },
            "GO:0005488": {
                "term_id": "GO:0005488",
                "name": "binding",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [{"id": "GO:0005524", "relation": "is_a"}],
                "parent_ids": [],
            },
            "GO:0005524": {
                "term_id": "GO:0005524",
                "name": "ATP binding",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [],
                "parent_ids": [],
            },
        }

        GOClient._fill_parent_ids(term_details)

        assert term_details["GO:0005488"]["parent_ids"] == ["GO:0003674"]
        assert term_details["GO:0005524"]["parent_ids"] == ["GO:0005488"]
        assert term_details["GO:0003674"]["parent_ids"] == []

    def test_skips_unknown_children(self):
        term_details = {
            "GO:0003674": {
                "term_id": "GO:0003674",
                "name": "molecular_function",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [{"id": "GO:9999999", "relation": "is_a"}],
                "parent_ids": [],
            },
        }

        GOClient._fill_parent_ids(term_details)
        assert term_details["GO:0003674"]["parent_ids"] == []

    def test_no_duplicate_parents(self):
        term_details = {
            "GO:0003674": {
                "term_id": "GO:0003674",
                "name": "molecular_function",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [{"id": "GO:0005488", "relation": "is_a"}],
                "parent_ids": [],
            },
            "GO:0005488": {
                "term_id": "GO:0005488",
                "name": "binding",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [],
                "parent_ids": [],
            },
        }

        # Chamar duas vezes não deve duplicar
        GOClient._fill_parent_ids(term_details)
        GOClient._fill_parent_ids(term_details)
        assert term_details["GO:0005488"]["parent_ids"] == ["GO:0003674"]


# ---------------------------------------------------------------------------
# _fetch_ancestors (mocked)
# ---------------------------------------------------------------------------

class TestFetchAncestors:
    @patch("src.infrastructure.data_sources.go_client.time.sleep")
    def test_returns_ancestor_ids(self, mock_sleep, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": ["GO:0005488", "GO:0003674"]
        }
        mock_response.raise_for_status = MagicMock()
        client._session.get = MagicMock(return_value=mock_response)

        ancestors = client._fetch_ancestors("GO:0005524")
        assert ancestors == {"GO:0005488", "GO:0003674"}

    @patch("src.infrastructure.data_sources.go_client.time.sleep")
    def test_returns_empty_on_404(self, mock_sleep, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        client._session.get = MagicMock(return_value=mock_response)

        ancestors = client._fetch_ancestors("GO:INVALID")
        assert ancestors == set()


# ---------------------------------------------------------------------------
# _fetch_details_batch (mocked)
# ---------------------------------------------------------------------------

class TestFetchDetailsBatch:
    @patch("src.infrastructure.data_sources.go_client.time.sleep")
    def test_returns_term_details(self, mock_sleep, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "GO:0005524",
                    "name": "ATP binding",
                    "aspect": "molecular_function",
                    "isObsolete": False,
                    "children": [],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        client._session.get = MagicMock(return_value=mock_response)

        details = client._fetch_details_batch({"GO:0005524"})

        assert "GO:0005524" in details
        assert details["GO:0005524"]["name"] == "ATP binding"
        assert details["GO:0005524"]["namespace"] == "molecular_function"
        assert details["GO:0005524"]["parent_ids"] == []


# ---------------------------------------------------------------------------
# fetch_go_terms (integration com mocks)
# ---------------------------------------------------------------------------

class TestFetchGoTerms:
    @patch("src.infrastructure.data_sources.go_client.time.sleep")
    def test_full_pipeline(self, mock_sleep, client, tmp_path):
        client._raw_path = tmp_path

        ancestors_response = MagicMock()
        ancestors_response.status_code = 200
        ancestors_response.json.return_value = {
            "results": ["GO:0005488", "GO:0003674"]
        }
        ancestors_response.raise_for_status = MagicMock()

        details_response = MagicMock()
        details_response.status_code = 200
        details_response.json.return_value = {
            "results": [
                {
                    "id": "GO:0003674",
                    "name": "molecular_function",
                    "aspect": "molecular_function",
                    "isObsolete": False,
                    "children": [{"id": "GO:0005488", "relation": "is_a"}],
                },
                {
                    "id": "GO:0005488",
                    "name": "binding",
                    "aspect": "molecular_function",
                    "isObsolete": False,
                    "children": [{"id": "GO:0005524", "relation": "is_a"}],
                },
                {
                    "id": "GO:0005524",
                    "name": "ATP binding",
                    "aspect": "molecular_function",
                    "isObsolete": False,
                    "children": [],
                },
            ]
        }
        details_response.raise_for_status = MagicMock()

        def side_effect(url, **kwargs):
            if "/ancestors" in url:
                return ancestors_response
            return details_response

        client._session.get = MagicMock(side_effect=side_effect)

        terms = client.fetch_go_terms(["GO:0005524"])

        assert len(terms) == 3

        json_path = tmp_path / "go_terms.json"
        assert json_path.exists()
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(saved) == 3

        atp = next(t for t in terms if t["term_id"] == "GO:0005524")
        assert "GO:0005488" in atp["parent_ids"]
