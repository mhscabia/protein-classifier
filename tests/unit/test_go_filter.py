import json

import pytest

from src.infrastructure.hierarchy.go_dag_builder import GODagBuilder


@pytest.fixture
def config(tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    return {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(tmp_path / "processed"),
            "uniprot_limit": 500,
            "go_namespace": "molecular_function",
        },
        "model": {"random_seed": 42},
        "hierarchy": {"min_term_support": 5},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def sample_terms():
    return [
        {
            "term_id": "GO:0003674",
            "name": "molecular_function",
            "is_obsolete": False,
            "parent_ids": [],
        },
        {
            "term_id": "GO:0005488",
            "name": "binding",
            "is_obsolete": False,
            "parent_ids": ["GO:0003674"],
        },
        {
            "term_id": "GO:0005524",
            "name": "ATP binding",
            "is_obsolete": False,
            "parent_ids": ["GO:0005488"],
        },
        {
            "term_id": "GO:9999999",
            "name": "rare term",
            "is_obsolete": False,
            "parent_ids": ["GO:0005488"],
        },
    ]


@pytest.fixture
def builder(config, sample_terms):
    raw_path = config["data"]["raw_path"]
    with open(f"{raw_path}/go_terms.json", "w", encoding="utf-8") as f:
        json.dump(sample_terms, f)
    return GODagBuilder(config=config)


class TestFilterBySupport:
    def test_removes_terms_below_min_support(self, builder):
        all_terms = ["GO:0005524", "GO:0005488", "GO:0003674", "GO:9999999"]
        counts = {
            "GO:0005524": 10,
            "GO:0005488": 12,
            "GO:0003674": 15,
            "GO:9999999": 2,
        }
        graph = builder.build(all_terms, term_counts=counts, min_support=5)

        assert graph.get_node("GO:9999999") is None
        assert graph.get_node("GO:0005524") is not None

    def test_preserves_ancestors_of_kept_terms(self, builder):
        all_terms = ["GO:0005524", "GO:0005488", "GO:0003674"]
        counts = {
            "GO:0005524": 10,
            "GO:0005488": 1,
            "GO:0003674": 1,
        }
        graph = builder.build(all_terms, term_counts=counts, min_support=5)

        assert graph.get_node("GO:0005524") is not None
        assert graph.get_node("GO:0005488") is not None
        assert graph.get_node("GO:0003674") is not None
