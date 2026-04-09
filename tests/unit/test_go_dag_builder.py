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
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def sample_terms():
    """Hierarquia simples: molecular_function -> binding -> ATP binding."""
    return [
        {
            "term_id": "GO:0003674",
            "name": "molecular_function",
            "namespace": "molecular_function",
            "is_obsolete": False,
            "children": [{"id": "GO:0005488", "relation": "is_a"}],
            "parent_ids": [],
        },
        {
            "term_id": "GO:0005488",
            "name": "binding",
            "namespace": "molecular_function",
            "is_obsolete": False,
            "children": [{"id": "GO:0005524", "relation": "is_a"}],
            "parent_ids": ["GO:0003674"],
        },
        {
            "term_id": "GO:0005524",
            "name": "ATP binding",
            "namespace": "molecular_function",
            "is_obsolete": False,
            "children": [],
            "parent_ids": ["GO:0005488"],
        },
    ]


@pytest.fixture
def builder_with_terms(config, sample_terms):
    raw_path = config["data"]["raw_path"]
    go_path = f"{raw_path}/go_terms.json"
    with open(go_path, "w", encoding="utf-8") as f:
        json.dump(sample_terms, f)
    return GODagBuilder(config=config)


# ---------------------------------------------------------------------------
# build — hierarquia completa
# ---------------------------------------------------------------------------


class TestBuild:
    def test_builds_full_hierarchy(self, builder_with_terms):
        graph = builder_with_terms.build(
            ["GO:0005524", "GO:0005488", "GO:0003674"]
        )

        assert len(graph) == 3
        assert graph.get_node("GO:0003674") is not None
        assert graph.get_node("GO:0005488") is not None
        assert graph.get_node("GO:0005524") is not None

    def test_parent_ids_correct(self, builder_with_terms):
        graph = builder_with_terms.build(
            ["GO:0005524", "GO:0005488", "GO:0003674"]
        )

        root = graph.get_node("GO:0003674")
        assert root.parent_ids == []

        binding = graph.get_node("GO:0005488")
        assert binding.parent_ids == ["GO:0003674"]

        atp = graph.get_node("GO:0005524")
        assert atp.parent_ids == ["GO:0005488"]

    def test_children_ids_correct(self, builder_with_terms):
        graph = builder_with_terms.build(
            ["GO:0005524", "GO:0005488", "GO:0003674"]
        )

        root = graph.get_node("GO:0003674")
        assert "GO:0005488" in root.children_ids

        binding = graph.get_node("GO:0005488")
        assert "GO:0005524" in binding.children_ids

        atp = graph.get_node("GO:0005524")
        assert atp.children_ids == []

    def test_get_ancestors(self, builder_with_terms):
        graph = builder_with_terms.build(
            ["GO:0005524", "GO:0005488", "GO:0003674"]
        )

        ancestors = graph.get_ancestors("GO:0005524")
        assert "GO:0005488" in ancestors
        assert "GO:0003674" in ancestors

    def test_ancestors_of_root_is_empty(self, builder_with_terms):
        graph = builder_with_terms.build(["GO:0003674"])
        assert graph.get_ancestors("GO:0003674") == []


# ---------------------------------------------------------------------------
# build — filtragem por termos relevantes
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_includes_ancestors_of_leaf(self, builder_with_terms):
        """Pedir apenas o leaf deve trazer toda a cadeia de ancestrais."""
        graph = builder_with_terms.build(["GO:0005524"])

        assert len(graph) == 3
        assert graph.get_node("GO:0003674") is not None
        assert graph.get_node("GO:0005488") is not None
        assert graph.get_node("GO:0005524") is not None

    def test_filters_out_unrelated_terms(self, config):
        terms = [
            {
                "term_id": "GO:0003674",
                "name": "molecular_function",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [],
                "parent_ids": [],
            },
            {
                "term_id": "GO:9999999",
                "name": "unrelated term",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [],
                "parent_ids": [],
            },
        ]
        raw_path = config["data"]["raw_path"]
        with open(f"{raw_path}/go_terms.json", "w", encoding="utf-8") as f:
            json.dump(terms, f)

        builder = GODagBuilder(config=config)
        graph = builder.build(["GO:0003674"])

        assert len(graph) == 1
        assert graph.get_node("GO:9999999") is None

    def test_unknown_terms_ignored(self, builder_with_terms):
        """Termos que nao existem no JSON sao ignorados sem erro."""
        graph = builder_with_terms.build(["GO:INEXISTENTE"])
        assert len(graph) == 0


# ---------------------------------------------------------------------------
# Termos obsoletos
# ---------------------------------------------------------------------------


class TestObsoleteTerms:
    def test_obsolete_terms_excluded(self, config):
        terms = [
            {
                "term_id": "GO:0003674",
                "name": "molecular_function",
                "namespace": "molecular_function",
                "is_obsolete": False,
                "children": [],
                "parent_ids": [],
            },
            {
                "term_id": "GO:0000001",
                "name": "obsolete term",
                "namespace": "molecular_function",
                "is_obsolete": True,
                "children": [],
                "parent_ids": ["GO:0003674"],
            },
        ]
        raw_path = config["data"]["raw_path"]
        with open(f"{raw_path}/go_terms.json", "w", encoding="utf-8") as f:
            json.dump(terms, f)

        builder = GODagBuilder(config=config)
        graph = builder.build(["GO:0003674", "GO:0000001"])

        assert graph.get_node("GO:0003674") is not None
        assert graph.get_node("GO:0000001") is None


# ---------------------------------------------------------------------------
# Erro quando go_terms.json nao existe
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_raises_file_not_found(self, config):
        builder = GODagBuilder(config=config)

        with pytest.raises(FileNotFoundError, match="go_terms.json"):
            builder.build(["GO:0005524"])
