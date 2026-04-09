"""Testes de integracao para o pipeline Modulos 1-3.

APIs externas (UniProt, QuickGO) sao mockadas.
A integracao real entre preprocessor e hierarchy builder e exercitada.
"""

import json

import pandas as pd
import pytest

from src.application.use_cases.prepare_data_pipeline import (
    PrepareDataPipeline,
    PipelineResult,
)
from src.domain.interfaces.data_source import ProteinDataSource
from src.infrastructure.hierarchy.go_dag_builder import GODagBuilder
from src.infrastructure.preprocessing.pandas_preprocessor import PandasPreprocessor


# ---------------------------------------------------------------------------
# Fake data source (substitui UniProtClient + GOClient sem rede)
# ---------------------------------------------------------------------------


class FakeProteinDataSource(ProteinDataSource):
    """Simula aquisicao de dados para testes de integracao."""

    def __init__(self, raw_path: str, go_terms_data: list[dict]):
        self._raw_path = raw_path
        self._go_terms_data = go_terms_data

    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        proteins = [
            {"protein_id": "P001", "sequence": "ACDEFGHIKLMNPQRSTVWY", "go_terms": "GO:0005524"},
            {"protein_id": "P002", "sequence": "ACDEFGHIKLMN", "go_terms": "GO:0003824"},
            {"protein_id": "P003", "sequence": "PQRSTVWYACDE", "go_terms": "GO:0005524;GO:0003824"},
            {"protein_id": "P004", "sequence": "FGHIKLMNPQRS", "go_terms": "GO:0005488"},
            {"protein_id": "P005", "sequence": "ACDEFGHIKLMN", "go_terms": "GO:0005524"},
            {"protein_id": "P006", "sequence": "STVWYACDEFGH", "go_terms": "GO:0003824"},
        ]
        df = pd.DataFrame(proteins[:limit])

        # Salva go_terms.json (necessario para o GODagBuilder)
        import os
        os.makedirs(self._raw_path, exist_ok=True)
        go_path = os.path.join(self._raw_path, "go_terms.json")
        with open(go_path, "w", encoding="utf-8") as f:
            json.dump(self._go_terms_data, f)

        return df

    def verify_conformity(self, data: pd.DataFrame) -> bool:
        required = {"protein_id", "sequence", "go_terms"}
        if not required.issubset(data.columns):
            return False
        if data.empty:
            return False
        return data[list(required)].notnull().all().all()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def go_terms_data():
    """Hierarquia GO minima para testes."""
    return [
        {
            "term_id": "GO:0003674",
            "name": "molecular_function",
            "namespace": "molecular_function",
            "is_obsolete": False,
            "children": [{"id": "GO:0005488", "relation": "is_a"}, {"id": "GO:0003824", "relation": "is_a"}],
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
        {
            "term_id": "GO:0003824",
            "name": "catalytic activity",
            "namespace": "molecular_function",
            "is_obsolete": False,
            "children": [],
            "parent_ids": ["GO:0003674"],
        },
    ]


@pytest.fixture
def config(tmp_path):
    raw_path = str(tmp_path / "raw")
    processed_path = str(tmp_path / "processed")
    return {
        "data": {
            "raw_path": raw_path,
            "processed_path": processed_path,
            "uniprot_limit": 6,
            "go_namespace": "molecular_function",
        },
        "model": {"random_seed": 42, "test_size": 0.2, "validation_size": 0.1},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def pipeline(config, go_terms_data):
    raw_path = config["data"]["raw_path"]
    data_source = FakeProteinDataSource(raw_path, go_terms_data)
    preprocessor = PandasPreprocessor(config=config)
    builder = GODagBuilder(config=config)
    return PrepareDataPipeline(data_source, preprocessor, builder)


# ---------------------------------------------------------------------------
# Testes de integracao
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.execute(limit=6)
        assert isinstance(result, PipelineResult)

    def test_proteins_are_preprocessed(self, pipeline):
        result = pipeline.execute(limit=6)
        df = result.proteins

        # Deve ter colunas de features do preprocessador
        assert "seq_length" in df.columns
        assert "molecular_weight" in df.columns
        assert "aa_A" in df.columns

        # Colunas originais preservadas
        assert "protein_id" in df.columns
        assert "go_terms" in df.columns

    def test_hierarchy_built_from_processed_data(self, pipeline):
        result = pipeline.execute(limit=6)
        graph = result.hierarchy

        assert len(graph) > 0
        # Todos os termos das proteinas devem estar no DAG
        assert graph.get_node("GO:0005524") is not None
        assert graph.get_node("GO:0003824") is not None
        # Ancestrais tambem
        assert graph.get_node("GO:0005488") is not None
        assert graph.get_node("GO:0003674") is not None

    def test_hierarchy_has_correct_structure(self, pipeline):
        result = pipeline.execute(limit=6)
        graph = result.hierarchy

        # ATP binding -> binding -> root
        atp = graph.get_node("GO:0005524")
        assert "GO:0005488" in atp.parent_ids

        binding = graph.get_node("GO:0005488")
        assert "GO:0003674" in binding.parent_ids

        root = graph.get_node("GO:0003674")
        assert root.parent_ids == []

    def test_ancestors_traversal_works(self, pipeline):
        result = pipeline.execute(limit=6)
        graph = result.hierarchy

        ancestors = graph.get_ancestors("GO:0005524")
        assert "GO:0005488" in ancestors
        assert "GO:0003674" in ancestors

    def test_no_duplicate_proteins(self, pipeline):
        result = pipeline.execute(limit=6)
        df = result.proteins

        assert df["protein_id"].is_unique


class TestPipelineDataFlow:
    def test_processed_csv_saved(self, pipeline, config):
        pipeline.execute(limit=6)

        import os
        processed_path = config["data"]["processed_path"]
        csv_path = os.path.join(processed_path, "proteins_clean.csv")
        assert os.path.exists(csv_path)

    def test_go_terms_json_saved(self, pipeline, config):
        pipeline.execute(limit=6)

        import os
        raw_path = config["data"]["raw_path"]
        json_path = os.path.join(raw_path, "go_terms.json")
        assert os.path.exists(json_path)

    def test_protein_count_consistent(self, pipeline):
        """Numero de proteinas no resultado <= limite solicitado."""
        result = pipeline.execute(limit=4)
        assert len(result.proteins) <= 4


class TestPipelineConformity:
    def test_rejects_nonconformant_data(self, config, go_terms_data):
        """Pipeline deve falhar se dados nao passam na conformidade."""

        class BadDataSource(ProteinDataSource):
            def fetch_proteins(self, limit: int) -> pd.DataFrame:
                return pd.DataFrame(columns=["protein_id", "sequence", "go_terms"])

            def verify_conformity(self, data: pd.DataFrame) -> bool:
                return False

        pipeline = PrepareDataPipeline(
            BadDataSource(),
            PandasPreprocessor(config=config),
            GODagBuilder(config=config),
        )

        with pytest.raises(ValueError, match="conformidade"):
            pipeline.execute(limit=10)


class TestExtractGoTerms:
    def test_extracts_unique_terms(self):
        df = pd.DataFrame({
            "go_terms": ["GO:0001;GO:0002", "GO:0002;GO:0003", "GO:0001"],
        })
        terms = PrepareDataPipeline._extract_go_terms(df)
        assert set(terms) == {"GO:0001", "GO:0002", "GO:0003"}

    def test_returns_sorted(self):
        df = pd.DataFrame({
            "go_terms": ["GO:0003;GO:0001;GO:0002"],
        })
        terms = PrepareDataPipeline._extract_go_terms(df)
        assert terms == sorted(terms)
