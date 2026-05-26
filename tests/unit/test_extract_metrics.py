import pytest

from src.domain.entities.hierarchy_graph import FunctionNode, HierarchyGraph
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
)
from src.application.use_cases.extract_metrics import (
    ExtractMetricsUseCase,
    MetricsExtractionResult,
    SampleMetric,
    _parse_terms,
)
from src.infrastructure.reporting.markdown_report_writer import MarkdownReportWriter


@pytest.fixture
def hierarchy():
    graph = HierarchyGraph()
    graph.add_node(FunctionNode(
        term_id="GO:0003674", name="molecular_function",
        parent_ids=[], children_ids=["GO:0005488"],
    ))
    graph.add_node(FunctionNode(
        term_id="GO:0005488", name="binding",
        parent_ids=["GO:0003674"], children_ids=["GO:0005524"],
    ))
    graph.add_node(FunctionNode(
        term_id="GO:0005524", name="ATP binding",
        parent_ids=["GO:0005488"], children_ids=[],
    ))
    return graph


@pytest.fixture
def base_config():
    return {
        "data": {"uniprot_limit": 10000, "processed_path": "data/processed/"},
        "model": {"random_seed": 42, "test_size": 0.2, "persist_path": "data/models/"},
        "hierarchy": {"min_term_support": 20},
        "features": {"use_esm": True, "esm_model": "facebook/esm2_t6_8M_UR50D"},
        "output": {"path": "data/output/"},
        "evaluation": {"sample_size": 100, "random_seed": 42},
    }


class TestBuildPerSampleMetrics:
    def test_perfect_match_gives_hf_1(self, hierarchy, base_config):
        uc = ExtractMetricsUseCase(base_config)
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        samples = uc._build_per_sample_metrics(
            protein_ids=["P1"],
            y_true=["GO:0005524"],
            y_pred=["GO:0005524"],
            evaluator=evaluator,
        )
        assert len(samples) == 1
        assert samples[0].hF == pytest.approx(1.0)
        assert samples[0].n_pred == 1
        assert samples[0].n_true == 1
        assert samples[0].n_inter == 1

    def test_empty_prediction_gives_hf_0(self, hierarchy, base_config):
        uc = ExtractMetricsUseCase(base_config)
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        samples = uc._build_per_sample_metrics(
            protein_ids=["P1"],
            y_true=["GO:0005524"],
            y_pred=[""],
            evaluator=evaluator,
        )
        assert samples[0].hF == pytest.approx(0.0)
        assert samples[0].n_pred == 0
        assert samples[0].n_inter == 0

    def test_multi_label_counts(self, hierarchy, base_config):
        uc = ExtractMetricsUseCase(base_config)
        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        samples = uc._build_per_sample_metrics(
            protein_ids=["P1"],
            y_true=["GO:0005524;GO:0003674"],
            y_pred=["GO:0005488"],
            evaluator=evaluator,
        )
        assert samples[0].n_true == 2
        assert samples[0].n_pred == 1


class TestSnapshotConfig:
    def test_extracts_relevant_keys(self, base_config):
        uc = ExtractMetricsUseCase(base_config)
        snap = uc._snapshot_config()
        assert snap["uniprot_limit"] == 10000
        assert snap["use_esm"] is True
        assert snap["min_term_support"] == 20
        assert snap["random_seed"] == 42


class TestMarkdownReportWriter:
    def _make_result(self, samples_count=10) -> MetricsExtractionResult:
        samples = [
            SampleMetric(
                protein_id=f"P{i}",
                y_true="GO:0005524",
                y_pred="GO:0005524" if i % 2 == 0 else "GO:0003674",
                hF=1.0 if i % 2 == 0 else 0.5,
                n_pred=1,
                n_true=1,
                n_inter=1 if i % 2 == 0 else 0,
            )
            for i in range(samples_count)
        ]
        return MetricsExtractionResult(
            sample_size=samples_count,
            test_set_size=2000,
            hierarchical={"hP": 0.85, "hR": 0.30, "hF": 0.44},
            flat={"flat_P": 0.50, "flat_R": 0.05, "flat_F": 0.09},
            samples=samples,
            top_samples=samples[:10],
            bottom_samples=samples[:10],
            avg_predicted=5.5,
            avg_leaf_predicted=2.0,
            avg_true=3.0,
            pct_hf_gt_05=50.0,
            pct_hf_zero=10.0,
            latency_ms_per_sample=12.5,
            total_latency_s=1.25,
            model_metadata={
                "classifier_name": "LCN",
                "uniprot_limit": 10000,
                "date": "2026-05-26",
                "use_esm": True,
            },
            n_dag_nodes=450,
            n_classifiers=400,
            feature_dim=320,
            config_snapshot={
                "uniprot_limit": 10000,
                "use_esm": True,
                "min_term_support": 20,
                "random_seed": 42,
            },
        )

    def test_renders_all_sections(self, base_config):
        writer = MarkdownReportWriter(base_config)
        md = writer._render(self._make_result())
        for heading in [
            "## 1. Estado do modelo",
            "## 2. Métricas agregadas",
            "## 3. Distribuição por amostra",
            "## 4. Top-10",
            "## 5. Top-10",
            "## 6. Plano de testes",
            "## 7. Notas",
        ]:
            assert heading in md

    def test_writes_metrics_values(self, base_config):
        writer = MarkdownReportWriter(base_config)
        md = writer._render(self._make_result())
        assert "0.8500" in md  # hP
        assert "0.3000" in md  # hR
        assert "0.4400" in md  # hF
        assert "0.0900" in md  # flat_F
        assert "450" in md     # n_dag_nodes
        assert "320 dimensões" in md  # feature dim

    def test_ratio_uses_infinity_when_flat_is_zero(self, base_config):
        result = self._make_result()
        result.flat = {"flat_P": 0.0, "flat_R": 0.0, "flat_F": 0.0}
        md = MarkdownReportWriter(base_config)._render(result)
        assert "flat_F = 0" in md
