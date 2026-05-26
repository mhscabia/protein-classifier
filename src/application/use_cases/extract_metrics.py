import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.domain.entities.hierarchy_graph import HierarchyGraph
from src.infrastructure.evaluation.hierarchical_metrics import (
    HierarchicalMetricsEvaluator,
)
from src.infrastructure.persistence.model_persistence import (
    hierarchy_exists,
    load_hierarchy,
    model_exists,
    try_load_model,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)

FEATURE_PREFIXES = ("seq_length", "molecular_weight", "aa_", "esm_")


def _parse_terms(term_str: str) -> set[str]:
    return {t.strip() for t in str(term_str).split(";") if t.strip()}


@dataclass
class SampleMetric:
    protein_id: str
    y_true: str
    y_pred: str
    hF: float
    n_pred: int
    n_true: int
    n_inter: int


@dataclass
class MetricsExtractionResult:
    sample_size: int
    test_set_size: int
    hierarchical: dict
    flat: dict
    samples: list[SampleMetric]
    top_samples: list[SampleMetric]
    bottom_samples: list[SampleMetric]
    avg_predicted: float
    avg_leaf_predicted: float
    avg_true: float
    pct_hf_gt_05: float
    pct_hf_zero: float
    latency_ms_per_sample: float
    total_latency_s: float
    model_metadata: dict
    n_dag_nodes: int
    n_classifiers: int
    feature_dim: int
    config_snapshot: dict = field(default_factory=dict)


class ExtractMetricsUseCase:
    """Carrega modelo persistido e avalia em N amostras do test set."""

    def __init__(self, config: dict):
        self._config = config

    def execute(self) -> MetricsExtractionResult:
        persist_path = self._config["model"]["persist_path"]

        if not model_exists(persist_path):
            raise FileNotFoundError(
                f"Modelo não encontrado em {persist_path}. "
                "Rode 'python main.py' primeiro para treinar e persistir."
            )
        if not hierarchy_exists(persist_path):
            raise FileNotFoundError(
                f"DAG não encontrado em {persist_path}. "
                "Rode 'python main.py' primeiro para gerar a hierarquia."
            )

        classifier, _scaler, metadata = try_load_model(persist_path)
        hierarchy: HierarchyGraph = load_hierarchy(persist_path)

        proteins = self._load_proteins()
        X_sample, y_sample = self._sample_from_test_set(proteins)

        feature_cols = [
            c for c in X_sample.columns if c.startswith(FEATURE_PREFIXES)
        ]
        X_features = X_sample[feature_cols]

        logger.info("Predizendo %d amostras...", len(X_sample))
        t0 = time.time()
        y_pred = classifier.predict(X_features)
        total_latency = time.time() - t0
        latency_per_sample = (total_latency * 1000) / len(X_sample)

        evaluator = HierarchicalMetricsEvaluator(hierarchy)
        y_true_list = y_sample.tolist()
        hierarchical = evaluator.evaluate(y_true_list, y_pred)
        flat = evaluator.evaluate_flat(y_true_list, y_pred)

        samples = self._build_per_sample_metrics(
            X_sample["protein_id"].tolist(),
            y_true_list,
            y_pred,
            evaluator,
        )

        sorted_by_hf = sorted(samples, key=lambda s: s.hF)
        bottom = sorted_by_hf[:10]
        top = list(reversed(sorted_by_hf[-10:]))

        avg_predicted = float(np.mean([s.n_pred for s in samples]))
        avg_true = float(np.mean([s.n_true for s in samples]))
        avg_leaf = float(np.mean([
            len(hierarchy.get_leaf_predicted(_parse_terms(s.y_pred)))
            for s in samples
        ]))
        pct_gt_05 = float(np.mean([1 if s.hF > 0.5 else 0 for s in samples])) * 100
        pct_zero = float(np.mean([1 if s.hF == 0 else 0 for s in samples])) * 100

        n_classifiers = self._count_classifiers(classifier)

        return MetricsExtractionResult(
            sample_size=len(samples),
            test_set_size=int(len(proteins) * self._config["model"]["test_size"]),
            hierarchical=hierarchical,
            flat=flat,
            samples=samples,
            top_samples=top,
            bottom_samples=bottom,
            avg_predicted=avg_predicted,
            avg_leaf_predicted=avg_leaf,
            avg_true=avg_true,
            pct_hf_gt_05=pct_gt_05,
            pct_hf_zero=pct_zero,
            latency_ms_per_sample=latency_per_sample,
            total_latency_s=total_latency,
            model_metadata=metadata or {},
            n_dag_nodes=len(hierarchy),
            n_classifiers=n_classifiers,
            feature_dim=len(feature_cols),
            config_snapshot=self._snapshot_config(),
        )

    def _load_proteins(self) -> pd.DataFrame:
        csv_path = Path(self._config["data"]["processed_path"]) / "proteins_clean.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} não existe. Rode 'python main.py' para gerar."
            )
        return pd.read_csv(csv_path)

    def _sample_from_test_set(
        self, proteins: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        seed = self._config["model"]["random_seed"]
        test_size = self._config["model"]["test_size"]
        sample_size = self._config["evaluation"]["sample_size"]
        eval_seed = self._config["evaluation"].get("random_seed", seed)

        feature_cols = [
            c for c in proteins.columns if c.startswith(FEATURE_PREFIXES)
        ]
        carry_cols = ["protein_id"] + feature_cols

        X = proteins[carry_cols]
        y = proteins["go_terms"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )

        n = min(sample_size, len(X_test))
        rng = np.random.RandomState(eval_seed)
        idx = rng.choice(len(X_test), size=n, replace=False)
        return X_test.iloc[idx].reset_index(drop=True), y_test.iloc[idx].reset_index(drop=True)

    def _build_per_sample_metrics(
        self,
        protein_ids: list[str],
        y_true: list[str],
        y_pred: list[str],
        evaluator: HierarchicalMetricsEvaluator,
    ) -> list[SampleMetric]:
        samples = []
        for pid, t, p in zip(protein_ids, y_true, y_pred):
            single = evaluator.evaluate([t], [p])
            true_set = _parse_terms(t)
            pred_set = _parse_terms(p)
            samples.append(
                SampleMetric(
                    protein_id=pid,
                    y_true=t,
                    y_pred=p,
                    hF=single["hF"],
                    n_pred=len(pred_set),
                    n_true=len(true_set),
                    n_inter=len(pred_set & true_set),
                )
            )
        return samples

    def _count_classifiers(self, classifier) -> int:
        node_clfs = getattr(classifier, "_node_classifiers", None)
        if node_clfs is None:
            return 0
        return len(node_clfs)

    def _snapshot_config(self) -> dict:
        return {
            "uniprot_limit": self._config["data"]["uniprot_limit"],
            "use_esm": self._config["features"].get("use_esm", False),
            "esm_model": self._config["features"].get("esm_model"),
            "min_term_support": self._config["hierarchy"].get("min_term_support"),
            "test_size": self._config["model"]["test_size"],
            "random_seed": self._config["model"]["random_seed"],
        }
