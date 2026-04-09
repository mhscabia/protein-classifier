from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from src.infrastructure.models._base_hierarchical import BaseHierarchicalLCN
from src.shared.config_loader import load_config


class RandomForestHierarchicalClassifier(BaseHierarchicalLCN):
    """Classificador hierarquico LCN usando Random Forest como estimador base."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        super().__init__(seed=cfg["model"]["random_seed"])
        self._n_estimators = cfg["model"].get("rf_n_estimators", 100)

    def _create_estimator(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            random_state=self._seed,
            n_jobs=-1,
        )
