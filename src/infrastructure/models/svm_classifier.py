from sklearn.base import BaseEstimator
from sklearn.svm import SVC

from src.infrastructure.models._base_hierarchical import BaseHierarchicalLCN
from src.shared.config_loader import load_config


class SVMHierarchicalClassifier(BaseHierarchicalLCN):
    """Classificador hierarquico LCN usando SVM como estimador base."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        super().__init__(seed=cfg["model"]["random_seed"])
        self._kernel = cfg["model"].get("svm_kernel", "rbf")
        self._C = cfg["model"].get("svm_C", 1.0)

    def _create_estimator(self) -> BaseEstimator:
        return SVC(
            kernel=self._kernel,
            C=self._C,
            random_state=self._seed,
        )
