import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.domain.interfaces.classifier import HierarchicalClassifier
from src.infrastructure.preprocessing.pandas_preprocessor import (
    AMINO_ACIDS,
    _composition,
    _molecular_weight,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Extrai features de uma sequencia e classifica via classificador treinado."""

    def __init__(
        self,
        classifier: HierarchicalClassifier,
        scaler: StandardScaler | None = None,
    ):
        self._classifier = classifier
        self._scaler = scaler

    def predict(self, sequence: str) -> set[str]:
        """Classifica uma sequencia de proteina, retornando termos GO preditos."""
        features: dict[str, float] = {
            "seq_length": float(len(sequence)),
            "molecular_weight": _molecular_weight(sequence),
        }
        features.update(_composition(sequence))

        feature_cols = ["seq_length", "molecular_weight"] + [
            f"aa_{aa}" for aa in AMINO_ACIDS
        ]
        df = pd.DataFrame([features], columns=feature_cols)

        if self._scaler is not None:
            df[feature_cols] = self._scaler.transform(df[feature_cols])

        predictions = self._classifier.predict(df)

        terms: set[str] = set()
        for pred_str in predictions:
            for t in str(pred_str).split(";"):
                t = t.strip()
                if t:
                    terms.add(t)

        logger.info("Predicao: %d termos GO identificados", len(terms))
        return terms
