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
        embedder=None,
    ):
        self._classifier = classifier
        self._scaler = scaler
        self._embedder = embedder

    def predict(self, sequence: str) -> set[str]:
        """Classifica uma sequencia de proteina, retornando termos GO preditos."""
        if self._embedder is not None:
            df = self._build_esm_features(sequence)
        else:
            df = self._build_manual_features(sequence)

        feature_cols = list(df.columns)
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

    def _build_manual_features(self, sequence: str) -> pd.DataFrame:
        features: dict[str, float] = {
            "seq_length": float(len(sequence)),
            "molecular_weight": _molecular_weight(sequence),
        }
        features.update(_composition(sequence))
        feature_cols = ["seq_length", "molecular_weight"] + [
            f"aa_{aa}" for aa in AMINO_ACIDS
        ]
        return pd.DataFrame([features], columns=feature_cols)

    def _build_esm_features(self, sequence: str) -> pd.DataFrame:
        embedding = self._embedder.embed_single(sequence)
        feature_cols = [f"esm_{i}" for i in range(embedding.shape[0])]
        return pd.DataFrame([embedding], columns=feature_cols)
