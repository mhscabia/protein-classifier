import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.domain.interfaces.preprocessor import ProteinPreprocessor
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VALID_SEQUENCE_PATTERN = re.compile(f"^[{AMINO_ACIDS}]+$", re.IGNORECASE)


class PandasPreprocessor(ProteinPreprocessor):
    """Limpeza e extração de features numéricas a partir de proteínas."""

    def __init__(self, config: dict | None = None, embedder=None):
        self._config = config or load_config()
        self._processed_path = Path(self._config["data"]["processed_path"])
        self._scaler: StandardScaler | None = None
        self._embedder = embedder

    @property
    def scaler(self) -> StandardScaler | None:
        return self._scaler

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        initial_count = len(df)

        df = df.dropna(subset=["protein_id", "sequence", "go_terms"])
        dropped_nulls = initial_count - len(df)
        if dropped_nulls > 0:
            logger.info("Removidas %d linhas com valores nulos", dropped_nulls)

        df = df[df["sequence"].astype(str).str.strip().str.len() > 0]
        df = df[df["go_terms"].astype(str).str.strip().str.len() > 0]

        before_dup = len(df)
        df = df.drop_duplicates(subset=["protein_id"], keep="first")
        dropped_dups = before_dup - len(df)
        if dropped_dups > 0:
            logger.info("Removidas %d linhas com protein_id duplicado", dropped_dups)

        valid_mask = df["sequence"].apply(
            lambda s: bool(VALID_SEQUENCE_PATTERN.match(str(s)))
        )
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.info(
                "Removidas %d linhas com caracteres inválidos na sequência",
                invalid_count,
            )
        df = df[valid_mask].reset_index(drop=True)

        logger.info("Limpeza concluída: %d → %d registros", initial_count, len(df))
        return df

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        feature_cols = self._add_esm_features(df)

        self._scaler = StandardScaler()
        df[feature_cols] = self._scaler.fit_transform(df[feature_cols])

        self._processed_path.mkdir(parents=True, exist_ok=True)
        output_path = self._processed_path / "proteins_clean.csv"
        df.to_csv(output_path, index=False)
        logger.info("Salvas %d proteínas processadas em %s", len(df), output_path)

        return df

    def _add_esm_features(self, df: pd.DataFrame) -> list[str]:
        sequences = df["sequence"].astype(str).tolist()
        protein_ids = df["protein_id"].astype(str).tolist()
        embeddings = self._embedder.embed(sequences, protein_ids=protein_ids)
        feature_cols = [f"esm_{i}" for i in range(embeddings.shape[1])]
        for i, col in enumerate(feature_cols):
            df[col] = embeddings[:, i]
        return feature_cols
