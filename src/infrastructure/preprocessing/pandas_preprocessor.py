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

MOLECULAR_WEIGHTS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13,
    "F": 165.19, "G": 75.03, "H": 155.16, "I": 131.17,
    "K": 146.19, "L": 131.17, "M": 149.21, "N": 132.12,
    "P": 115.13, "Q": 146.15, "R": 174.20, "S": 105.09,
    "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}

WATER_WEIGHT = 18.015


def _composition(sequence: str) -> dict[str, float]:
    """Calcula a fração de cada aminoácido na sequência."""
    seq = sequence.upper()
    length = len(seq)
    if length == 0:
        return {f"aa_{aa}": 0.0 for aa in AMINO_ACIDS}
    return {f"aa_{aa}": seq.count(aa) / length for aa in AMINO_ACIDS}


def _molecular_weight(sequence: str) -> float:
    """Estima o peso molecular (Da) pela soma dos resíduos menos água."""
    seq = sequence.upper()
    if not seq:
        return 0.0
    weight = sum(MOLECULAR_WEIGHTS.get(aa, 0.0) for aa in seq)
    weight -= (len(seq) - 1) * WATER_WEIGHT
    return round(weight, 2)


class PandasPreprocessor(ProteinPreprocessor):
    """Limpeza e extração de features numéricas a partir de proteínas."""

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        self._processed_path = Path(self._config["data"]["processed_path"])
        self._scaler: StandardScaler | None = None

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        initial_count = len(df)

        df = df.dropna(subset=["protein_id", "sequence", "go_terms"])
        after_nulls = len(df)
        dropped_nulls = initial_count - after_nulls
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
        df = df[valid_mask]

        df = df.reset_index(drop=True)
        logger.info(
            "Limpeza concluída: %d → %d registros", initial_count, len(df)
        )
        return df

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["seq_length"] = df["sequence"].apply(len)
        df["molecular_weight"] = df["sequence"].apply(_molecular_weight)

        composition_df = df["sequence"].apply(_composition).apply(pd.Series)
        df = pd.concat([df, composition_df], axis=1)

        feature_cols = ["seq_length", "molecular_weight"] + [
            f"aa_{aa}" for aa in AMINO_ACIDS
        ]

        self._scaler = StandardScaler()
        df[feature_cols] = self._scaler.fit_transform(df[feature_cols])

        self._processed_path.mkdir(parents=True, exist_ok=True)
        output_path = self._processed_path / "proteins_clean.csv"
        df.to_csv(output_path, index=False)
        logger.info("Salvas %d proteínas processadas em %s", len(df), output_path)

        return df
