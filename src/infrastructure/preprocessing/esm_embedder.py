from pathlib import Path

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.shared.logger import get_logger

logger = get_logger(__name__)


class ESMEmbedder:
    """Gera embeddings de proteinas via ESM-2 (mean pooling sobre residuos).

    Carrega o modelo HuggingFace de forma preguicosa (somente quando preciso).
    Cache em disco indexado por protein_ids — invalidado se o conjunto mudar.
    """

    def __init__(
        self,
        model_name: str,
        cache_path: str | None = None,
        max_length: int = 1022,
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._cache_path = Path(cache_path) if cache_path else None
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        self._tokenizer = None
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # lazy import
        import torch
        from rich.console import Console

        _console = Console()
        try:
            with _console.status(f"[cyan]Carregando modelo ESM '{self._model_name}' do cache local...[/cyan]", spinner="dots"):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name, local_files_only=True
                )
                self._model = AutoModel.from_pretrained(
                    self._model_name, local_files_only=True
                )
        except Exception:
            with _console.status("[cyan]Baixando modelo ESM do HuggingFace...[/cyan]", spinner="dots"):
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()
        self._torch = torch

    @property
    def embedding_dim(self) -> int:
        """Dimensionalidade dos embeddings (ex: 320 para esm2_t6_8M_UR50D)."""
        self._ensure_model_loaded()
        return int(self._model.config.hidden_size)

    def embed(
        self,
        sequences: list[str],
        protein_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Embeddings para uma lista de sequencias. Usa cache se disponivel."""
        if protein_ids is not None and self._cache_path is not None:
            cached = self._load_cache(protein_ids)
            if cached is not None:
                logger.info(
                    "Cache ESM reutilizado: %d embeddings de %s",
                    len(cached),
                    self._cache_path,
                )
                return cached

        self._ensure_model_loaded()
        embeddings = self._compute_batches(sequences)

        if protein_ids is not None and self._cache_path is not None:
            self._save_cache(protein_ids, embeddings)

        return embeddings

    def embed_single(self, sequence: str) -> np.ndarray:
        """Embedding (1D) para uma unica sequencia. Usado em inferencia."""
        return self.embed([sequence])[0]

    def _compute_batches(self, sequences: list[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        total = len(sequences)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Gerando embeddings ESM-2...", total=total)
            for start in range(0, total, self._batch_size):
                chunk = sequences[start : start + self._batch_size]
                tokens = self._tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                )
                with torch.no_grad():
                    outputs = self._model(**tokens)
                hidden = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()
                out.append(pooled)
                progress.advance(task, len(chunk))
        return np.vstack(out).astype(np.float32)

    def _load_cache(self, protein_ids: list[str]) -> np.ndarray | None:
        if not self._cache_path or not self._cache_path.exists():
            return None
        try:
            data = np.load(self._cache_path, allow_pickle=False)
            cached_ids = [str(x) for x in data["protein_ids"].tolist()]
            embeddings = data["embeddings"]
        except Exception as exc:
            logger.warning("Cache ESM corrompido (%s) — recomputando", exc)
            return None
        if cached_ids != list(protein_ids):
            logger.info("Cache ESM desatualizado (IDs mudaram) — recomputando")
            return None
        return embeddings

    def _save_cache(self, protein_ids: list[str], embeddings: np.ndarray) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._cache_path,
            embeddings=embeddings,
            protein_ids=np.array(protein_ids, dtype=str),
        )
        logger.info("Cache ESM salvo em %s", self._cache_path)
