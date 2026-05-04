import numpy as np
import pytest

from src.infrastructure.preprocessing.esm_embedder import ESMEmbedder


class _FakeEmbedder(ESMEmbedder):
    """Substitui o modelo HuggingFace por um stub deterministico.

    Cada sequencia gera um embedding de 8 dims onde o valor i e a contagem
    do i-esimo aminoacido em "ACDEFGHI" (para que sequencias diferentes
    gerem embeddings diferentes).
    """

    DIM = 8
    ALPHABET = "ACDEFGHI"

    def _ensure_model_loaded(self) -> None:
        self._tokenizer = object()
        self._model = object()

    def _compute_batches(self, sequences):
        out = []
        for seq in sequences:
            vec = np.array(
                [float(seq.upper().count(a)) for a in self.ALPHABET],
                dtype=np.float32,
            )
            out.append(vec)
        return np.vstack(out)


@pytest.fixture
def embedder(tmp_path):
    cache = tmp_path / "esm_cache.npz"
    return _FakeEmbedder(
        model_name="fake/model",
        cache_path=str(cache),
        max_length=1022,
        batch_size=4,
    )


def test_embed_single_returns_1d_array(embedder):
    out = embedder.embed_single("ACDEFG")
    assert out.shape == (_FakeEmbedder.DIM,)


def test_different_sequences_produce_different_embeddings(embedder):
    a = embedder.embed_single("AAAAAA")
    b = embedder.embed_single("CCCCCC")
    assert not np.allclose(a, b)


def test_embed_batch_shape(embedder):
    seqs = ["AAA", "CCC", "DDD", "EEE", "FFF"]
    out = embedder.embed(seqs)
    assert out.shape == (5, _FakeEmbedder.DIM)


def test_cache_is_created_and_reused(tmp_path):
    cache_path = tmp_path / "cache.npz"
    embedder = _FakeEmbedder(
        model_name="fake/model",
        cache_path=str(cache_path),
        batch_size=4,
    )
    seqs = ["AAA", "CCC"]
    ids = ["P1", "P2"]
    first = embedder.embed(seqs, protein_ids=ids)
    assert cache_path.exists()

    embedder2 = _FakeEmbedder(
        model_name="fake/model",
        cache_path=str(cache_path),
        batch_size=4,
    )

    def _fail(*args, **kwargs):
        raise AssertionError("nao deveria recomputar — cache deveria ser usado")

    embedder2._compute_batches = _fail  # type: ignore[assignment]
    second = embedder2.embed(seqs, protein_ids=ids)
    assert np.array_equal(first, second)
