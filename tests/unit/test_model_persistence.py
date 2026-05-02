import pytest
from sklearn.preprocessing import StandardScaler

from src.infrastructure.persistence.model_persistence import (
    load_model,
    model_exists,
    save_model,
)


class FakeClassifier:
    def __init__(self, value):
        self.value = value


@pytest.fixture
def tmp_path_str(tmp_path):
    return str(tmp_path / "models")


def test_model_exists_returns_false_when_missing(tmp_path_str):
    assert not model_exists(tmp_path_str)


def test_save_creates_file(tmp_path_str):
    clf = FakeClassifier(42)
    scaler = StandardScaler()
    save_model(clf, scaler, {"version": "1"}, tmp_path_str)
    assert model_exists(tmp_path_str)


def test_load_returns_saved_objects(tmp_path_str):
    clf = FakeClassifier(99)
    scaler = StandardScaler()
    metadata = {"uniprot_limit": 500, "date": "2026-01-01"}
    save_model(clf, scaler, metadata, tmp_path_str)

    loaded_clf, loaded_scaler, loaded_meta = load_model(tmp_path_str)
    assert loaded_clf.value == 99
    assert isinstance(loaded_scaler, StandardScaler)
    assert loaded_meta["uniprot_limit"] == 500


def test_save_overwrites_existing(tmp_path_str):
    save_model(FakeClassifier(1), StandardScaler(), {}, tmp_path_str)
    save_model(FakeClassifier(2), StandardScaler(), {"v": "2"}, tmp_path_str)

    loaded_clf, _, loaded_meta = load_model(tmp_path_str)
    assert loaded_clf.value == 2
    assert loaded_meta["v"] == "2"
