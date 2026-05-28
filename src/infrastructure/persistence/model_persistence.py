import joblib
from pathlib import Path


def save_model(classifier, scaler, metadata: dict, path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"classifier": classifier, "scaler": scaler, "metadata": metadata},
        p / "model.joblib",
    )


def load_model(path: str) -> tuple:
    data = joblib.load(Path(path) / "model.joblib")
    return data["classifier"], data["scaler"], data["metadata"]


def model_exists(path: str) -> bool:
    return (Path(path) / "model.joblib").exists()


def save_hierarchy(hierarchy, path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(hierarchy, p / "hierarchy.joblib")


def load_hierarchy(path: str):
    return joblib.load(Path(path) / "hierarchy.joblib")


def hierarchy_exists(path: str) -> bool:
    return (Path(path) / "hierarchy.joblib").exists()


def try_load_model(path: str) -> tuple | None:
    if not model_exists(path):
        return None
    try:
        return load_model(path)
    except Exception:
        return None


def is_compatible_meta(metadata: dict, *, feature_dim: int | None = None) -> bool:
    """Verifica compatibilidade checando dimensão de features do modelo salvo."""
    if feature_dim is not None:
        saved_dim = metadata.get("feature_dim")
        if saved_dim is not None and int(saved_dim) != int(feature_dim):
            return False
    return True


def is_compatible(path: str, *, feature_dim: int | None = None) -> bool:
    loaded = try_load_model(path)
    if loaded is None:
        return False
    _, _, metadata = loaded
    return is_compatible_meta(metadata, feature_dim=feature_dim)
