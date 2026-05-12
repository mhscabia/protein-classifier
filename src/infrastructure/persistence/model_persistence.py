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


def is_compatible(path: str, *, use_esm: bool, feature_dim: int | None = None) -> bool:
    """Verifica se o modelo persistido foi treinado com o mesmo regime de features."""
    if not model_exists(path):
        return False
    try:
        _, _, metadata = load_model(path)
    except Exception:
        return False
    saved_use_esm = metadata.get("use_esm")
    if saved_use_esm is None:
        return False
    if bool(saved_use_esm) != bool(use_esm):
        return False
    if feature_dim is not None:
        saved_dim = metadata.get("feature_dim")
        if saved_dim is not None and int(saved_dim) != int(feature_dim):
            return False
    return True
