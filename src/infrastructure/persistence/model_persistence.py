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
