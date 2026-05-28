from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

from src.shared.logger import get_logger

logger = get_logger(__name__)
console = Console()

_MODEL_FILES = ("model.joblib", "hierarchy.joblib")


def download_models(repo_id: str, dest_path: str) -> None:
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILES:
        console.print(f"  Baixando [bold]{filename}[/bold] de [cyan]{repo_id}[/cyan]...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_path,
            repo_type="model",
        )
        console.print(f"  [green]✓[/green] {filename} salvo em [dim]{dest_path}[/dim]")
