import json
import time
from pathlib import Path

import requests

from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)

QUICKGO_BASE_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
REQUEST_DELAY = 0.15  # segundos entre requisições para respeitar rate limit
BATCH_SIZE = 25  # máximo de IDs por requisição ao QuickGO


class GOClient:
    """Busca termos GO e suas relações hierárquicas via API do QuickGO."""

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        self._raw_path = Path(self._config["data"]["raw_path"])
        self._namespace = self._config["data"]["go_namespace"]
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def fetch_go_terms(self, go_term_ids: list[str]) -> list[dict]:
        """Busca termos GO e seus ancestrais, retornando lista de dicts com
        id, name, namespace, is_obsolete e parent_ids."""
        seed_ids = set(go_term_ids)
        logger.info("Coletando ancestrais para %d termos GO...", len(seed_ids))

        ancestor_ids = self._collect_all_ancestors(seed_ids)
        all_ids = seed_ids | ancestor_ids
        logger.info(
            "Total de termos a buscar: %d (%d seeds + %d ancestrais)",
            len(all_ids),
            len(seed_ids),
            len(ancestor_ids),
        )

        term_details = self._fetch_details_batch(all_ids)
        self._fill_parent_ids(term_details)

        terms_list = list(term_details.values())

        self._raw_path.mkdir(parents=True, exist_ok=True)
        output_path = self._raw_path / "go_terms.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(terms_list, f, ensure_ascii=False, indent=2)
        logger.info("Salvos %d termos GO em %s", len(terms_list), output_path)

        return terms_list

    def _collect_all_ancestors(self, seed_ids: set[str]) -> set[str]:
        """Busca ancestrais de cada seed term via QuickGO, com deduplicação."""
        all_ancestors: set[str] = set()
        processed: set[str] = set()
        to_process = list(seed_ids)

        for i, term_id in enumerate(to_process):
            if term_id in processed or term_id in all_ancestors:
                continue

            ancestors = self._fetch_ancestors(term_id)
            processed.add(term_id)
            new_ancestors = ancestors - all_ancestors
            all_ancestors.update(ancestors)

            if (i + 1) % 100 == 0:
                logger.info(
                    "Progresso ancestrais: %d/%d termos processados, "
                    "%d ancestrais únicos",
                    i + 1,
                    len(to_process),
                    len(all_ancestors),
                )

        return all_ancestors

    def _fetch_ancestors(self, term_id: str) -> set[str]:
        """Retorna IDs de todos os ancestrais de um termo GO."""
        url = f"{QUICKGO_BASE_URL}/{term_id}/ancestors"
        params = {"relations": "is_a,part_of"}

        try:
            time.sleep(REQUEST_DELAY)
            response = self._session.get(url, params=params, timeout=30)
            if response.status_code == 404:
                logger.debug("Termo %s não encontrado no QuickGO", term_id)
                return set()
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            ancestor_ids: set[str] = set()
            for item in results:
                if isinstance(item, dict):
                    ancestor_ids.update(item.get("ancestors", []))
            return ancestor_ids - {term_id}
        except requests.RequestException as e:
            logger.warning("Falha ao buscar ancestrais de %s: %s", term_id, e)
            return set()

    def _fetch_details_batch(self, term_ids: set[str]) -> dict[str, dict]:
        """Busca detalhes de termos GO em lotes."""
        term_details: dict[str, dict] = {}
        ids_list = sorted(term_ids)

        for i in range(0, len(ids_list), BATCH_SIZE):
            batch = ids_list[i : i + BATCH_SIZE]
            ids_param = ",".join(batch)
            url = f"{QUICKGO_BASE_URL}/{ids_param}"

            try:
                time.sleep(REQUEST_DELAY)
                response = self._session.get(url, timeout=30)
                if response.status_code == 404:
                    logger.debug("Lote não encontrado: %s", ids_param[:80])
                    continue
                response.raise_for_status()
                data = response.json()

                for result in data.get("results", []):
                    term_id = result.get("id", "")
                    if not term_id:
                        continue
                    term_details[term_id] = {
                        "term_id": term_id,
                        "name": result.get("name", ""),
                        "namespace": result.get("aspect", ""),
                        "is_obsolete": result.get("isObsolete", False),
                        "children": [
                            {
                                "id": c.get("id", ""),
                                "relation": c.get("relation", "is_a"),
                            }
                            for c in result.get("children", [])
                        ],
                        "parent_ids": [],
                    }
            except requests.RequestException as e:
                logger.warning("Falha ao buscar lote de termos: %s", e)

            if (i // BATCH_SIZE + 1) % 20 == 0:
                logger.info(
                    "Progresso detalhes: %d/%d termos buscados",
                    min(i + BATCH_SIZE, len(ids_list)),
                    len(ids_list),
                )

        return term_details

    @staticmethod
    def _fill_parent_ids(term_details: dict[str, dict]) -> None:
        """Deriva parent_ids a partir dos children de cada termo.

        Se o termo A lista B como child (is_a), então A é parent de B.
        """
        for term_id, info in term_details.items():
            for child in info.get("children", []):
                child_id = child["id"]
                if child_id in term_details:
                    parent_list = term_details[child_id]["parent_ids"]
                    if term_id not in parent_list:
                        parent_list.append(term_id)
