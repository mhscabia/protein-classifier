import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.domain.interfaces.data_source import ProteinDataSource
from src.shared.config_loader import load_config
from src.shared.logger import get_logger

logger = get_logger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

GO_NAMESPACE_ROOTS = {
    "molecular_function": "0003674",
    "biological_process": "0008150",
    "cellular_component": "0005575",
}

GO_FIELD_BY_NAMESPACE = {
    "molecular_function": "go_f",
    "biological_process": "go_p",
    "cellular_component": "go_c",
}


def _extract_go_ids(go_text: str) -> list[str]:
    """Extrai IDs GO (ex: GO:0005524) de uma string no formato UniProt TSV."""
    if not go_text or pd.isna(go_text):
        return []
    return re.findall(r"GO:\d+", go_text)


class UniProtClient(ProteinDataSource):
    """Busca proteínas revisadas (Swiss-Prot) da API REST do UniProt."""

    def __init__(self, config: dict | None = None):
        self._config = config or load_config()
        self._raw_path = Path(self._config["data"]["raw_path"])
        self._namespace = self._config["data"]["go_namespace"]

    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        go_root = GO_NAMESPACE_ROOTS[self._namespace]
        go_field = GO_FIELD_BY_NAMESPACE[self._namespace]

        query = f"(reviewed:true) AND (go:{go_root})"
        page_size = min(limit, 500)

        all_rows: list[pd.DataFrame] = []
        collected = 0

        next_url: str | None = (
            f"{UNIPROT_SEARCH_URL}"
            f"?query={query}"
            f"&format=tsv"
            f"&fields=accession,sequence,{go_field}"
            f"&size={page_size}"
        )

        while next_url and collected < limit:
            logger.info("Requisitando UniProt (%d/%d coletadas)...", collected, limit)
            response = requests.get(next_url, timeout=60)
            response.raise_for_status()

            page_df = pd.read_csv(StringIO(response.text), sep="\t")
            if page_df.empty:
                break

            remaining = limit - collected
            page_df = page_df.head(remaining)
            all_rows.append(page_df)
            collected += len(page_df)

            next_url = self._parse_next_link(response.headers.get("Link", ""))

        if not all_rows:
            logger.warning("Nenhuma proteína retornada pela UniProt.")
            return pd.DataFrame(columns=["protein_id", "sequence", "go_terms"])

        raw_df = pd.concat(all_rows, ignore_index=True)
        df = self._normalize_dataframe(raw_df)

        self._raw_path.mkdir(parents=True, exist_ok=True)
        output_path = self._raw_path / "proteins.csv"
        df.to_csv(output_path, index=False)
        logger.info("Salvas %d proteínas em %s", len(df), output_path)

        return df

    def verify_conformity(self, data: pd.DataFrame) -> bool:
        required_columns = {"protein_id", "sequence", "go_terms"}

        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            logger.error("Colunas ausentes: %s", missing)
            return False

        if data.empty:
            logger.error("DataFrame vazio")
            return False

        for col in required_columns:
            null_count = data[col].isnull().sum()
            if null_count > 0:
                logger.error("Coluna '%s' contém %d valores nulos", col, null_count)
                return False

        for col in required_columns:
            empty_count = (data[col].astype(str).str.strip() == "").sum()
            if empty_count > 0:
                logger.error("Coluna '%s' contém %d valores vazios", col, empty_count)
                return False

        dup_count = data["protein_id"].duplicated().sum()
        if dup_count > 0:
            logger.warning("Encontrados %d protein_ids duplicados", dup_count)

        logger.info("Verificação de conformidade OK — %d registros válidos", len(data))
        return True

    def _normalize_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Converte colunas do formato TSV UniProt para o schema do projeto."""
        col_map = {
            "Entry": "protein_id",
            "Sequence": "sequence",
        }
        go_col = [c for c in raw_df.columns if "Gene Ontology" in c]
        if go_col:
            col_map[go_col[0]] = "go_terms_raw"

        df = raw_df.rename(columns=col_map)

        if "go_terms_raw" in df.columns:
            df["go_terms"] = df["go_terms_raw"].apply(
                lambda x: ";".join(_extract_go_ids(x))
            )
            df = df.drop(columns=["go_terms_raw"])
        else:
            df["go_terms"] = ""

        df = df[["protein_id", "sequence", "go_terms"]]

        # Remove linhas sem GO terms após extração
        df = df[df["go_terms"].str.len() > 0].reset_index(drop=True)

        return df

    @staticmethod
    def _parse_next_link(link_header: str) -> str | None:
        """Extrai URL da próxima página do header Link da UniProt."""
        if not link_header:
            return None
        # Usa regex para evitar falso split em vírgulas dentro da URL
        for match in re.finditer(r'<([^>]+)>\s*;\s*rel="next"', link_header):
            return match.group(1)
        return None
