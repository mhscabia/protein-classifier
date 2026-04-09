# CLAUDE.md — Contexto do Projeto TCC
## Previsão de Função de Proteínas com Classificação Hierárquica

> Este arquivo é lido automaticamente pelo Claude Code a cada sessão.
> Ele define a arquitetura, convenções, escopo e estado atual do projeto.
> **Nunca remova ou edite este arquivo sem instrução explícita.**

---

## IDENTIDADE DO PROJETO

| Campo | Valor |
|---|---|
| Aluno | Matheus Henrique Scabia de Jesus |
| Orientador | Julio Cesar Nievola |
| Instituição | PUCPR — Engenharia de Computação |
| Fase atual | 3º bimestre IOPT — Protótipo |
| Entrega crítica | 22/04/2026 — Projeto Físico Revisado |
| Defesa | 28–30/04/2026 — Defesa do Protótipo |

---

## OBJETIVO DO SISTEMA

Desenvolver um pipeline de aprendizado de máquina que receba dados de proteínas
e classifique suas funções biológicas utilizando **classificação hierárquica**,
respeitando a estrutura DAG (Directed Acyclic Graph) da Gene Ontology (GO).

Diferente da classificação flat: as dependências entre classes são levadas em conta.
Cada nó filho herda características do nó pai na hierarquia.

---

## DATASET

- **Fonte de proteínas:** UniProt Swiss-Prot (API pública)
- **Fonte da hierarquia:** Gene Ontology — subontologia `molecular_function`
- **Limite do protótipo:** 500 proteínas (`config.yaml`)
- **Seed fixa:** 42 (reprodutibilidade obrigatória)

---

## ARQUITETURA — CLEAN ARCHITECTURE ESTRITA

```
protein_classifier/
│
├── CLAUDE.md                    ← ESTE ARQUIVO
├── main.py                      ← Ponto de entrada
├── config.yaml                  ← Parâmetros centralizados
├── requirements.txt
│
├── src/
│   ├── domain/                  ← Regras de negócio puras (zero dependência externa)
│   │   ├── entities/            ← Protein, FunctionNode, HierarchyGraph
│   │   └── interfaces/          ← ABCs de cada módulo (contratos)
│   │
│   ├── application/             ← Casos de uso (orquestra domínio)
│   │   └── use_cases/
│   │
│   ├── infrastructure/          ← Implementações concretas
│   │   ├── data_sources/        ← Módulo 1: cliente UniProt + GO
│   │   ├── preprocessing/       ← Módulo 2: Pandas
│   │   ├── hierarchy/           ← Módulo 3: construção do DAG
│   │   ├── models/              ← Módulo 4: Random Forest + SVM
│   │   ├── evaluation/          ← Módulo 5: métricas hierárquicas
│   │   └── prediction/          ← Módulo 6: pipeline de inferência
│   │
│   └── shared/                  ← Logging, config loader, exceptions
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
│   ├── raw/                     ← Dados brutos (nunca editar manualmente)
│   └── processed/               ← Dados após pré-processamento
│
└── notebooks/                   ← Exploração/visualização (não entra no pipeline)
```

### Regra de dependência (NUNCA violar)
```
infrastructure → application → domain
```
O domínio não importa nada de fora. Infraestrutura depende de abstrações, não de concretos.

---

## MÓDULOS FUNCIONAIS (em ordem de implementação)

### Módulo 1 — Aquisição de Dados
- **Interface:** `src/domain/interfaces/data_source.py` → `ProteinDataSource`
- **Implementação:** `src/infrastructure/data_sources/uniprot_client.py`
- **Responsabilidade:** buscar proteínas da UniProt Swiss-Prot via API REST e termos GO via OBO Foundry
- **Saída:** `data/raw/proteins.csv` e `data/raw/go_terms.json`
- **Obrigatório:** script de verificação de conformidade (`verify_conformity`)

### Módulo 2 — Pré-processamento
- **Interface:** `src/domain/interfaces/preprocessor.py` → `ProteinPreprocessor`
- **Implementação:** `src/infrastructure/preprocessing/pandas_preprocessor.py`
- **Responsabilidade:** limpeza (nulos, duplicatas), normalização de features
- **Ferramenta principal:** Pandas
- **Saída:** `data/processed/proteins_clean.csv`

### Módulo 3 — Construção da Hierarquia
- **Interface:** `src/domain/interfaces/hierarchy_builder.py` → `HierarchyBuilder`
- **Implementação:** `src/infrastructure/hierarchy/go_dag_builder.py`
- **Responsabilidade:** montar o DAG de termos GO com relações `is_a` e `part_of`
- **Entidade central:** `HierarchyGraph` com método `get_ancestors(term_id)`

### Módulo 4 — Treinamento (PROTÓTIPO: RF + SVM)
- **Interface:** `src/domain/interfaces/classifier.py` → `HierarchicalClassifier`
- **Implementações:**
  - `src/infrastructure/models/random_forest_classifier.py`
  - `src/infrastructure/models/svm_classifier.py`
- **Ferramenta:** scikit-learn
- **⚠️ Redes neurais (TensorFlow/Keras) ficam para a implementação final — 4º bimestre**

### Módulo 5 — Avaliação
- **Interface:** `src/domain/interfaces/evaluator.py` → `HierarchicalEvaluator`
- **Implementação:** `src/infrastructure/evaluation/hierarchical_metrics.py`
- **Métricas obrigatórias:** hP (hierárquica precision), hR (hierárquica recall), hF (hierárquica F-measure)
- **Comparação:** incluir baseline flat (sem considerar hierarquia)
- **Ferramentas:** NumPy + Matplotlib

### Módulo 6 — Previsão
- **Interface:** implícita no caso de uso `ClassifyProteinUseCase`
- **Implementação:** `src/infrastructure/prediction/inference_pipeline.py`
- **Responsabilidade:** receber nova sequência → retornar nó GO mais próximo
- **Saída:** visualização básica com Matplotlib

---

## ENTIDADES DO DOMÍNIO

### `src/domain/entities/protein.py`
```python
from dataclasses import dataclass

@dataclass
class Protein:
    protein_id: str
    sequence: str
    go_terms: list[str]
```

### `src/domain/entities/hierarchy_graph.py`
```python
from dataclasses import dataclass, field

@dataclass
class FunctionNode:
    term_id: str        # Ex: "GO:0003674"
    name: str
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)

class HierarchyGraph:
    """DAG de termos GO — estrutura central do sistema."""

    def __init__(self):
        self._nodes: dict[str, FunctionNode] = {}

    def add_node(self, node: FunctionNode) -> None:
        self._nodes[node.term_id] = node

    def get_node(self, term_id: str) -> FunctionNode | None:
        return self._nodes.get(term_id)

    def get_ancestors(self, term_id: str) -> list[str]:
        """Retorna todos os ancestrais — essencial para métricas hierárquicas."""
        ancestors = []
        queue = list(self._nodes[term_id].parent_ids) if term_id in self._nodes else []
        while queue:
            parent_id = queue.pop(0)
            if parent_id not in ancestors:
                ancestors.append(parent_id)
                if parent_id in self._nodes:
                    queue.extend(self._nodes[parent_id].parent_ids)
        return ancestors

    def __len__(self) -> int:
        return len(self._nodes)
```

---

## INTERFACES (ABCs) DO DOMÍNIO

### `src/domain/interfaces/data_source.py`
```python
from abc import ABC, abstractmethod
import pandas as pd

class ProteinDataSource(ABC):
    @abstractmethod
    def fetch_proteins(self, limit: int) -> pd.DataFrame:
        """Retorna DataFrame: protein_id, sequence, go_terms."""
        ...
    @abstractmethod
    def verify_conformity(self, data: pd.DataFrame) -> bool:
        """Verifica colunas obrigatórias e ausência de nulos críticos."""
        ...
```

### `src/domain/interfaces/preprocessor.py`
```python
from abc import ABC, abstractmethod
import pandas as pd

class ProteinPreprocessor(ABC):
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame: ...
```

### `src/domain/interfaces/hierarchy_builder.py`
```python
from abc import ABC, abstractmethod
from src.domain.entities.hierarchy_graph import HierarchyGraph

class HierarchyBuilder(ABC):
    @abstractmethod
    def build(self, go_terms: list[str]) -> HierarchyGraph: ...
```

### `src/domain/interfaces/classifier.py`
```python
from abc import ABC, abstractmethod
import pandas as pd
from src.domain.entities.hierarchy_graph import HierarchyGraph

class HierarchicalClassifier(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, hierarchy: HierarchyGraph) -> None: ...
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list[str]: ...
```

### `src/domain/interfaces/evaluator.py`
```python
from abc import ABC, abstractmethod

class HierarchicalEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        """Retorna: {'hP': float, 'hR': float, 'hF': float}"""
        ...
```

---

## CONVENÇÕES DE CÓDIGO — OBRIGATÓRIAS

### PEP 8 + Clean Code
- Nomes de variáveis e funções: `snake_case`
- Nomes de classes: `PascalCase`
- Funções com **responsabilidade única** — se passar de 20 linhas, questionar se deve ser dividida
- **Sem comentários óbvios** — o código deve se explicar pelos nomes
- Comentários apenas onde a intenção não é óbvia pelo código
- **DRY** — nunca duplicar lógica; extrair para função/classe compartilhada

### SOLID
- **S:** cada classe tem uma única razão para mudar
- **O:** novas implementações de classificadores/datasources são adicionadas, não modificam o existente
- **L:** todas as implementações de `HierarchicalClassifier` são intercambiáveis
- **I:** interfaces enxutas — nenhuma ABC tem método desnecessário
- **D:** casos de uso recebem interfaces, nunca implementações concretas

### Reprodutibilidade
- Seed `42` fixada em todo código que usa aleatoriedade
- Todos os parâmetros configuráveis via `config.yaml`, nunca hardcoded no código

### Testes
- Todo módulo de `infrastructure/` deve ter teste correspondente em `tests/unit/`
- Nomear: `test_<nome_do_modulo>.py`
- Usar `pytest`

---

## TECNOLOGIAS PERMITIDAS

| Biblioteca | Módulos que usa | Versão mínima |
|---|---|---|
| pandas | 2, 3 | 2.0.0 |
| numpy | 5 | 1.26.0 |
| scikit-learn | 4, 5 | 1.4.0 |
| matplotlib | 5, 6 | 3.8.0 |
| requests | 1 | 2.31.0 |
| pyyaml | shared | 6.0.1 |
| pytest | tests | 8.0.0 |

**⚠️ TensorFlow/Keras:** apenas na implementação final (4º bimestre). Não instalar agora.
**⚠️ Não adicionar bibliotecas fora desta lista sem aprovação explícita.**

---

## RISCOS — MONITORAR ATIVAMENTE

| Risco | Sinal de alerta | Ação |
|---|---|---|
| Integridade dos dados | Nulos inesperados, IDs duplicados | Executar `verify_conformity` antes de prosseguir |
| Complexidade > O(n²) | Loops aninhados sobre o dataset completo | Sinalizar e propor alternativa |
| Treinamento insuficiente | Métricas abaixo de baseline flat | Revisar hiperparâmetros e features |
| Dataset comprometido | API retorna dados inconsistentes com GO | Fora do escopo — comunicar ao aluno |

---

## CONFIG.YAML (referência)

```yaml
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  uniprot_limit: 500
  go_namespace: "molecular_function"

model:
  random_seed: 42
  test_size: 0.2
  validation_size: 0.1

logging:
  level: "INFO"
```

---

## REQUIREMENTS.TXT (referência)

```
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
requests>=2.31.0
pyyaml>=6.0.1
pytest>=8.0.0
```

---

## TAREFA DO DIA — 08/04/2026

**Objetivo:** criar toda a estrutura de pastas, arquivos `__init__.py`, entidades, interfaces, `config.yaml`, `requirements.txt` e `main.py` esqueleto.

### O que gerar agora:

1. **Estrutura de diretórios completa** com `__init__.py` em cada pasta de módulo Python
2. **`src/domain/entities/protein.py`** — dataclass `Protein`
3. **`src/domain/entities/hierarchy_graph.py`** — `FunctionNode` + `HierarchyGraph`
4. **`src/domain/interfaces/data_source.py`** — ABC `ProteinDataSource`
5. **`src/domain/interfaces/preprocessor.py`** — ABC `ProteinPreprocessor`
6. **`src/domain/interfaces/hierarchy_builder.py`** — ABC `HierarchyBuilder`
7. **`src/domain/interfaces/classifier.py`** — ABC `HierarchicalClassifier`
8. **`src/domain/interfaces/evaluator.py`** — ABC `HierarchicalEvaluator`
9. **`src/shared/config_loader.py`** — carrega `config.yaml` com PyYAML
10. **`src/shared/logger.py`** — logger padrão usando `logging` nativo do Python
11. **`config.yaml`** — parâmetros centralizados
12. **`requirements.txt`** — dependências listadas acima
13. **`main.py`** — esqueleto com `if __name__ == "__main__"` e importação dos casos de uso
14. **`.gitignore`** — excluir `data/raw/`, `data/processed/`, `__pycache__/`, `.env`, `*.pyc`

### Implementações concretas (stubs para hoje — implementar nos próximos dias):
- `src/infrastructure/data_sources/uniprot_client.py` — classe `UniProtClient(ProteinDataSource)` com métodos que levantam `NotImplementedError`
- `src/infrastructure/preprocessing/pandas_preprocessor.py` — stub
- `src/infrastructure/hierarchy/go_dag_builder.py` — stub
- `src/infrastructure/models/random_forest_classifier.py` — stub
- `src/infrastructure/models/svm_classifier.py` — stub
- `src/infrastructure/evaluation/hierarchical_metrics.py` — stub
- `src/infrastructure/prediction/inference_pipeline.py` — stub

### Commit ao final:
```
git init
git add .
git commit -m "feat: initial project structure — Clean Architecture, domain entities and interfaces"
```

---

## ESTADO DO CRONOGRAMA

| Data | Atividade | Status |
|---|---|---|
| 08/04 | Estrutura, entidades, interfaces | ✅ Concluído |
| 09/04 | Módulo 1 — Aquisição de dados | ✅ Concluído |
| 10/04 | Módulo 2 — Pré-processamento | ✅ Concluído |
| 11/04 | Módulo 3 — Hierarquia DAG | ✅ Concluído |
| 12/04 | Integração módulos 1–3 + testes | ⬜ Pendente |
| 13/04 | Buffer | ⬜ Pendente |
| 14/04 | Módulo 4 — Treinamento | ✅ Concluído |
| 15/04 | Módulo 5 — Avaliação | ⬜ Pendente |
| 16/04 | Módulo 6 — Previsão | ⬜ Pendente |
| 17/04 | Integração completa + teste ponta a ponta | ⬜ Pendente |
| 18–21/04 | Escrita do documento | ⬜ Pendente |
| **22/04** | **ENTREGA Projeto Físico Revisado** | ⬜ Pendente |

> Atualize os status neste arquivo ao final de cada sessão de trabalho.
