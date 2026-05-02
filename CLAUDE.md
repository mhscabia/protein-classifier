# CLAUDE.md — Protein Classifier
## Previsão de Função de Proteínas com Classificação Hierárquica

> Este arquivo é lido automaticamente pelo Claude Code a cada sessão.
> **Nunca remova ou edite este arquivo sem instrução explícita.**

---

## OBJETIVO DO SISTEMA

Pipeline de aprendizado de máquina que recebe dados de proteínas e classifica
suas funções biológicas usando classificação hierárquica, respeitando a estrutura
DAG da Gene Ontology (GO).

---

## ARQUITETURA — CLEAN ARCHITECTURE ESTRITA

```
protein_classifier/
│
├── main.py                      ← Ponto de entrada
├── config.yaml                  ← Parâmetros centralizados
│
├── src/
│   ├── domain/
│   │   ├── entities/            ← Protein, FunctionNode, HierarchyGraph
│   │   └── interfaces/          ← ABCs: HierarchicalClassifier, ProteinDataSource, etc.
│   │
│   ├── application/
│   │   └── use_cases/           ← Orquestração dos módulos
│   │
│   ├── infrastructure/
│   │   ├── data_sources/        ← M1: UniProtClient, GOClient
│   │   ├── preprocessing/       ← M2: PandasPreprocessor
│   │   ├── hierarchy/           ← M3: GODagBuilder
│   │   ├── models/              ← M4: RF, SVM, LCN (a implementar)
│   │   ├── evaluation/          ← M5: HierarchicalMetricsEvaluator
│   │   ├── prediction/          ← M6: InferencePipeline
│   │   ├── persistence/         ← NOVO: ModelPersistence (a implementar)
│   │   └── visualization/       ← ResultVisualizer
│   │
│   └── shared/                  ← logger, config_loader, presenter
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── data/
    ├── raw/                     ← proteins.csv, go_terms.json
    ├── processed/               ← proteins_clean.csv
    ├── models/                  ← NOVO: modelos persistidos (gitignored)
    └── output/                  ← Gráficos gerados (gitignored)
```

### Regra de dependência (NUNCA violar)
```
infrastructure → application → domain
```

---

## O QUE ESTÁ IMPLEMENTADO

### M1 — Aquisição de dados ✅
- `uniprot_client.py` — busca proteínas da Swiss-Prot via API REST com paginação
- `go_client.py` — coleta termos GO e ancestrais via QuickGO em lotes de 25
- Saída: `data/raw/proteins.csv` e `data/raw/go_terms.json`
- `go_terms.json` só é rebuscado se não existir em disco

### M2 — Pré-processamento ✅
- `pandas_preprocessor.py`
- `clean()`: remove nulos, vazios, duplicatas, sequências com caracteres inválidos
- `normalize()`: extrai 22 features numéricas + aplica `StandardScaler`
- Features: `seq_length`, `molecular_weight`, `aa_A`...`aa_Y` (20 aminoácidos)
- O scaler deve ser salvo junto com o modelo — essencial para inferência correta

### M3 — Hierarquia DAG ✅
- `go_dag_builder.py`
- Lê `go_terms.json`, monta `HierarchyGraph` com `FunctionNode`
- `get_ancestors(term_id)` é o método central — usado em M4, M5 e M6
- Filtra apenas termos relevantes às proteínas do dataset + seus ancestrais

### M4 — Treinamento ✅ (RF + SVM)
- `random_forest_classifier.py` — 100 árvores, multi-label com `MultiLabelBinarizer`
- `svm_classifier.py` — `OneVsRestClassifier` com `SVC(kernel="rbf")`
- Ambos propagam ancestrais nos labels antes de treinar
- Ambos propagam ancestrais nas predições para garantir consistência hierárquica

### M5 — Avaliação ✅
- `hierarchical_metrics.py`
- `evaluate()`: hP, hR, hF com expansão de ancestrais antes de comparar
- `evaluate_flat()`: mesmas métricas sem expansão — baseline de comparação
- hP = Σ|pred∩true| / Σ|pred| ; hR = Σ|pred∩true| / Σ|true| ; hF = média harmônica

### M6 — Inferência e Visualização ✅
- `inference_pipeline.py` — extrai 22 features de sequência nova, aplica scaler, classifica
- `result_visualizer.py` — gera `dag_predictions.png` e `metrics_comparison.png`

### Resultados do E2E (500 proteínas, seed=42)

| Classificador | hP | hR | hF | flat_F |
|---|---|---|---|---|
| SVM ⭐ | 0.8785 | 0.2119 | **0.3415** | 0.0315 |
| Random Forest | 0.8884 | 0.2021 | 0.3294 | 0.0326 |
| LCN | 0.8646 | 0.1934 | 0.3160 | 0.0330 |

Ganho hierárquico vs flat: ~10x.

---

## O QUE FALTA IMPLEMENTAR

### 1 — LCN (Local Classifier per Node) ⬜

Classificador hierárquico puro. Principal evolução em relação ao protótipo.

**Problema que resolve:** RF e SVM são classificadores flat — a hierarquia aparece
só no pré-processamento e nas métricas, não no algoritmo. O LCN incorpora a
hierarquia diretamente no modelo.

**Conceito:** um classificador binário por nó do DAG. Cada um responde
"essa proteína pertence a este nó ou não?". Predição top-down — começa na raiz,
desce pela hierarquia, para quando o classificador do nó retorna negativo.

**Por que melhora o recall:** divide o problema de 615 classes em subproblemas
menores e mais balanceados. Cada nó treina apenas com proteínas relevantes
para aquele ramo.

**Arquivo:** `src/infrastructure/models/lcn_classifier.py`

**Interface:** `HierarchicalClassifier` — métodos `train()` e `predict()`

**Lógica do `train()`:**
```
Para cada nó do DAG:
    positivos = proteínas que têm esse termo GO
    negativos = proteínas que não têm esse termo GO
    treinar RandomForestClassifier binário para esse nó
    armazenar classificador indexado por term_id
```

**Lógica do `predict()`:**
```
Para cada proteína:
    fila = [raiz do DAG]
    preditos = set()
    Para cada nó na fila:
        se classificador[nó].predict(X) == positivo:
            preditos.add(nó.term_id)
            fila.extend(nó.children_ids)
    retornar preditos
```

**Classificador binário por nó:**
```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

**Integração no `main.py`:**
```python
classifiers = {
    "RandomForest": RandomForestHierarchicalClassifier(config),
    "SVM": SVMHierarchicalClassifier(config),
    "LCN": LCNClassifier(config),
}
```

**Teste:** `tests/unit/test_lcn_classifier.py`

---

### 2 — Persistência do modelo em disco ⬜

**Problema que resolve:** atualmente o modelo retreina a cada execução.
Com datasets maiores isso se torna inviável.

**Arquivo:** `src/infrastructure/persistence/model_persistence.py`

**O que salvar:** classificador treinado + `StandardScaler` do M2 + metadados
(data de treino, `uniprot_limit`, métricas obtidas)

**Biblioteca:** `joblib` — já disponível via scikit-learn

**Interface:**
```python
def save_model(classifier, scaler, metadata: dict, path: str) -> None: ...
def load_model(path: str) -> tuple: ...
def model_exists(path: str) -> bool: ...
```

**Parâmetro no `config.yaml`:**
```yaml
model:
  persist_path: "data/models/"
```

**Lógica no `main.py`:**
```
Se model_exists(persist_path):
    classifier, scaler = load_model(persist_path)
    pular M4
Senão:
    executar M4 normalmente
    save_model(classifier, scaler, metadata, persist_path)
```

**`.gitignore`:** adicionar `data/models/`

**Teste:** `tests/unit/test_model_persistence.py`

---

### 3 — Aumentar volume de dados ⬜

Após persistência implementada, aumentar `uniprot_limit` progressivamente.

- Atual: 500 proteínas
- Meta: 2000–5000 (testar escalabilidade antes de ir além)
- Teto da Swiss-Prot: ~570 mil proteínas revisadas

---

## CONVENÇÕES — OBRIGATÓRIAS

- `snake_case` funções/variáveis, `PascalCase` classes
- Responsabilidade única por função — questionar se passar de 20 linhas
- Sem comentários óbvios — o código se explica pelos nomes
- DRY — nunca duplicar lógica
- Seed `42` em todo código que usa aleatoriedade
- Todos os parâmetros no `config.yaml`, nunca hardcoded no código
- Todo módulo novo em `infrastructure/` deve ter teste em `tests/unit/`
- Dependências sempre apontam para dentro: `infrastructure → application → domain`

---

## TECNOLOGIAS PERMITIDAS

| Biblioteca | Uso |
|---|---|
| pandas | M2, M3 |
| numpy | M5 |
| scikit-learn | M4, M5 |
| matplotlib | M5, M6 |
| requests | M1 |
| pyyaml | shared |
| pytest | tests |
| networkx | M6 visualização |
| joblib | persistence (NOVO) — via scikit-learn |

**⚠️ Não adicionar bibliotecas fora desta lista sem aprovação explícita.**

---

## RISCOS

| Risco | Sinal | Ação |
|---|---|---|
| LCN lento | Treino > 10 min para 500 proteínas | Limitar profundidade do DAG ou reduzir estimadores por nó |
| Modelo desatualizado | Dataset mudou, modelo não retreinou | Salvar hash do `proteins.csv` com o modelo |
| Recall do LCN não melhora | hR < 0.21 | Revisar balanceamento positivo/negativo por nó |
| Complexidade > O(n²) | Loops aninhados sobre dataset | Sinalizar e propor alternativa |

---

## STATUS

| Módulo | Status |
|---|---|
| M1 — Aquisição | ✅ |
| M2 — Pré-processamento | ✅ |
| M3 — Hierarquia DAG | ✅ |
| M4 — Treinamento LCN | ✅ |
| M5 — Avaliação hierárquica | ✅ |
| M6 — Inferência e visualização | ✅ |
| Persistência do modelo | ✅ |
| Aumentar volume de dados | ⬜ |

> Branch `archive/all-classifiers` preserva a versão com RF + SVM + LCN para referência histórica.

---

## CHECKLIST DE IMPLEMENTAÇÃO

> Atualizar checkboxes conforme tarefas são concluídas.
> Ao iniciar uma sessão, ler esta seção para saber por onde continuar.
> Formato: `- [x]` = concluído, `- [ ]` = pendente, `- [~]` = em andamento.

### Bloco A — LCN (Local Classifier per Node)
- [x] TASK-01 — Criar `src/infrastructure/models/lcn_classifier.py` com classe `LCNClassifier` e `__init__`
- [x] TASK-02 — Implementar `train()` no LCNClassifier (loop por nó do DAG, RF binário por nó)
- [x] TASK-03 — Implementar `predict()` no LCNClassifier (travessia top-down, fila BFS)
- [x] TASK-04 — Implementar `_augment_with_ancestors()` (reutilizar padrão do RF/SVM)
- [x] TASK-05 — Criar `tests/unit/test_lcn_classifier.py` com fixtures e 3 testes
- [x] TASK-06 — Adicionar `LCNClassifier` ao dict `classifiers` em `main.py`
- [x] TASK-07 — Executar `pytest tests/unit/test_lcn_classifier.py` e corrigir falhas
- [x] TASK-08 — Executar pipeline completo e registrar métricas do LCN na tabela acima

### Bloco B — Persistência de modelo
- [x] TASK-09 — Criar diretório `src/infrastructure/persistence/` + `__init__.py`
- [x] TASK-10 — Criar `model_persistence.py` com `save_model()`, `load_model()`, `model_exists()`
- [x] TASK-11 — Adicionar `persist_path: "data/models/"` em `config.yaml` (dentro de `model:`)
- [x] TASK-12 — Adicionar `data/models/` ao `.gitignore`
- [x] TASK-13 — Criar `tests/unit/test_model_persistence.py` com 4 testes
- [x] TASK-14 — Integrar persistência em `main.py`: checar, carregar ou salvar após treino
- [x] TASK-15 — Executar `pytest tests/unit/test_model_persistence.py` e corrigir falhas
- [ ] TASK-16 — Executar pipeline completo validando que segunda execução pula treino

### Bloco C — Escalar dados
- [x] TASK-17 — Alterar `uniprot_limit` para `2000` em `config.yaml`
- [x] TASK-18 — Deletar `data/models/` e `data/raw/` para forçar re-fetch
- [x] TASK-19 — Executar pipeline completo com 2000 proteínas e registrar métricas
- [x] TASK-20 — Atualizar tabela de resultados no CLAUDE.md com novos números

### Bloco D — Refatoração LCN-only
- [x] TASK-21 — Criar branch `archive/all-classifiers` com RF + SVM + LCN preservados
- [x] TASK-22 — Remover RF e SVM do `main.py` (pipeline LCN-only)
- [x] TASK-23 — Atualizar CLAUDE.md (STATUS + nota sobre archive branch)
- [ ] TASK-24 — Executar `pytest tests/unit/` e confirmar sem regressões
- [ ] TASK-25 — Executar `python main.py` e confirmar pipeline LCN-only funciona