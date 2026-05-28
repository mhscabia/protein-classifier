# Protein Classifier

Pipeline de aprendizado de máquina para classificação hierárquica de funções biológicas de proteínas, respeitando a estrutura DAG da Gene Ontology (GO).

## Como funciona

O sistema recebe uma sequência de aminoácidos e prediz os termos GO (Gene Ontology) associados à sua função molecular. A classificação é feita por um **LCN (Local Classifier per Node)** — um modelo Random Forest binário por nó do DAG — que garante consistência hierárquica propagando predições de ancestrais para descendentes.

As features são **embeddings ESM-2 (320 dims)** gerados pelo modelo de linguagem de proteínas `facebook/esm2_t6_8M_UR50D`. O embedding não prediz função — é um extrator de representação contextual; o LCN é o classificador hierárquico.

## Requisitos

- Python 3.9+
- Dependências:

```bash
pip install -r requirements.txt
```

## Como usar

### Uso rápido — modelo pré-treinado

```bash
python main.py
```

Se não houver modelo salvo localmente, o programa pergunta se você deseja baixar o modelo pré-treinado (~2 GB) do Hugging Face Hub. Responda `s` para confirmar.

O modelo fica em cache em `data/models/` — nas próximas execuções é carregado direto do disco.

### Treinar do zero

```bash
# Usa o número de proteínas definido em config.yaml (padrão: 15000)
python main.py --train

# Treina com N proteínas customizado
python main.py --train 5000
python main.py --train 10000
```

O treinamento busca proteínas da Swiss-Prot via UniProt REST API e termos GO via QuickGO. Dependendo do número de amostras e da conexão, pode levar vários minutos.

### Ajuda

```bash
python main.py --help
```

## Configuração

Todos os parâmetros ficam em [`config.yaml`](config.yaml). Os mais relevantes:

| Chave | Padrão | Descrição |
|-------|--------|-----------|
| `data.uniprot_limit` | `15000` | Proteínas buscadas no treinamento |
| `hierarchy.min_term_support` | `20` | Remove termos GO com menos de N proteínas anotadas |
| `model.hf_repo` | `mathscabia/protein-classifier` | Repositório HF com modelos pré-treinados |
| `pipeline.mode` | `auto` | `"interactive"` pausa entre módulos |
| `predict.sequence_source` | `"input"` | `"index"` usa linha do CSV em vez do terminal |

## Arquitetura

Clean Architecture estrita — dependências sempre apontam para dentro:

```
infrastructure → application → domain
```

```
protein_classifier/
├── main.py                        ← Ponto de entrada (CLI)
├── config.yaml                    ← Parâmetros centralizados
└── src/
    ├── domain/
    │   ├── entities/              ← Protein, FunctionNode, HierarchyGraph
    │   └── interfaces/            ← ABCs: HierarchicalClassifier, etc.
    ├── application/
    │   └── use_cases/             ← Orquestração dos módulos
    └── infrastructure/
        ├── data_sources/          ← M1: UniProtClient, GOClient
        ├── preprocessing/         ← M2: PandasPreprocessor, ESMEmbedder
        ├── hierarchy/             ← M3: GODagBuilder
        ├── models/                ← M4: LCNClassifier
        ├── evaluation/            ← M5: HierarchicalMetricsEvaluator
        ├── prediction/            ← M6: InferencePipeline
        ├── persistence/           ← ModelPersistence, HFDownloader
        └── visualization/         ← ResultVisualizer
```

### Módulos

| Módulo | Responsabilidade | Status |
|--------|-----------------|--------|
| M1 — Aquisição | Busca proteínas (UniProt) e termos GO (QuickGO) | ✅ |
| M2 — Pré-processamento | Limpeza, normalização, embeddings ESM-2 | ✅ |
| M3 — Hierarquia DAG | Monta grafo GO com filtro por suporte mínimo | ✅ |
| M4 — Treinamento LCN | RF binário por nó, propagação top-down | ✅ |
| M5 — Avaliação | hP, hR, hF com expansão hierárquica de ancestrais | ✅ |
| M6 — Inferência | Predição para sequência avulsa + visualizações | ✅ |

## Métricas (ESM-2, 320 features, 15000 proteínas, seed=42)

| Configuração | hP | hR | hF | Nós DAG |
|---|---|---|---|---|
| ESM-2 + `min_term_support=20` | 0.9684 | 0.6622 | 0.7866 | 817 |

> Avaliado em 100 amostras aleatórias do test set (~2396 proteínas). hF hierárquico é **6.26×** maior que flat_F (0.1257), refletindo o crédito parcial dado pela propagação de ancestrais.

## Modelo no Hugging Face Hub

Os modelos pré-treinados estão disponíveis em:
[huggingface.co/mathscabia/protein-classifier](https://huggingface.co/mathscabia/protein-classifier)

Arquivos:
- `model.joblib` — LCNClassifier + StandardScaler (~2 GB)
- `hierarchy.joblib` — HierarchyGraph com termos GO (~0.7 MB)
