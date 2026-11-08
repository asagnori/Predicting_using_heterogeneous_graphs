# Predicting Using Heterogeneous Graphs

Projeto acadêmico do TCC que investiga a previsão de atrasos em pedidos de uma
Bike Store usando dados relacionais, modelos tabulares e redes neurais de
grafos heterogêneos.

## Objetivo

O alvo da classificação é criado a partir das datas do pedido:

```text
atrasado = shipped_date > required_date
```

O grafo possui os seguintes tipos de nós:

- `customer`
- `order`
- `product`
- `brand`
- `category`
- `store`
- `staff`

E as relações:

```text
customer -> order
product  -> order
brand    -> product
category -> product
store    -> order
staff    -> order
```

O experimento compara um baseline XGBoost com GraphSAGE heterogêneo,
validação temporal e explicabilidade por SHAP e GNNExplainer.

O pipeline modular também inclui um `DummyClassifier` estratificado como
baseline de acaso. A função `run_temporal_experiment()` executa quatro folds
temporais por padrão e retorna, para cada fold e modelo, acurácia, precisão,
recall, F1 e PR-AUC. A tabela pode ser materializada com
`temporal_results_table()`. O SHAP do XGBoost modular usa exclusivamente as
features de negócio (`store_id`, `staff_id`, `order_month` e
`required_month`), sem IDs brutos de alta cardinalidade.

## Resultados visuais

As figuras abaixo documentam a evolução do experimento, desde a análise
preliminar até os resultados finais do modelo.

### Análise preliminar

#### Distribuição do alvo

![Distribuição da variável alvo](<docs/TCC/img/preliminary results/2- Distribution_Target_Chart.png>)

#### Matriz de confusão preliminar

![Matriz de confusão preliminar](<docs/TCC/img/preliminary results/3- Confusion_Matrix_Chart.png>)

#### Sazonalidade dos pedidos

![Análise de sazonalidade](<docs/TCC/img/preliminary results/4- Seasonality_Chart.png>)

#### Desbalanceamento das classes

![Desbalanceamento das classes](<docs/TCC/img/preliminary results/5- Class Imbalance_Chart.png>)

#### Explicabilidade preliminar

![Explicabilidade preliminar com SHAP e GNNExplainer](<docs/TCC/img/preliminary results/6- Explicability_Chart (SHAP & GNNExplainer).png>)

### Resultados finais

#### Esquema do grafo heterogêneo

![Esquema do grafo heterogêneo](<docs/TCC/img/final results/1- Hetero_Graph_Schema_Chart.png>)

#### Explicabilidade SHAP do XGBoost

![Explicabilidade SHAP do XGBoost](<docs/TCC/img/final results/2- Explicability SHAP XGBoost.png>)

#### Importância global das features

![Importância global das features](<docs/TCC/img/final results/3- Explicability SHAP XGBoost - Global Importance Features.png>)

#### Importância real das features

![Importância real das features](<docs/TCC/img/final results/4- Explicability SHAP XGBoost - Real Importance Features.png>)

#### Curva Precision-Recall

![Curva Precision-Recall e PR-AUC](<docs/TCC/img/final results/5- Precision &  Recall Curve - PR-AUC.png>)

#### Matrizes de confusão da validação temporal

![Matrizes de confusão da validação temporal](<docs/TCC/img/final results/6- Temporal Comparative Confusion Matriz - K4 Folds.png>)

#### Importância global das relações

![Importância global das relações pelo GNNExplainer](<docs/TCC/img/final results/7- GNNExplainer - Global Importance & elationship.png>)

#### Comparação por classe real

![Comparação das explicações por classe real](<docs/TCC/img/final results/8- GNNExplainer Real Classe Comparision.png>)

## Organização do repositório

```text
data/                       Dados CSV e dados externos
docs/TCC/python/            Scripts monolíticos experimentais da tese
docs/TCC/img/              Figuras preliminares e resultados finais
docs/                       Documentos e materiais acadêmicos
images/                     Gráficos gerados anteriormente
notebooks/                  Análises exploratórias
scripts/                    SQL de criação e carga do banco
src/
  config/                   Configuração e variáveis de ambiente
  data/                     Leitura de tabelas e features
  graph/                    Mapeamentos e construção do HeteroData
  models/                   GraphSAGE e baseline XGBoost
  training/                 Treinamento e validação temporal
  explainability/           Integração com GNNExplainer
  visualization/            Gráficos e tabelas de resultados
```

## Scripts experimentais

Os arquivos abaixo são referências acadêmicas e permanecem monolíticos
intencionalmente:

- `docs/TCC/python/Predicting_heterogeneous_graphs.py`
- `docs/TCC/python/Predicting_heterogeneous_graphs_preliminary.py`
- `docs/TCC/python/Predicting_heterogeneous_graphs_final.py`
- `docs/TCC/python/Predicting_heterogeneous_graphs_final_w_GNNExplainer.py`

Eles preservam os experimentos utilizados na evolução da tese. A implementação
modular em `src/` aplica a lógica do script final sem substituir essas
referências.

## Configuração do banco

As credenciais não ficam no código. Crie ou mantenha um arquivo `.env` na raiz
do projeto. O código modular aceita as variáveis recomendadas:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_NAME=bike_store
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
```

Também é suportado temporariamente o bloco legado `DatabaseConfig` já usado no
arquivo `.env` deste projeto. Recomenda-se migrar para as variáveis `DB_*`.

Parâmetros opcionais:

```dotenv
GNN_EPOCHS=100
GNN_LEARNING_RATE=0.01
GNN_HIDDEN_CHANNELS=64
GNN_OUT_CHANNELS=2
RANDOM_STATE=42
TEMPORAL_CUTOFF=2018-01-01
```

Antes do treinamento, o pipeline valida os dispositivos na ordem CUDA, MPS e
CPU. O primeiro backend capaz de alocar e operar um tensor é selecionado
automaticamente.

O arquivo `.env` é ignorado pelo Git e não deve ser compartilhado.

Se a execução retornar `MySQL error 1045 (Access denied)`, o servidor foi
alcançado, mas recusou a combinação de usuário e senha. Nesse caso, confirme
as credenciais no MySQL e que o usuário possui permissão para o banco
`bike_store`. O formato legado usado anteriormente podia armazenar caracteres
especiais da senha codificados para URL; essa compatibilidade é convertida
automaticamente pelo carregamento modular.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução modular

O entrypoint modular carrega as tabelas, prepara o alvo, constrói o grafo e
exibe um resumo:

```bash
python -m src.main
```

As funções de treinamento e explicabilidade podem ser importadas sem executar
conexão ou treinamento automaticamente:

```python
from src.training.cross_validation import run_temporal_experiment
from src.explainability.gnn import create_order_explainer
```

## Validações

O repositório contém os dados e scripts de análise, mas não possui uma suíte de
testes automatizados. A validação mínima da implementação modular pode ser
executada com:

```bash
python -m compileall -q src
git diff --check
```

## Tecnologias

Python, Pandas, NumPy, PyTorch, PyTorch Geometric, XGBoost, SHAP, Scikit-Learn,
SQLAlchemy, NetworkX, Matplotlib e Seaborn.

## Autor

Angëlo Sagnori
MBA Data Science & Analytics – USP/ESALQ
