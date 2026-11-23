# Otimização de Carteira de Crédito Corporativo via Programação Linear

Repositório dedicado à implementação computacional, bases de dados e experimentos do projeto **Otimização da Alocação de Capital em Carteira de Crédito Corporativo via Programação Linear**.

## Estrutura do Projeto

```text
.
├── data/
│   ├── raw/            # Bases originais coletadas (imutáveis)
│   ├── processed/      # Bases tratadas e parametrizadas para o modelo LP
│   └── outputs/        # Resultados exportados (alocações, folgas, preços-sombra, sensibilidade)
├── notebooks/          # Notebooks exploratórios e análises interativas
├── reports/
│   └── figures/        # Gráficos gerados para o relatório e apresentação
├── scripts/            # Scripts executáveis para rodar cenários e gerar saídas
├── src/
│   ├── data/           # Leitura, validação e pré-processamento das bases
│   ├── model/          # Formulação LP, cenários (irrestrito/diversificado), dualidade e sensibilidade
│   └── visualization/  # Geração de tabelas e gráficos analíticos
├── tests/              # Testes unitários e validação matemática do modelo
├── pyproject.toml      # Metadados e dependências do projeto
└── requirements.txt    # Dependências principais para instalação via pip
```

## Configuração do Ambiente Python

Recomendado utilizar ambiente virtual isolado (`.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para incluir ferramentas de desenvolvimento e notebooks (`pytest`, `ruff`, `jupyter`):

```bash
pip install -e ".[dev]"
```

## Convenções de Organização

- **Bases brutas (`data/raw/`):** armazenar arquivos originais de referência (taxas, spreads, PDs, LGDs, limites setoriais) sem edição manual direta.
- **Bases processadas (`data/processed/`):** armazenar datasets estruturados com os parâmetros canônicos do modelo ($j_i$, $\text{PD}_i$, $\text{LGD}_i$, $\text{EL}_i$, $U_i$, $L_i$, setor e *rating*).
- **Resultados (`data/outputs/`):** armazenar CSVs/JSONs gerados pelos scripts de execução (cenário irrestrito vs. diversificado, folgas, preços-sombra e curva de sensibilidade).
- **Código-fonte (`src/`):** manter a lógica de dados, modelagem LP e visualização separada em módulos reutilizáveis e testáveis.
