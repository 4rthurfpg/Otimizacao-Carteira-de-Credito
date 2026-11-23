# Otimização de Carteira de Crédito Corporativo via Programação Linear

Implementação em Python do modelo de Programação Linear (PL) para alocação ótima de capital em uma carteira de 20 operações corporativas brasileiras, utilizando o solver **HiGHS** (`scipy.optimize.linprog`).

## Estrutura do Repositório

```text
modelagem/
├── dados/
│   ├── operacoes.csv        # 20 operações corporativas calibradas (spread, PD, LGD, setor, rating)
│   └── parametros.yaml      # Parâmetros da política de crédito (B, L_c, tetos setoriais, cotas, rating, EL_max)
├── resultados/              # Tabelas (.csv) e gráficos (.png, 300 dpi) gerados automaticamente
├── modelo.py                # Leitura, validação, montagem explícita de A_ub/b_ub e resolução (HiGHS)
├── experimentos.py          # Execução dos cenários, pós-otimalidade, sensibilidade, inviabilidade e gráficos
├── requirements.txt         # Dependências Python (pandas, numpy, scipy, matplotlib, pyyaml)
├── .gitignore
└── README.md
```

## Execução

Com o ambiente virtual ativo (ou apontando diretamente para `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experimentos.py
```

O comando `python experimentos.py` executa todas as análises e grava automaticamente os arquivos `.csv` e `.png` em `resultados/`.

## Visão Geral do Modelo

- **Decisão:** $x_i \ge 0$ (capital em R\$ milhões alocado à operação $i$).
- **Objetivo:** maximizar $\sum_i \pi_i x_i$, onde $\pi_i = \text{spread}_i - \text{PD}_i \cdot \text{LGD}_i$ (implementado como $\min \sum_i (-\pi_i) x_i$).
- **Cenários comparados:**
  - **Irrestrito:** orçamento ($B = \text{R\$ } 500\text{ Mi}$) e teto por cliente ($L_c = \text{R\$ } 75\text{ Mi}$).
  - **Diversificado:** adiciona tetos setoriais ($\alpha_s = 30\%$), teto agregado para `AA(bra)` e inferiores ($\le 20\%$ de $B$), cotas mínimas estratégicas (`Energia Elétrica >= 15%`, `Agronegócio e Alimentos >= 12%`) e teto de perda esperada agregada ($EL_{\max} = \text{R\$ } 1{,}75\text{ Mi}$).
