# Organização das Bases de Dados

- `raw/`: dados brutos coletados de fontes externas ou planilhas de referência originais. Não sobrescrever manualmente após a ingestão.
- `processed/`: dados limpos, padronizados e prontos para alimentar o solver de Programação Linear (ex.: carteira parametrizada com retorno líquido $\rho_i = j_i - \text{EL}_i$, tetos $U_i$, cotas $L_i$, setor e *rating*).
- `outputs/`: resultados gerados automaticamente pelas rotinas do modelo (alocações ótimas $x_i^*$, folgas, variáveis duais / preços-sombra $\lambda$ e análises de sensibilidade).
