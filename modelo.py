"""
Módulo principal de modelagem e resolução do problema de Programação Linear (PL)
para otimização da alocação de capital em carteira de crédito corporativo.

Implementação direta e autocontida utilizando pandas, PyYAML, NumPy e SciPy (solver HiGHS).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog


# Caminhos padrão relativos à raiz do repositório ./modelagem
RAIZ_REPO = Path(__file__).resolve().parent
CAMINHO_OPERACOES_PADRAO = RAIZ_REPO / "dados" / "operacoes.csv"
CAMINHO_PARAMETROS_PADRAO = RAIZ_REPO / "dados" / "parametros.yaml"

# Tolerância numérica padrão para identificar restrições ativas (folga ~ 0)
TOL_ATIVA = 1e-7


def carregar_dados(
    caminho_operacoes: Path | str = CAMINHO_OPERACOES_PADRAO,
    caminho_parametros: Path | str = CAMINHO_PARAMETROS_PADRAO,
) -> tuple[pd.DataFrame, dict]:
    """
    Lê e valida a base de operações (CSV) e os parâmetros da política de crédito (YAML).

    Também calcula as grandezas unitárias por operação:
    - EL_i = PD_i * LGD_i  (perda esperada por R$ 1 alocado)
    - pi_i = spread_i - EL_i  (retorno líquido esperado por R$ 1 alocado)
    """
    df = pd.read_csv(caminho_operacoes)
    with open(caminho_parametros, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Validação mínima de integridade da tabela de operações
    colunas_esperadas = {"id", "cliente", "setor", "rating", "spread", "pd", "lgd"}
    faltantes = colunas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes em operacoes.csv: {sorted(faltantes)}")

    if df[["id", "cliente", "setor", "rating", "spread", "pd", "lgd"]].isna().any().any():
        raise ValueError("Existem valores nulos em campos obrigatórios de operacoes.csv.")

    if (df["pd"] < 0).any() or (df["pd"] > 1).any():
        raise ValueError("Probabilidade de inadimplência (pd) fora do intervalo [0, 1].")
    if (df["lgd"] < 0).any() or (df["lgd"] > 1).any():
        raise ValueError("Loss Given Default (lgd) fora do intervalo [0, 1].")
    if (df["spread"] < 0).any():
        raise ValueError("Existem spreads negativos em operacoes.csv.")
    if float(params["B"]) <= 0:
        raise ValueError("O orçamento total B deve ser estritamente positivo.")

    # Verificação de consistência entre setores da base e parâmetros do YAML
    setores_csv = set(df["setor"].unique())
    setores_yaml = set(params["tetos_setor"].keys())
    if not setores_csv.issubset(setores_yaml):
        raise ValueError(f"Setores no CSV sem teto definido no YAML: {setores_csv - setores_yaml}")

    # Cálculo financeiro linear por operação (unidades em fração decimal):
    # 1) Perda Esperada unitária: EL_i = PD_i * LGD_i (como EAD_i = x_i, EL total = sum EL_i * x_i)
    # 2) Retorno líquido unitário: pi_i = spread_i - EL_i (coeficiente da função objetivo)
    df = df.copy()
    df["el"] = df["pd"] * df["lgd"]
    df["pi"] = df["spread"] - df["el"]

    return df, params


def montar_matrizes_lp(
    df: pd.DataFrame,
    params: dict,
    cenario: str = "diversificado",
    modo_rating: str = "grupo_maior_risco",
    el_max_override: float | None = None,
    tetos_setor_override: dict[str, float] | None = None,
) -> dict:
    """
    Constrói explícita e integralmente os vetores e matrizes do modelo LP:
        min  c^T x
        s.a. A_ub @ x <= b_ub
             x >= 0

    Parâmetros:
    - cenario: "irrestrito" (apenas orçamento B e teto por cliente L_c) ou
               "diversificado" (todas as 5 famílias de restrições).
    - modo_rating: "grupo_maior_risco" (padrão validado no glpsol: AA(bra) e inferiores <= 20% de B)
                   ou "por_rating_individual" (tetos beta_r individuais do YAML).
    - el_max_override: valor opcional em R$ milhões para substituir o teto EL_max do YAML.
    - tetos_setor_override: dicionário opcional {setor: limite_em_brl_milhoes} para sensibilidade.
    """
    if cenario not in {"irrestrito", "diversificado"}:
        raise ValueError("Parâmetro 'cenario' deve ser 'irrestrito' ou 'diversificado'.")
    if modo_rating not in {"grupo_maior_risco", "por_rating_individual"}:
        raise ValueError("Parâmetro 'modo_rating' inválido.")

    n = len(df)
    B = float(params["B"])
    L_c = float(params["limites_cliente"]["brl_milhoes"])
    el_max = float(params["el_max_brl_milhoes"]) if el_max_override is None else float(el_max_override)

    # Inversão de sinal da função objetivo:
    # A formulação matemática maximiza Z = sum(pi_i * x_i), mas scipy.optimize.linprog
    # resolve exclusivamente problemas de minimização. Como max(pi^T x) = -min(-pi^T x),
    # definimos c = -pi e, ao final, recuperamos Z* = -res.fun.
    c = -df["pi"].to_numpy(dtype=float)

    linhas_A = []
    valores_b = []
    metadados = []

    def adicionar_restricao(nome: str, familia: str, sinal_orig: str, coefs: np.ndarray, rhs: float, descricao: str):
        """Adiciona uma linha à matriz A_ub mantendo metadados para análise pós-ótima."""
        linhas_A.append(coefs.astype(float))
        valores_b.append(float(rhs))
        metadados.append(
            {
                "nome": nome,
                "familia": familia,
                "sinal_original": sinal_orig,
                "limite_original": float(rhs if sinal_orig == "<=" else -rhs),
                "descricao": descricao,
            }
        )

    # 1) Restrição de orçamento total: sum(x_i) <= B
    adicionar_restricao(
        nome="orcamento_total",
        familia="Orcamento",
        sinal_orig="<=",
        coefs=np.ones(n, dtype=float),
        rhs=B,
        descricao=f"Orcamento total da carteira <= R$ {B:.2f} Mi",
    )

    # 2) Tetos de exposição por cliente: sum_{i in I_c} x_i <= L_c
    # Modelamos explicitamente todas as restrições por cliente dentro de A_ub e b_ub
    # (deixando em `bounds` apenas a não negatividade x_i >= 0).
    # Isso permite extrair todos os preços-sombra (inclusive lambda_c^cli) de forma
    # uniforme em `res.ineqlin.marginals` e as folgas em `res.ineqlin.slack`.
    for cliente, grupo in df.groupby("cliente", sort=False):
        linha = np.zeros(n, dtype=float)
        linha[grupo.index.to_numpy()] = 1.0
        ids_ops = ",".join(grupo["id"].tolist())
        adicionar_restricao(
            nome=f"teto_cliente[{ids_ops}]",
            familia="Teto Cliente",
            sinal_orig="<=",
            coefs=linha,
            rhs=L_c,
            descricao=f"Exposicao maxima ao cliente {cliente} <= R$ {L_c:.2f} Mi",
        )

    # No cenário diversificado, adicionamos as demais políticas de crédito
    if cenario == "diversificado":
        # 3) Tetos máximos por setor econômico: sum_{i in I_s} x_i <= alpha_s * B
        for setor, cfg in params["tetos_setor"].items():
            teto_s = (
                float(tetos_setor_override[setor])
                if tetos_setor_override is not None and setor in tetos_setor_override
                else float(cfg["brl_milhoes"])
            )
            linha = (df["setor"] == setor).to_numpy(dtype=float)
            adicionar_restricao(
                nome=f"teto_setor[{setor}]",
                familia="Teto Setorial",
                sinal_orig="<=",
                coefs=linha,
                rhs=teto_s,
                descricao=f"Teto setorial {setor} <= R$ {teto_s:.2f} Mi",
            )

        # 4) Exposição máxima por faixa de rating: sum_{i in I_r} x_i <= beta_r * B
        if modo_rating == "grupo_maior_risco":
            cfg_risco = params["tetos_rating"]["grupo_maior_risco"]
            ratings_risco = set(cfg_risco["ratings_incluidos"])
            teto_risco = float(cfg_risco["brl_milhoes"])
            linha = df["rating"].isin(ratings_risco).to_numpy(dtype=float)
            adicionar_restricao(
                nome="teto_rating[AA(bra)_e_inferiores]",
                familia="Teto Rating",
                sinal_orig="<=",
                coefs=linha,
                rhs=teto_risco,
                descricao=f"Exposicao em AA(bra) e inferiores <= R$ {teto_risco:.2f} Mi",
            )
        else:
            for rating, cfg in params["tetos_rating"]["por_rating_individual"].items():
                teto_r = float(cfg["brl_milhoes"])
                linha = (df["rating"] == rating).to_numpy(dtype=float)
                adicionar_restricao(
                    nome=f"teto_rating[{rating}]",
                    familia="Teto Rating",
                    sinal_orig="<=",
                    coefs=linha,
                    rhs=teto_r,
                    descricao=f"Exposicao maxima no rating {rating} <= R$ {teto_r:.2f} Mi",
                )

        # 5) Cotas mínimas estratégicas por setor: sum_{i in I_s} x_i >= gamma_s * B
        # Como `linprog` aceita apenas desigualdades do tipo A_ub @ x <= b_ub, convertemos
        # cada restrição (>=) multiplicando ambos os lados por -1:
        #     -sum_{i in I_s} x_i <= -gamma_s * B
        for setor, cfg in params["cotas_min_setor"].items():
            cota_s = float(cfg["brl_milhoes"])
            if cota_s <= 0.0:
                continue
            linha = -(df["setor"] == setor).to_numpy(dtype=float)
            adicionar_restricao(
                nome=f"cota_min_setor[{setor}]",
                familia="Cota Minima",
                sinal_orig=">=",
                coefs=linha,
                rhs=-cota_s,
                descricao=f"Cota minima estrategica {setor} >= R$ {cota_s:.2f} Mi",
            )

        # 6) Teto de Perda Esperada (EL) agregada: sum(EL_i * x_i) <= EL_max
        adicionar_restricao(
            nome="teto_el_agregada",
            familia="Perda Esperada",
            sinal_orig="<=",
            coefs=df["el"].to_numpy(dtype=float),
            rhs=el_max,
            descricao=f"Perda esperada total da carteira <= R$ {el_max:.4f} Mi",
        )

    A_ub = np.vstack(linhas_A)
    b_ub = np.array(valores_b, dtype=float)

    # Limites individuais das variáveis: apenas não negatividade (x_i >= 0)
    bounds = [(0.0, None) for _ in range(n)]

    return {
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "bounds": bounds,
        "metadados_restricoes": metadados,
        "cenario": cenario,
        "modo_rating": modo_rating,
        "B": B,
        "el_max": el_max,
    }


def resolver_carteira(
    df: pd.DataFrame,
    params: dict,
    cenario: str = "diversificado",
    modo_rating: str = "grupo_maior_risco",
    el_max_override: float | None = None,
    tetos_setor_override: dict[str, float] | None = None,
) -> dict:
    """
    Resolve o problema de Programação Linear via solver HiGHS (`scipy.optimize.linprog`)
    e organiza as tabelas primais (alocação e custos reduzidos) e duais (folgas e preços-sombra).
    """
    modelo = montar_matrizes_lp(
        df=df,
        params=params,
        cenario=cenario,
        modo_rating=modo_rating,
        el_max_override=el_max_override,
        tetos_setor_override=tetos_setor_override,
    )

    res = linprog(
        c=modelo["c"],
        A_ub=modelo["A_ub"],
        b_ub=modelo["b_ub"],
        bounds=modelo["bounds"],
        method="highs",
    )

    # Caso o problema seja inviável ou ilimitado (status != 0), retornamos diagnóstico limpo
    if not res.success or res.status != 0:
        return {
            "viavel": False,
            "status": int(res.status),
            "mensagem": str(res.message),
            "cenario": cenario,
            "modelo": modelo,
            "z_otimo": np.nan,
            "spread_bruto_total": np.nan,
            "el_total": np.nan,
            "alocacao": pd.DataFrame(),
            "restricoes": pd.DataFrame(),
        }

    x = np.maximum(res.x, 0.0)
    # Limpeza visual apenas para zeros numéricos infinitesimais (ex.: 1e-15 -> 0.0)
    x_limpo = np.where(np.abs(x) < 1e-9, 0.0, x)

    # Recuperação da função objetivo original de maximização: Z* = -min(-pi^T x)
    z_otimo = float(-res.fun)
    spread_bruto_total = float(np.dot(df["spread"].to_numpy(dtype=float), x_limpo))
    el_total = float(np.dot(df["el"].to_numpy(dtype=float), x_limpo))

    # Custos reduzidos das variáveis x_i >= 0:
    # No problema de minimização min(-pi^T x), `res.lower.marginals` >= 0 indica quanto
    # o objetivo de minimização piora (aumenta) ao forçar x_i a subir 1 unidade acima de 0.
    # Logo, no problema de maximização, representa a penalidade (queda em Z*) por R$ 1 Mi
    # forçado em uma operação não selecionada (ou 0 para operações ativas na base).
    custos_reduzidos = np.where(np.abs(res.lower.marginals) < 1e-12, 0.0, res.lower.marginals)

    alocacao = df.copy()
    alocacao["capital_alocado"] = x_limpo
    alocacao["participacao_B"] = x_limpo / modelo["B"]
    alocacao["retorno_liq_brl"] = alocacao["pi"] * x_limpo
    alocacao["spread_bruto_brl"] = alocacao["spread"] * x_limpo
    alocacao["el_brl"] = alocacao["el"] * x_limpo
    alocacao["custo_reduzido"] = custos_reduzidos

    # Extração das folgas e preços-sombra de todas as restrições em A_ub @ x <= b_ub:
    #
    # 1) Folga (`res.ineqlin.residual`, equivalente a `res.slack` = b_ub - A_ub @ x >= 0):
    #    - Para restrições `<=`: folga = limite - valor_realizado >= 0.
    #    - Para restrições `>=` convertidas em `-A x <= -cota`:
    #      folga = (-cota) - (-valor_realizado) = valor_realizado - cota >= 0.
    #    Em ambos os casos, folga == 0 indica restrição ativa (gargalo).
    #
    # 2) Preço-sombra (`res.ineqlin.marginals`):
    #    No SciPy/HiGHS, para min c^T x sujeito a A_ub x <= b_ub, o vetor `marginals`
    #    representa d(min c^T x)/d(b_ub) <= 0 (relaxar b_ub reduz o custo mínimo).
    #    Como Z* = -(min c^T x):
    #    - Para restrições `<=`: dZ*/d(limite) = -marginal >= 0.
    #      Esse é exatamente o lambda >= 0 da formulação dual canônica (ganho em Z*
    #      por R$ 1 Mi adicional no teto).
    #    - Para restrições `>=` (multiplicadas por -1, logo b_ub = -cota):
    #      aumentar a cota mínima em +1 altera b_ub em -1, portanto:
    #      dZ*/d(cota) = (-marginal) * (-1) = marginal <= 0.
    #      Assim, mu = -marginal >= 0 é a variável dual canônica não negativa, e seu
    #      efeito sobre Z* ao elevar a cota mínima em +R$ 1 Mi é negativo (-mu).
    slacks = res.ineqlin.residual
    marginals_scipy = res.ineqlin.marginals
    lado_esquerdo_ub = modelo["A_ub"] @ x_limpo

    linhas_rest = []
    for idx, meta in enumerate(modelo["metadados_restricoes"]):
        sinal = meta["sinal_original"]
        valor_realizado = float(lado_esquerdo_ub[idx] if sinal == "<=" else -lado_esquerdo_ub[idx])
        folga = float(0.0 if abs(slacks[idx]) < 1e-9 else slacks[idx])
        marginal_raw = float(0.0 if abs(marginals_scipy[idx]) < 1e-12 else marginals_scipy[idx])
        preco_sombra_canonico = float(0.0 if abs(marginal_raw) < 1e-12 else -marginal_raw)

        # Impacto direto em Z* se aumentarmos o parâmetro original da restrição em +R$ 1 Mi
        impacto_z_mais_1mi = preco_sombra_canonico if sinal == "<=" else -preco_sombra_canonico

        linhas_rest.append(
            {
                "restricao": meta["nome"],
                "familia": meta["familia"],
                "sinal": sinal,
                "limite_brl": meta["limite_original"],
                "realizado_brl": valor_realizado,
                "folga_brl": folga,
                "ativa": folga <= TOL_ATIVA,
                "dual_scipy_raw": marginal_raw,
                "preco_sombra_canonico": preco_sombra_canonico,
                "impacto_z_por_1mi": impacto_z_mais_1mi,
                "descricao": meta["descricao"],
            }
        )

    restricoes_df = pd.DataFrame(linhas_rest)

    return {
        "viavel": True,
        "status": int(res.status),
        "mensagem": str(res.message),
        "cenario": cenario,
        "modelo": modelo,
        "z_otimo": z_otimo,
        "spread_bruto_total": spread_bruto_total,
        "el_total": el_total,
        "alocacao": alocacao,
        "restricoes": restricoes_df,
    }


def verificar_precos_sombra_perturbacao(
    resultado_base: dict,
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    """
    Valida os preços-sombra das restrições ativas por perturbação numérica bilateral (+eps / -eps).

    Justificativa teórica (Seção 6 de 02_modelo_matematico.md):
    Em vértices com degeneração primal, os preços-sombra não são únicos; as derivadas
    laterais (à direita e à esquerda) podem divergir. Perturbar o lado direito `b_ub` em
    +/- epsilon e re-otimizar confirma se `preco_sombra_canonico` coincide com ambas as
    taxas de variação marginais ou se há degeneração.
    """
    if not resultado_base["viavel"]:
        raise ValueError("Não é possível validar preços-sombra de uma solução inviável.")

    modelo = resultado_base["modelo"]
    c = modelo["c"]
    A_ub = modelo["A_ub"]
    b_ub_base = modelo["b_ub"]
    bounds = modelo["bounds"]
    z_base = resultado_base["z_otimo"]
    rest_df = resultado_base["restricoes"]

    registros = []
    for idx, row in rest_df.iterrows():
        if not row["ativa"]:
            continue

        # Perturbação positiva e negativa no limite original da restrição:
        # Para `<=`, aumentar o limite em +eps soma +eps em b_ub[idx].
        # Para `>=` (-soma <= -cota), aumentar a cota em +eps soma -eps em b_ub[idx].
        sinal_b = 1.0 if row["sinal"] == "<=" else -1.0

        b_mais = b_ub_base.copy()
        b_mais[idx] += sinal_b * epsilon
        res_mais = linprog(c=c, A_ub=A_ub, b_ub=b_mais, bounds=bounds, method="highs")

        b_menos = b_ub_base.copy()
        b_menos[idx] -= sinal_b * epsilon
        res_menos = linprog(c=c, A_ub=A_ub, b_ub=b_menos, bounds=bounds, method="highs")

        taxa_direita = (-res_mais.fun - z_base) / epsilon if res_mais.success else np.nan
        taxa_esquerda = (z_base - (-res_menos.fun)) / epsilon if res_menos.success else np.nan
        impacto_dual = row["impacto_z_por_1mi"]

        degenerada = False
        if not np.isnan(taxa_direita) and not np.isnan(taxa_esquerda):
            degenerada = abs(taxa_direita - taxa_esquerda) > 1e-6

        registros.append(
            {
                "restricao": row["restricao"],
                "sinal": row["sinal"],
                "impacto_dual_highs": impacto_dual,
                "taxa_perturb_mais_eps": taxa_direita,
                "taxa_perturb_menos_eps": taxa_esquerda,
                "diferenca_max": max(
                    abs(impacto_dual - taxa_direita) if not np.isnan(taxa_direita) else 0.0,
                    abs(impacto_dual - taxa_esquerda) if not np.isnan(taxa_esquerda) else 0.0,
                ),
                "degeneracao_detectada": degenerada,
            }
        )

    return pd.DataFrame(registros)
