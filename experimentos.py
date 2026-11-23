"""
Script de execução dos experimentos, análises pós-ótimas, sensibilidade,
testes de inviabilidade e geração de tabelas (.csv) e gráficos (.png)
para a Seção 5 do relatório.

Uso:
    python experimentos.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from modelo import (
    RAIZ_REPO,
    carregar_dados,
    resolver_carteira,
    verificar_precos_sombra_perturbacao,
)


DIR_RESULTADOS = RAIZ_REPO / "resultados"


def configurar_estilo_graficos() -> None:
    """Padroniza fonte, grade e nitidez dos gráficos para inclusão no relatório."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Desativa interpretacao automatica de '$...$' como LaTeX para que 'R$' nunca vire italico
            "text.parse_math": False,
        }
    )


def gerar_tabelas_comparativas(
    res_irr: dict,
    res_div: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constrói e salva as tabelas comparativas entre o Cenário Irrestrito (base)
    e o Cenário Diversificado (completo): por operação, por setor, por rating e resumo global.
    """
    B = float(params["B"])
    df_irr = res_irr["alocacao"]
    df_div = res_div["alocacao"]

    # 1) Comparativo operação a operação
    comp_ops = df_irr[
        ["id", "cliente", "setor", "rating", "garantia", "spread", "pd", "lgd", "el", "pi"]
    ].copy()
    comp_ops["alocacao_irrestrito_brl"] = df_irr["capital_alocado"]
    comp_ops["alocacao_diversificado_brl"] = df_div["capital_alocado"]
    comp_ops["diferenca_brl"] = comp_ops["alocacao_diversificado_brl"] - comp_ops["alocacao_irrestrito_brl"]
    comp_ops["custo_reduzido_irrestrito"] = df_irr["custo_reduzido"]
    comp_ops["custo_reduzido_diversificado"] = df_div["custo_reduzido"]
    comp_ops.to_csv(DIR_RESULTADOS / "comparativo_operacoes.csv", index=False)

    # 2) Resumo por setor econômico
    ordem_setores = list(params["tetos_setor"].keys())
    set_irr = comp_ops.groupby("setor")["alocacao_irrestrito_brl"].sum().reindex(ordem_setores, fill_value=0.0)
    set_div = comp_ops.groupby("setor")["alocacao_diversificado_brl"].sum().reindex(ordem_setores, fill_value=0.0)

    comp_setores = pd.DataFrame(
        {
            "setor": ordem_setores,
            "irrestrito_brl": set_irr.values,
            "irrestrito_pct_B": 100.0 * set_irr.values / B,
            "diversificado_brl": set_div.values,
            "diversificado_pct_B": 100.0 * set_div.values / B,
            "diferenca_brl": set_div.values - set_irr.values,
            "cota_min_brl": [float(params["cotas_min_setor"][s]["brl_milhoes"]) for s in ordem_setores],
            "teto_setor_brl": [float(params["tetos_setor"][s]["brl_milhoes"]) for s in ordem_setores],
        }
    )
    comp_setores.to_csv(DIR_RESULTADOS / "comparativo_setores.csv", index=False)

    # 3) Resumo por faixa de rating (ordenado do menor para o maior risco)
    ordem_ratings = list(params["tetos_rating"]["por_rating_individual"].keys())
    ratings_maior_risco = set(params["tetos_rating"]["grupo_maior_risco"]["ratings_incluidos"])
    rat_irr = comp_ops.groupby("rating")["alocacao_irrestrito_brl"].sum().reindex(ordem_ratings, fill_value=0.0)
    rat_div = comp_ops.groupby("rating")["alocacao_diversificado_brl"].sum().reindex(ordem_ratings, fill_value=0.0)

    comp_ratings = pd.DataFrame(
        {
            "rating": ordem_ratings,
            "grupo_risco": [
                "AA(bra) e inferiores (teto agregado 20%)" if r in ratings_maior_risco else "Baixo risco (AAA/AA+)"
                for r in ordem_ratings
            ],
            "irrestrito_brl": rat_irr.values,
            "irrestrito_pct_B": 100.0 * rat_irr.values / B,
            "diversificado_brl": rat_div.values,
            "diversificado_pct_B": 100.0 * rat_div.values / B,
            "diferenca_brl": rat_div.values - rat_irr.values,
        }
    )
    comp_ratings.to_csv(DIR_RESULTADOS / "comparativo_ratings.csv", index=False)

    # 4) Resumo executivo global (Z*, Delta Z, EL e Delta EL)
    z_irr, z_div = res_irr["z_otimo"], res_div["z_otimo"]
    el_irr, el_div = res_irr["el_total"], res_div["el_total"]
    delta_z = z_irr - z_div
    delta_el = el_div - el_irr

    resumo = pd.DataFrame(
        [
            {
                "cenario": "Irrestrito (B + L_c)",
                "capital_alocado_brl": df_irr["capital_alocado"].sum(),
                "z_otimo_brl": z_irr,
                "retorno_liq_pct_B": 100.0 * z_irr / B,
                "spread_bruto_brl": res_irr["spread_bruto_total"],
                "el_total_brl": el_irr,
                "el_pct_B": 100.0 * el_irr / B,
                "num_operacoes_ativas": int((df_irr["capital_alocado"] > 1e-6).sum()),
                "delta_z_custo_diversificacao_brl": 0.0,
                "delta_z_pct": 0.0,
                "delta_el_brl": 0.0,
                "delta_el_pct": 0.0,
            },
            {
                "cenario": "Diversificado (Completo)",
                "capital_alocado_brl": df_div["capital_alocado"].sum(),
                "z_otimo_brl": z_div,
                "retorno_liq_pct_B": 100.0 * z_div / B,
                "spread_bruto_brl": res_div["spread_bruto_total"],
                "el_total_brl": el_div,
                "el_pct_B": 100.0 * el_div / B,
                "num_operacoes_ativas": int((df_div["capital_alocado"] > 1e-6).sum()),
                "delta_z_custo_diversificacao_brl": delta_z,
                "delta_z_pct": -100.0 * delta_z / z_irr,
                "delta_el_brl": delta_el,
                "delta_el_pct": 100.0 * delta_el / el_irr,
            },
        ]
    )
    resumo.to_csv(DIR_RESULTADOS / "resumo_cenarios.csv", index=False)

    return comp_ops, comp_setores, comp_ratings, resumo


def executar_curva_tradeoff_el(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Varia EL_max de 0,01% até 0,75% de B (R$ 0,05 Mi a R$ 3,75 Mi) em passos finos,
    incluindo explicitamente os limiares analíticos críticos para capturar com precisão:
    - Região inviável (abaixo de R$ 0,1458 Mi = 0,02916% de B)
    - Subutilização de orçamento (entre 0,02916% e ~0,1176%, passando por 0,108%)
    - Crescimento côncavo com orçamento pleno (inclui o cenário base em 0,35%)
    - Saturação onde EL_max deixa de ser ativa (a partir de ~0,4169% = R$ 2,0846 Mi)
    """
    B = float(params["B"])
    grade_pct = np.round(np.arange(0.01, 0.755, 0.005), 4)
    pontos_criticos_pct = np.array([0.02916, 0.108, 0.35, 0.416925])
    todos_pct = np.unique(np.sort(np.concatenate([grade_pct, pontos_criticos_pct])))

    registros = []
    for pct in todos_pct:
        frac = float(pct / 100.0)
        el_max_brl = B * frac
        res = resolver_carteira(df, params, cenario="diversificado", el_max_override=el_max_brl)

        if not res["viavel"]:
            registros.append(
                {
                    "el_max_pct_B": pct,
                    "el_max_frac": frac,
                    "el_max_brl": el_max_brl,
                    "viavel": False,
                    "status": res["status"],
                    "capital_alocado_brl": np.nan,
                    "z_otimo_brl": np.nan,
                    "spread_bruto_brl": np.nan,
                    "el_realizado_brl": np.nan,
                    "folga_el_brl": np.nan,
                    "preco_sombra_el": np.nan,
                    "el_ativa": False,
                    "regime": "Inviável",
                }
            )
            continue

        linha_el = res["restricoes"].loc[res["restricoes"]["restricao"] == "teto_el_agregada"].iloc[0]
        cap_alocado = float(res["alocacao"]["capital_alocado"].sum())
        el_ativa = bool(linha_el["ativa"])

        if not el_ativa:
            regime = "Saturado (EL_max inativa)"
        elif cap_alocado < B - 1e-6:
            regime = "Factível (orçamento parcial)"
        else:
            regime = "Ativo (orçamento pleno)"

        registros.append(
            {
                "el_max_pct_B": pct,
                "el_max_frac": frac,
                "el_max_brl": el_max_brl,
                "viavel": True,
                "status": res["status"],
                "capital_alocado_brl": cap_alocado,
                "z_otimo_brl": res["z_otimo"],
                "spread_bruto_brl": res["spread_bruto_total"],
                "el_realizado_brl": res["el_total"],
                "folga_el_brl": float(linha_el["folga_brl"]),
                "preco_sombra_el": float(linha_el["preco_sombra_canonico"]),
                "el_ativa": el_ativa,
                "regime": regime,
            }
        )

    df_curva = pd.DataFrame(registros)
    df_curva.to_csv(DIR_RESULTADOS / "curva_tradeoff_el.csv", index=False)
    return df_curva


def executar_sensibilidade_combinada(
    df: pd.DataFrame,
    params: dict,
    z_base: float,
) -> pd.DataFrame:
    """
    Executa a grade 3x3 variando teto setorial uniforme (25%, 30%, 35% de B)
    contra o limite de perda esperada EL_max (0,25%, 0,35%, 0,45% de B).
    """
    B = float(params["B"])
    alphas = [0.25, 0.30, 0.35]
    el_fracs = [0.0025, 0.0035, 0.0045]

    registros = []
    for alpha in alphas:
        tetos_override = {s: B * alpha for s in params["tetos_setor"].keys()}
        for el_frac in el_fracs:
            el_brl = B * el_frac
            res = resolver_carteira(
                df,
                params,
                cenario="diversificado",
                el_max_override=el_brl,
                tetos_setor_override=tetos_override,
            )
            registros.append(
                {
                    "teto_setor_pct_B": 100.0 * alpha,
                    "teto_setor_brl": B * alpha,
                    "el_max_pct_B": 100.0 * el_frac,
                    "el_max_brl": el_brl,
                    "viavel": res["viavel"],
                    "status": res["status"],
                    "z_otimo_brl": res["z_otimo"],
                    "el_realizado_brl": res["el_total"],
                    "delta_z_vs_base_brl": res["z_otimo"] - z_base if res["viavel"] else np.nan,
                    "num_operacoes_ativas": (
                        int((res["alocacao"]["capital_alocado"] > 1e-6).sum()) if res["viavel"] else 0
                    ),
                }
            )

    df_grade = pd.DataFrame(registros)
    df_grade.to_csv(DIR_RESULTADOS / "sensibilidade_combinada.csv", index=False)
    return df_grade


def executar_testes_inviabilidade(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Executa os dois testes obrigatórios de inviabilidade estrutural:
    (a) Teto setorial de Energia Elétrica = 10% de B (abaixo da cota mínima de 15%).
    (b) Limite agregado EL_max = R$ 0,05 Mi (abaixo do mínimo exigido pelas cotas = R$ 0,1458 Mi).
    """
    B = float(params["B"])

    # Teste (a): conflito intra-setorial (teto < cota mínima)
    tetos_a = {s: float(cfg["brl_milhoes"]) for s, cfg in params["tetos_setor"].items()}
    tetos_a["Energia Elétrica"] = 0.10 * B
    res_a = resolver_carteira(df, params, cenario="diversificado", tetos_setor_override=tetos_a)

    # Teste (b): conflito entre EL_max e cotas mínimas estratégicas
    res_b = resolver_carteira(df, params, cenario="diversificado", el_max_override=0.05)

    registros = [
        {
            "teste": "Teste (a) - Conflito Setorial",
            "parametro_alterado": "Teto de Energia Elétrica = 10% (R$ 50 Mi) < Cota mínima 15% (R$ 75 Mi)",
            "viavel": res_a["viavel"],
            "status_solver": res_a["status"],
            "mensagem_solver": res_a["mensagem"],
            "explicacao_matematica": "Viola a condição necessária intra-setorial gamma_s <= alpha_s.",
        },
        {
            "teste": "Teste (b) - Limite Extremo de Risco",
            "parametro_alterado": "EL_max = R$ 0,05 Mi (0,01% de B) < Mínimo das cotas (R$ 0,1458 Mi)",
            "viavel": res_b["viavel"],
            "status_solver": res_b["status"],
            "mensagem_solver": res_b["mensagem"],
            "explicacao_matematica": "Mesmo alocando apenas as cotas mínimas (R$ 135 Mi) nos ativos AAA(bra), EL >= 0,1458 Mi.",
        },
    ]

    df_inv = pd.DataFrame(registros)
    df_inv.to_csv(DIR_RESULTADOS / "testes_inviabilidade.csv", index=False)
    return df_inv


def plotar_alocacao_setor(comp_setores: pd.DataFrame) -> None:
    """Gráfico 1: Alocação por setor (Irrestrito vs. Diversificado) com tetos e cotas."""
    rotulos_curtos = [
        "Energia\nElétrica",
        "Saneamento\ne Logística",
        "Agronegócio\ne Alimentos",
        "Varejo\ne Saúde",
        "Indústria de Base\ne Mineração",
    ]
    x = np.arange(len(comp_setores))
    largura = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    barras_irr = ax.bar(
        x - largura / 2,
        comp_setores["irrestrito_brl"],
        width=largura,
        label="Cenário Irrestrito (B + Lc)",
        color="#7A8B99",
        edgecolor="#2C3E50",
        linewidth=0.7,
    )
    barras_div = ax.bar(
        x + largura / 2,
        comp_setores["diversificado_brl"],
        width=largura,
        label="Cenário Diversificado (Completo)",
        color="#1F6FEB",
        edgecolor="#0A3069",
        linewidth=0.7,
    )

    # Linhas de política setorial
    ax.axhline(150.0, color="#C0392B", linestyle="--", linewidth=1.4, label="Teto setorial (30% = R$ 150 Mi)")
    ax.plot(
        [0 - largura, 0 + largura],
        [75.0, 75.0],
        color="#1E8449",
        linestyle="-",
        linewidth=2.3,
        label="Cotas mínimas (Energia R$ 75 Mi; Agro R$ 60 Mi)",
    )
    ax.plot([2 - largura, 2 + largura], [60.0, 60.0], color="#1E8449", linestyle="-", linewidth=2.3)

    # Anotação numérica sobre as barras
    for barras in (barras_irr, barras_div):
        for barra in barras:
            h = barra.get_height()
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                h + 3.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos_curtos)
    ax.set_ylabel("Capital Alocado (R$ milhões)")
    ax.set_title("Alocação de Capital por Setor Econômico: Irrestrito vs. Diversificado")
    ax.set_ylim(0, 232)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "grafico_01_alocacao_setor.png", bbox_inches="tight")
    plt.close(fig)


def plotar_alocacao_rating(comp_ratings: pd.DataFrame) -> None:
    """Gráfico 2: Alocação por faixa de rating destacando o grupo de maior risco."""
    x = np.arange(len(comp_ratings))
    largura = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    # Destaque visual para as faixas sujeitas ao teto agregado (AA(bra) e inferiores, índices >= 2)
    ax.axvspan(1.5, len(comp_ratings) - 0.5, color="#FDEDEC", alpha=0.65, zorder=0)
    ax.axvline(1.5, color="#C0392B", linestyle=":", linewidth=1.3, zorder=1)

    barras_irr = ax.bar(
        x - largura / 2,
        comp_ratings["irrestrito_brl"],
        width=largura,
        label="Cenário Irrestrito (B + Lc)",
        color="#7A8B99",
        edgecolor="#2C3E50",
        linewidth=0.7,
        zorder=2,
    )
    barras_div = ax.bar(
        x + largura / 2,
        comp_ratings["diversificado_brl"],
        width=largura,
        label="Cenário Diversificado (Completo)",
        color="#1F6FEB",
        edgecolor="#0A3069",
        linewidth=0.7,
        zorder=2,
    )

    for barras in (barras_irr, barras_div):
        for barra in barras:
            h = barra.get_height()
            if h > 0.05:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    h + 3.0,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    fontweight="bold",
                )

    soma_risco_irr = comp_ratings.iloc[2:]["irrestrito_brl"].sum()
    soma_risco_div = comp_ratings.iloc[2:]["diversificado_brl"].sum()
    ax.text(
        4.5,
        195,
        f"Grupo AA(bra) e inferiores (teto <= R$ 100,0 Mi):\n"
        f"Irrestrito: R$ {soma_risco_irr:.1f} Mi (55,0%)\n"
        f"Diversificado: R$ {soma_risco_div:.1f} Mi (20,0% — teto ativo)",
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#C0392B", alpha=0.95),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_ratings["rating"], rotation=0)
    ax.set_ylabel("Capital Alocado (R$ milhões)")
    ax.set_title("Alocação de Capital por Faixa de Rating: Irrestrito vs. Diversificado")
    ax.set_ylim(0, 285)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "grafico_02_alocacao_rating.png", bbox_inches="tight")
    plt.close(fig)


def plotar_curva_tradeoff(df_curva: pd.DataFrame, res_irr: dict, res_div: dict) -> None:
    """Gráfico 3: Curva de trade-off Z* vs. EL_max com região inviável, base e saturação."""
    viaveis = df_curva[df_curva["viavel"]].copy()
    inviaveis = df_curva[~df_curva["viavel"]].copy()

    el_min_viavel = float(viaveis["el_max_brl"].min())
    z_min_viavel = float(viaveis.loc[viaveis["el_max_brl"].idxmin(), "z_otimo_brl"])

    # Primeiro ponto em que EL_max deixa de ser ativa (folga > 0)
    saturados = viaveis[~viaveis["el_ativa"]]
    el_sat = float(saturados["el_realizado_brl"].iloc[0])
    z_sat = float(saturados["z_otimo_brl"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.8, 5.6))

    # Região inviável à esquerda do limiar mínimo factível
    ax.axvspan(0.0, el_min_viavel, color="#FADBD8", alpha=0.75, label="Região inviável (EL_max < R$ 0,1458 Mi)")
    ax.axvline(el_min_viavel, color="#C0392B", linestyle="--", linewidth=1.2)

    # Curva ótima Z*(EL_max)
    ax.plot(
        viaveis["el_max_brl"],
        viaveis["z_otimo_brl"],
        color="#1F6FEB",
        linewidth=2.4,
        label="Fronteira ótima diversificada Z*(EL_max)",
    )

    # Referência do cenário irrestrito
    ax.axhline(
        res_irr["z_otimo"],
        color="#7A8B99",
        linestyle=":",
        linewidth=1.6,
        label=f"Cenário Irrestrito (Z* = R$ {res_irr['z_otimo']:.4f} Mi; EL = R$ {res_irr['el_total']:.2f} Mi)",
    )

    # Marcadores principais
    ax.scatter([el_min_viavel], [z_min_viavel], color="#C0392B", s=55, zorder=5)
    ax.annotate(
        f"Mínimo factível\nEL = R$ {el_min_viavel:.4f} Mi (0,029%)\nZ* = R$ {z_min_viavel:.2f} Mi",
        xy=(el_min_viavel, z_min_viavel),
        xytext=(0.45, 2.05),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C0392B", alpha=0.95),
    )

    # Referência de 0,108% de B (EL mínima para alocar 100% de B em AAA)
    el_108 = 0.54
    z_108 = float(viaveis.loc[(viaveis["el_max_brl"] - el_108).abs().idxmin(), "z_otimo_brl"])
    ax.scatter([el_108], [z_108], color="#D68910", s=50, zorder=5)
    ax.annotate(
        "Ref. 0,108% de B\n(R$ 0,54 Mi)",
        xy=(el_108, z_108),
        xytext=(0.75, 4.35),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#D68910", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D68910", alpha=0.95),
    )

    # Cenário Base Diversificado (EL_max = R$ 1,75 Mi)
    el_base = res_div["el_total"]
    z_base = res_div["z_otimo"]
    ax.scatter([el_base], [z_base], color="#1E8449", s=75, zorder=6, label=f"Cenário Base (EL = R$ {el_base:.2f} Mi; Z* = R$ {z_base:.4f} Mi)")
    ax.annotate(
        f"Cenário Base\nEL = R$ {el_base:.2f} Mi (0,35%)\nZ* = R$ {z_base:.4f} Mi",
        xy=(el_base, z_base),
        xytext=(1.35, 5.25),
        fontsize=8.7,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1E8449", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#1E8449", alpha=0.95),
    )

    # Ponto de saturação
    ax.scatter([el_sat], [z_sat], color="#6C3483", s=65, zorder=6)
    ax.axvline(el_sat, color="#6C3483", linestyle="-.", linewidth=1.1, alpha=0.7)
    ax.annotate(
        f"Saturação de EL_max\nEL = R$ {el_sat:.4f} Mi (0,417%)\nZ* = R$ {z_sat:.4f} Mi",
        xy=(el_sat, z_sat),
        xytext=(2.42, 6.02),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#6C3483", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#6C3483", alpha=0.95),
    )

    ax.set_xlim(0.0, 3.80)
    ax.set_ylim(1.0, 8.35)
    ax.set_xlabel("Limite Máximo de Perda Esperada — EL_max (R$ milhões)")
    ax.set_ylabel("Retorno Líquido Ótimo — Z* (R$ milhões)")
    ax.set_title("Curva de Trade-off Risco vs. Retorno: Z* em Função de EL_max", pad=26)

    # Eixo superior em % de B para leitura imediata em ambas as escalas
    ax_top = ax.twiny()
    ax_top.set_xlim(0.0, 100.0 * 3.80 / 500.0)
    ax_top.set_xlabel("EL_max como Percentual do Orçamento B (%)")
    ax_top.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "grafico_03_curva_tradeoff_el.png", bbox_inches="tight")
    plt.close(fig)


def plotar_sensibilidade_combinada(df_grade: pd.DataFrame) -> None:
    """Gráfico 4: Mapa de calor (heatmap) da grade Teto Setorial x EL_max."""
    pivot_z = df_grade.pivot(index="teto_setor_pct_B", columns="el_max_pct_B", values="z_otimo_brl")
    pivot_delta = df_grade.pivot(index="teto_setor_pct_B", columns="el_max_pct_B", values="delta_z_vs_base_brl")
    pivot_el = df_grade.pivot(index="teto_setor_pct_B", columns="el_max_pct_B", values="el_realizado_brl")

    # Ordena de baixo para cima: 25%, 30%, 35%
    pivot_z = pivot_z.sort_index(ascending=True)
    pivot_delta = pivot_delta.reindex(pivot_z.index)
    pivot_el = pivot_el.reindex(pivot_z.index)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    im = ax.imshow(pivot_z.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=6.35, vmax=7.25)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label("Retorno Líquido Ótimo Z* (R$ milhões)")

    alphas_pct = list(pivot_z.index)
    els_pct = list(pivot_z.columns)

    ax.set_xticks(np.arange(len(els_pct)))
    ax.set_xticklabels([f"{v:.2f}%\n(R$ {5.0*v:.2f} Mi)" for v in els_pct])
    ax.set_yticks(np.arange(len(alphas_pct)))
    ax.set_yticklabels([f"{v:.0f}%\n(R$ {5.0*v:.0f} Mi)" for v in alphas_pct])
    ax.set_xlabel("Limite Máximo de Perda Esperada (EL_max)")
    ax.set_ylabel("Teto Máximo por Setor Econômico (alpha_s)")
    ax.set_title("Sensibilidade Combinada: Z* por Teto Setorial vs. EL_max")
    ax.grid(False)

    # Anotação interna em cada célula da grade 3x3
    for i, a_pct in enumerate(alphas_pct):
        for j, el_pct in enumerate(els_pct):
            z_val = pivot_z.iloc[i, j]
            d_val = pivot_delta.iloc[i, j]
            el_real = pivot_el.iloc[i, j]
            cor_txt = "white" if z_val > 6.95 else "#111111"
            sinal_d = "+" if d_val > 1e-6 else ""
            eh_base = abs(a_pct - 30.0) < 1e-6 and abs(el_pct - 0.35) < 1e-6
            tag_base = " [Base]" if eh_base else ""
            texto = f"Z* = R$ {z_val:.4f} Mi{tag_base}\n({sinal_d}{d_val:.4f} vs. base)\nEL real = R$ {el_real:.2f} Mi"
            ax.text(j, i, texto, ha="center", va="center", color=cor_txt, fontsize=8.6, fontweight="bold")

            # Destaque para o cenário base (30%, 0,35%)
            if eh_base:
                rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#D35400", linewidth=2.6)
                ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "grafico_04_sensibilidade_combinada.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    configurar_estilo_graficos()

    print("=== 1. CARREGAMENTO E VALIDACAO DOS DADOS ===")
    df, params = carregar_dados()
    print(f"Operacoes carregadas: {len(df)} | Orcamento B = R$ {params['B']:.2f} Mi")

    print("\n=== 2. CENARIOS IRRESTRITO VS. DIVERSIFICADO ===")
    res_irr = resolver_carteira(df, params, cenario="irrestrito")
    res_div = resolver_carteira(df, params, cenario="diversificado")

    comp_ops, comp_setores, comp_ratings, resumo = gerar_tabelas_comparativas(res_irr, res_div, params)
    print(resumo.to_string(index=False))
    ops_sel = comp_ops[comp_ops["alocacao_diversificado_brl"] > 1e-6][
        ["id", "setor", "rating", "alocacao_irrestrito_brl", "alocacao_diversificado_brl"]
    ]
    print("\nOperacoes selecionadas no cenario diversificado:")
    print(ops_sel.to_string(index=False))

    print("\n=== 3. ANALISE POS-OTIMA (FOLGAS, PRECOS-SOMBRA E PERTURBACAO) ===")
    rest_div = res_div["restricoes"]
    rest_div.to_csv(DIR_RESULTADOS / "folgas_precos_sombra_diversificado.csv", index=False)
    print("Restricoes ativas (gargalos):")
    print(
        rest_div[rest_div["ativa"]][
            ["restricao", "sinal", "limite_brl", "realizado_brl", "folga_brl", "preco_sombra_canonico", "impacto_z_por_1mi"]
        ].to_string(index=False)
    )

    df_perturb = verificar_precos_sombra_perturbacao(res_div, epsilon=1e-3)
    df_perturb.to_csv(DIR_RESULTADOS / "validacao_perturbacao_degeneracao.csv", index=False)
    print("\nValidacao bilateral (+/- 1e-3) para checagem de degeneracao primal:")
    print(df_perturb.to_string(index=False))

    print("\n=== 4. CURVA DE TRADE-OFF (Z* VS. EL_MAX) ===")
    df_curva = executar_curva_tradeoff_el(df, params)
    viaveis = df_curva[df_curva["viavel"]]
    sat = viaveis[~viaveis["el_ativa"]].iloc[0]
    print(
        f"Pontos avaliados: {len(df_curva)} | Inviaveis: {(~df_curva['viavel']).sum()} | "
        f"Minimo factivel: EL = R$ {viaveis['el_max_brl'].min():.4f} Mi ({viaveis['el_max_pct_B'].min():.4f}%) | "
        f"Saturacao: EL = R$ {sat['el_realizado_brl']:.4f} Mi ({100*sat['el_realizado_brl']/500:.4f}%), Z* = R$ {sat['z_otimo_brl']:.4f} Mi"
    )

    print("\n=== 5. SENSIBILIDADE COMBINADA (TETO SETORIAL X EL_MAX) ===")
    df_grade = executar_sensibilidade_combinada(df, params, z_base=res_div["z_otimo"])
    print(df_grade.to_string(index=False))

    print("\n=== 6. TESTES DE INVIABILIDADE ===")
    df_inv = executar_testes_inviabilidade(df, params)
    print(df_inv[["teste", "viavel", "status_solver", "mensagem_solver"]].to_string(index=False))

    print("\n=== 7. GERACAO DOS GRAFICOS (PNG 300 DPI) ===")
    plotar_alocacao_setor(comp_setores)
    plotar_alocacao_rating(comp_ratings)
    plotar_curva_tradeoff(df_curva, res_irr, res_div)
    plotar_sensibilidade_combinada(df_grade)
    print(f"Arquivos gerados com sucesso em: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
