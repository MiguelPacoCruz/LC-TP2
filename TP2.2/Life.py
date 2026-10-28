import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from random import Random
    import pandas as pd
    import matplotlib.pyplot as plt
    from pysmt.shortcuts import (
        Symbol, And, Or, Not, Iff, Equals, Ite, Plus, Int, Solver
    )
    from pysmt.typing import BOOL


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Game of Life com espaço de estados finito

    O autómato é modelado como um *first-order transition system* (FOTS): um predicado `init` sobre o estado inicial e um predicado `trans` sobre o par (pré-estado, pós-estado). Primeiro formula-se a relação de transição em lógica; depois implementa-se em pySMT, com o modo de transição (síncrono ou assíncrono) escolhido por parâmetro.

    ## Formulação lógica

    **Estado.** Cada célula é uma variável booleana $c_{i,j}$ (e $c'_{i,j}$ no pós-estado), com $0 \le i,j \le N+1$. Os três tipos de célula são:

    - normais: $\mathcal{N}=\{1..N\}^2$
    - iniciais: $\mathcal{I}=\{(i,j) : i=0 \vee j=0\}$, em número $2N+1$
    - finais: $\mathcal{F}$, com $i=N+1$ ou $j=N+1$ e que não são iniciais, em número $2N+3$

    **Regra B3|S23.** Seja $V(i,j)$ o conjunto das 8 vizinhas de $(i,j)$ e $n(i,j)=\sum_{(u,v)\in V(i,j)} [c_{u,v}]$ o número de vizinhas vivas:

    $$R(i,j)\;\equiv\; n(i,j)=3 \;\vee\; \big(c_{i,j}\wedge n(i,j)=2\big)$$

    **Manutenção da fronteira** (igual nos dois modos):

    $$\mathit{Fr}\;\equiv\;\bigwedge_{(i,j)\in\mathcal{I}}\big(c'_{i,j}\leftrightarrow c_{i,j}\big)\;\wedge\;\bigwedge_{(i,j)\in\mathcal{F}}\neg c'_{i,j}$$

    **Modo síncrono** (todas as células normais transitam em simultâneo):

    $$T_{\mathit{sync}}\;\equiv\;\mathit{Fr}\;\wedge\;\bigwedge_{(i,j)\in\mathcal{N}}\big(c'_{i,j}\leftrightarrow R(i,j)\big)$$

    **Modo assíncrono** (uma e só uma célula normal transita). Cada disjunto é um caso, com a sua atualização e manutenção:

    $$T_{\mathit{async}}\;\equiv\;\mathit{Fr}\;\wedge\;\bigvee_{(a,b)\in\mathcal{N}}\Big(\underbrace{c'_{a,b}\leftrightarrow R(a,b)}_{\text{Atualização}}\;\wedge\;\underbrace{\bigwedge_{(i,j)\in\mathcal{N}\setminus\{(a,b)\}}\big(c'_{i,j}\leftrightarrow c_{i,j}\big)}_{\text{Manutenção}}\Big)$$

    A pré-condição é vazia: a célula escolhida pode ficar com o mesmo valor (*stuttering*), o que mantém a relação total (sem deadlocks). A variante estrita acrescentaria a cada disjunto a pré-condição $R(a,b)\not\leftrightarrow c_{a,b}$.
    """)
    return


@app.function
def declare(t, N):
    if not N > 16:
        raise ValueError("N must be greater than 16")
    return {
        (i, j): Symbol(f"cell_{i}_{j}_{t}", BOOL)
        for i in range(0, N + 2)
        for j in range(0, N + 2)
    }


@app.function
def iniciais(N):
    return [(i, 0) for i in range(N + 1)] + [(0, j) for j in range(1, N + 1)]


@app.function
def finais(N):
    return [(i, N + 1) for i in range(N + 2)] + [(N + 1, j) for j in range(N + 1)]


@app.function
def normais(N):
    return [(i, j) for i in range(1, N + 1) for j in range(1, N + 1)]


@app.function
def init(state, N):
    constraints = []
    for (i, j) in normais(N):
        if i <= 3 and j <= 3:
            constraints.append(state[(i, j)])
        else:
            constraints.append(Not(state[(i, j)]))
    return And(constraints)


@app.function
def initial_final_cells(state, N, seed=0):
    rng = Random(seed)
    constraints = []

    # 2N+1 células iniciais: valor aleatório (prob. 1/2), constante no tempo
    for c in iniciais(N):
        constraints.append(state[c] if rng.random() < 0.5 else Not(state[c]))

    # 2N+3 células finais: sempre mortas
    for c in finais(N):
        constraints.append(Not(state[c]))

    return And(constraints)


@app.function
def vizinhos(i, j):
    return [
        (i + di, j + dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if (di, dj) != (0, 0)
    ]


@app.function
def regra(curr, i, j):
    n = Plus([Ite(curr[v], Int(1), Int(0)) for v in vizinhos(i, j)])
    return Or(Equals(n, Int(3)), And(curr[(i, j)], Equals(n, Int(2))))


@app.function
def fronteira(curr, prox, N):
    return And(
        [Iff(prox[c], curr[c]) for c in iniciais(N)]
        + [Not(prox[c]) for c in finais(N)]
    )


@app.function
def trans_sync(curr, prox, N):
    return And(
        fronteira(curr, prox, N),
        And([Iff(prox[(i, j)], regra(curr, i, j)) for (i, j) in normais(N)]),
    )


@app.function
def trans_async(curr, prox, N):
    cel = normais(N)
    casos = []
    for a in cel:
        # Pré: vazia (permite stuttering)
        atualizacao = Iff(prox[a], regra(curr, *a))
        manutencao = And([Iff(prox[c], curr[c]) for c in cel if c != a])
        casos.append(And(atualizacao, manutencao))
    return And(fronteira(curr, prox, N), Or(casos))


@app.function
def trans(curr, prox, N, modo="sync"):
    if modo == "sync":
        return trans_sync(curr, prox, N)
    if modo == "async":
        return trans_async(curr, prox, N)
    raise ValueError("modo deve ser 'sync' ou 'async'")


@app.function
def mostra(grelha, N):
    def linha(i):
        r = grelha[i]
        return (
            str(r[0]) + " | "
            + " ".join(str(v) for v in r[1:N + 1])
            + " | " + str(r[N + 1])
        )
    larg = len(linha(1))
    print(linha(0))
    print("-" * larg)
    for i in range(1, N + 1):
        print(linha(i))
    print("-" * larg)
    print(linha(N + 1))


@app.function
def gera(N, k, seed=0, modo="sync", verbose=True):
    with Solver(name="z3") as s:
        states = [declare(t, N) for t in range(k + 1)]

        s.add_assertion(init(states[0], N))
        s.add_assertion(initial_final_cells(states[0], N, seed))
        for t in range(k):
            s.add_assertion(trans(states[t], states[t + 1], N, modo))

        if not s.solve():
            if verbose:
                print(f"Sem traço válido (N={N}, k={k}, seed={seed}, modo={modo}).")
            return None

        model = s.get_model()
        traco = [
            [[1 if model.get_py_value(st[(i, j)]) else 0
              for j in range(N + 2)]
             for i in range(N + 2)]
            for st in states
        ]
        if verbose:
            print(f"N={N}, k={k}, seed={seed}, modo={modo}")
            for t, grelha in enumerate(traco):
                print(f"Estado {t}:")
                mostra(grelha, N)
                print()
        return traco


@app.cell
def _():
    modo = mo.ui.dropdown(
        options=["sync", "async"], value="sync", label="Modo de transição"
    )
    modo
    return (modo,)


@app.cell
def _(modo):
    gera(N=17, k=5, seed=0, modo=modo.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Testes de tempo

    **Método.** Para cada par $(N,k)$ mediu-se, em cada modo, o tempo de construir a fórmula em pySMT e de resolver com o z3 um traço de $k$ passos (semente 0, uma execução por ponto). Os valores abaixo são os que se obtiveram; os tempos absolutos dependem da máquina, por isso o que se compara são as razões e os expoentes.

    **Tempo esperado.** O tamanho da fórmula por passo cresce com $N^2$ no síncrono e com $N^4$ no assíncrono ($N^2$ disjuntos, cada um com $N^2-1$ igualdades de manutenção), e linearmente com $k$ nos dois. Partindo do tempo medido na célula base ($N_0=17$, $k_0=2$):

    $$t_{\mathit{esp}}(N,k)\;=\;t_{\mathit{base}}\cdot\Big(\tfrac{N}{N_0}\Big)^{e}\cdot\tfrac{k}{k_0},\qquad e=2\ (\mathit{sync}),\;e=4\ (\mathit{async})$$

    Um `medido/esperado` acima de 1 significa que o solver cresce mais depressa do que o tamanho da fórmula.
    """)
    return


@app.cell(hide_code=True)
def _():
    medidas = pd.DataFrame(
        [
            (17, 2, 0.22, 0.22, 1.00, 1.99, 1.99, 1.00, 9.17),
            (17, 5, 0.51, 0.54, 0.93, 7.24, 4.97, 1.46, 14.29),
            (17, 10, 1.28, 1.08, 1.18, 17.68, 9.94, 1.78, 13.79),
            (20, 2, 0.46, 0.30, 1.54, 3.79, 3.81, 1.00, 8.19),
            (20, 5, 0.80, 0.75, 1.06, 11.54, 9.52, 1.21, 14.48),
            (20, 10, 1.62, 1.50, 1.08, 27.27, 19.04, 1.43, 16.82),
            (24, 2, 0.82, 0.43, 1.89, 7.87, 7.90, 1.00, 9.64),
            (24, 5, 1.25, 1.08, 1.15, 24.40, 19.74, 1.24, 19.59),
            (24, 10, 2.84, 2.16, 1.31, 53.21, 39.48, 1.35, 18.73),
        ],
        columns=[
            "N", "k",
            "sync (s)", "sync esperado (s)", "sync medido/esperado",
            "async (s)", "async esperado (s)", "async medido/esperado",
            "async/sync",
        ],
    )
    expoentes_ajustados = pd.DataFrame(
        [
            ("sync", 2.92, 2, 0.87, 1),
            ("async", 3.58, 4, 1.26, 1),
        ],
        columns=[
            "modo",
            "expoente de N (medido)", "expoente de N (teórico)",
            "expoente de k (medido)", "expoente de k (teórico)",
        ],
    )
    mo.vstack([
        mo.md("### Tempos medidos e esperados"),
        mo.ui.table(medidas, selection=None),
        mo.md("### Expoentes ajustados por mínimos quadrados "
              "($\\log t = \\log a + p\\log N + q\\log k$)"),
        mo.ui.table(expoentes_ajustados, selection=None),
    ])
    return (medidas,)


@app.cell(hide_code=True)
def _(medidas):
    _fig, (_ax_n, _ax_k) = plt.subplots(1, 2, figsize=(11, 4))
    _estilos = (("sync", 2, "tab:blue"), ("async", 4, "tab:red"))

    # Tempo vs N (k = 10), com a reta teórica ancorada no primeiro ponto
    _d = medidas[medidas["k"] == 10].sort_values("N")
    for _modo, _e, _cor in _estilos:
        _t = _d[f"{_modo} (s)"]
        _ax_n.loglog(_d["N"], _t, "o-", color=_cor, label=f"{_modo} (medido)")
        _ax_n.loglog(_d["N"], _t.iloc[0] * (_d["N"] / _d["N"].iloc[0]) ** _e,
                     "--", color=_cor, alpha=0.6, label=f"{_modo} (teórico, $N^{_e}$)")
    _ax_n.set_xticks(_d["N"], labels=[str(v) for v in _d["N"]])
    _ax_n.minorticks_off()
    _ax_n.set_xlabel("N")
    _ax_n.set_ylabel("tempo (s)")
    _ax_n.set_title("Tempo vs N (k = 10)")
    _ax_n.legend()
    _ax_n.grid(True, which="both", alpha=0.3)

    # Tempo vs k (N = 24), com a reta teórica (linear) ancorada no primeiro ponto
    _d = medidas[medidas["N"] == 24].sort_values("k")
    for _modo, _e, _cor in _estilos:
        _t = _d[f"{_modo} (s)"]
        _ax_k.loglog(_d["k"], _t, "o-", color=_cor, label=f"{_modo} (medido)")
        _ax_k.loglog(_d["k"], _t.iloc[0] * (_d["k"] / _d["k"].iloc[0]),
                     "--", color=_cor, alpha=0.6, label=f"{_modo} (teórico, $k$)")
    _ax_k.set_xticks(_d["k"], labels=[str(v) for v in _d["k"]])
    _ax_k.minorticks_off()
    _ax_k.set_xlabel("k")
    _ax_k.set_ylabel("tempo (s)")
    _ax_k.set_title("Tempo vs k (N = 24)")
    _ax_k.legend()
    _ax_k.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### O que se observou

    1. **Assíncrono em $N$: compatível com $N^4$.** Com $k=2$, o tempo medido coincide com o esperado (3,79 s contra 3,81 s em $N=20$; 7,87 s contra 7,90 s em $N=24$). Com $k$ maior o expoente ajustado desce para 3,58, porque o custo de $k$ (ponto seguinte) pesa mais nos $N$ pequenos: com $k=10$ o tempo só multiplica por 3,0 entre $N=17$ e $N=24$, quando a teoria dá 3,97.
    2. **Assíncrono em $k$: mais do que linear.** Para $N=17$, o tempo sobe $\times 3{,}6$ de $k=2$ para $k=5$ e $\times 8{,}9$ até $k=10$ (a teoria dá $\times 2{,}5$ e $\times 5$); o expoente ajustado é 1,26. A fórmula só cresce linearmente com $k$, por isso o custo extra deve vir do solver, que tem de escolher em cada passo qual das $N^2$ células transita (e se há *stuttering*).
    3. **Síncrono: dados pouco conclusivos.** Os tempos ficam entre 0,2 s e 2,8 s e variações de décimas de segundo alteram muito as razões (`medido/esperado` entre 0,93 e 1,89). Os expoentes ajustados (2,92 para $N$, 0,87 para $k$) são pouco fiáveis; só se pode dizer que o crescimento é suave, entre $N^2$ e $N^3$.
    4. **Assíncrono contra síncrono.** O assíncrono é entre 8 e 20 vezes mais lento, e a diferença tende a aumentar com $N$ e $k$ (coerente com o fator $N^2$ a mais na fórmula).
    5. **Consequência.** O BMC repete a verificação para $k=1,2,\dots$ e a propriedade 2 acrescenta restrições de ciclo, por isso o modo assíncrono vai ser o gargalo. A alternativa é uma codificação com variáveis seletoras (uma por célula e por passo, com um *exactly-one*), que torna a fórmula linear em $N^2$ por passo.

    *Limitações:* uma execução por ponto, uma só semente e uma só máquina; os tempos pequenos são ruidosos e o tempo esperado depende de uma única medição base.
    """)
    return


if __name__ == "__main__":
    app.run()
