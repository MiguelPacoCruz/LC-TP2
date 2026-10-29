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

    O autómato é modelado como um *first-order transition system* (FOTS): um predicado `init` sobre o estado inicial e um predicado `trans` sobre o par (pré-estado, pós-estado). Primeiro formula-se a relação de transição em lógica; depois implementa-se em pySMT, com o modo de transição (síncrono ou assíncrono) escolhido por parâmetro. A estrutura segue a da ficha 7 (FOTS, `declare`/`init`/`trans`, `gera_traco` e `bmc_always`), e os comentários indicam o exercício de origem.

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

    A estrutura de cada caso (pré-condição, atualização, manutenção) é a do exercício 2 da ficha 7. A pré-condição é vazia: a célula escolhida pode ficar com o mesmo valor (*stuttering*), o que mantém a relação total (sem deadlocks). A variante estrita acrescentaria a cada disjunto a pré-condição $R(a,b)\not\leftrightarrow c_{a,b}$.
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

    for c in iniciais(N):
        constraints.append(state[c] if rng.random() < 0.5 else Not(state[c]))

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


# estrutura Pré / Atualização / Manutenção do trans da ficha 7, exercício 2
@app.function
def trans_async(curr, prox, N):
    cel = normais(N)
    casos = []
    for a in cel:
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
def grelha_str(grelha):
    N = len(grelha) - 2

    def linha(i):
        r = grelha[i]
        return (
            str(r[0]) + " | "
            + " ".join(str(v) for v in r[1:N + 1])
            + " | " + str(r[N + 1])
        )
    larg = len(linha(1))
    return "\n".join(
        [linha(0), "-" * larg]
        + [linha(i) for i in range(1, N + 1)]
        + ["-" * larg, linha(N + 1)]
    )


@app.function
def mostra(grelha):
    print(grelha_str(grelha))


@app.function
def grelha_modelo(model, state):
    N = max(i for (i, _) in state) - 1
    return [
        [1 if model.get_py_value(state[(i, j)]) else 0 for j in range(N + 2)]
        for i in range(N + 2)
    ]


# ficha 7, exercícios 1 e 2 (declare, init, trans) agrupados para um dado N, modo e semente
@app.function
def fots(N, modo="sync", seed=0):
    def declare_(i):
        return declare(i, N)

    def init_(state):
        return And(init(state, N), initial_final_cells(state, N, seed))

    def trans_(curr, prox):
        return trans(curr, prox, N, modo)

    return declare_, init_, trans_


# ficha 7, exercício 3: gera_traco (k estados, ou seja, k-1 passos)
@app.function
def gera_traco(declare, init, trans, k, verbose=True):
    with Solver(name="z3") as s:
        states = [declare(i) for i in range(k)]

        constraints = []

        constraints.append(init(states[0]))

        for i in range(k - 1):
            constraints.append(trans(states[i], states[i + 1]))

        s.add_assertion(And(constraints))

        if s.solve():
            model = s.get_model()
            traco = [grelha_modelo(model, st) for st in states]
            if verbose:
                for i, grelha in enumerate(traco):
                    print(f"Estado {i}:")
                    mostra(grelha)
                    print()
            return traco
        if verbose:
            print("Não foi encontrado um traço válido")
        return None


@app.function
def regra_py(g, i, j):
    n = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                n += g[i + di][j + dj]
    return 1 if n == 3 or (g[i][j] == 1 and n == 2) else 0


@app.function
def passo_sync_py(g, N):
    nova = [linha[:] for linha in g]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            nova[i][j] = regra_py(g, i, j)
    return nova


@app.function
def valida_traco(traco, N, modo):
    ini, fin = set(iniciais(N)), set(finais(N))
    for t in range(len(traco) - 1):
        atual, seguinte = traco[t], traco[t + 1]
        for (i, j) in ini:
            if seguinte[i][j] != atual[i][j]:
                return False, f"passo {t}→{t+1}: célula inicial ({i},{j}) mudou"
        for (i, j) in fin:
            if seguinte[i][j] != 0:
                return False, f"passo {t}→{t+1}: célula final ({i},{j}) está viva"
        if modo == "sync":
            esperado = passo_sync_py(atual, N)
            for (i, j) in normais(N):
                if seguinte[i][j] != esperado[i][j]:
                    return False, f"passo {t}→{t+1}: célula ({i},{j}) devia ser {esperado[i][j]}"
        else:
            mudou = [(i, j) for (i, j) in normais(N) if seguinte[i][j] != atual[i][j]]
            if len(mudou) > 1:
                return False, f"passo {t}→{t+1}: mudaram {len(mudou)} células normais"
            if mudou:
                i, j = mudou[0]
                if seguinte[i][j] != regra_py(atual, i, j):
                    return False, f"passo {t}→{t+1}: ({i},{j}) não segue a regra"
    return True, f"{len(traco) - 1} passos verificados"


@app.function
def erro(state, N):
    return And([Not(state[c]) for c in normais(N)])


@app.function
def sem_erro(N):
    return lambda state: Not(erro(state, N))


# ficha 7, exercício 4: bmc_always, mas com um só solver incremental (push/pop)
# em vez de um solver novo por k; devolve o contra-exemplo (ou None) em vez de imprimir
@app.function
def bmc_always(declare, init, trans, inv, K):
    with Solver(name="z3") as s:
        states = [declare(0)]
        s.add_assertion(init(states[0]))
        for k in range(1, K + 1):
            if k > 1:
                states.append(declare(k - 1))
                s.add_assertion(trans(states[-2], states[-1]))
            s.push()
            s.add_assertion(Not(inv(states[-1])))
            if s.solve():
                model = s.get_model()
                return [grelha_modelo(model, st) for st in states]
            s.pop()
    return None


@app.function
def passo_inducao(N, k, modo="sync"):
    with Solver(name="z3") as s:
        states = [declare(t, N) for t in range(k + 1)]
        s.add_assertion(And([Not(states[0][c]) for c in finais(N)]))
        for t in range(k):
            s.add_assertion(Not(erro(states[t], N)))
            s.add_assertion(trans(states[t], states[t + 1], N, modo))
        s.add_assertion(erro(states[k], N))
        if s.solve():
            model = s.get_model()
            return [grelha_modelo(model, st) for st in states]
        return None


@app.function
def estado_inicial_py(N, seed=0):
    rng = Random(seed)
    g = [[0] * (N + 2) for _ in range(N + 2)]
    for (i, j) in iniciais(N):
        g[i][j] = 1 if rng.random() < 0.5 else 0
    for i in range(1, 4):
        for j in range(1, 4):
            g[i][j] = 1
    return g


@app.function
def orbita_sync(N, seed=0, limite=1_000_000):
    g = estado_inicial_py(N, seed)
    visto = {tuple(map(tuple, g)): 0}
    t = 0
    com_erro = all(g[i][j] == 0 for (i, j) in normais(N))
    while t < limite:
        g = passo_sync_py(g, N)
        t += 1
        if all(g[i][j] == 0 for (i, j) in normais(N)):
            com_erro = True
        chave = tuple(map(tuple, g))
        if chave in visto:
            return {
                "pre-período": visto[chave],
                "período": t - visto[chave],
                "estados distintos": t,
                "erro na órbita": com_erro,
            }
        visto[chave] = t
    return None


@app.function
def mudancas(traco, N):
    return [
        (t + 1, (i, j), traco[t][i][j], traco[t + 1][i][j])
        for t in range(len(traco) - 1)
        for (i, j) in normais(N)
        if traco[t][i][j] != traco[t + 1][i][j]
    ]


@app.function
def ciclo_sync(N, seed=0, limite=1_000_000):
    g = estado_inicial_py(N, seed)
    visto = {tuple(map(tuple, g)): 0}
    estados = [g]
    while len(estados) < limite:
        g = passo_sync_py(g, N)
        chave = tuple(map(tuple, g))
        if chave in visto:
            inicio = visto[chave]
            return inicio, estados[inicio:]
        visto[chave] = len(estados)
        estados.append(g)
    return None, None


@app.function
def classifica_celulas(ciclo, N):
    cel = normais(N)
    mortas = [(i, j) for (i, j) in cel if all(g[i][j] == 0 for g in ciclo)]
    vivas = [(i, j) for (i, j) in cel if all(g[i][j] == 1 for g in ciclo)]
    return mortas, vivas


@app.function
def alguma_constante(N):
    cel = normais(N)
    return lambda estados: Or([
        Or(And([st[c] for st in estados]), And([Not(st[c]) for st in estados]))
        for c in cel
    ])


# ficha 7, exercícios 6 e 7: bmc_eventually adaptado a G F. O ciclo pode ser uma
# auto-transição do último estado (j até k-1) e a condição só se impõe aos estados
# do ciclo; o início do ciclo obtém-se com get_py_value, como no exercício 7
@app.function
def bmc_gf(declare, init, trans, constante, bound):
    for k in range(1, bound + 1):
        with Solver(name="z3") as s:
            states = [declare(i) for i in range(k)]

            constraints = []

            constraints.append(init(states[0]))

            for i in range(k - 1):
                constraints.append(trans(states[i], states[i + 1]))

            ciclos = [
                And(trans(states[-1], states[j]), constante(states[j:]))
                for j in range(k)
            ]
            constraints.append(Or(ciclos))

            s.add_assertion(And(constraints))

            if s.solve():
                model = s.get_model()
                j = next(j for j in range(k) if model.get_py_value(ciclos[j]))
                return [grelha_modelo(model, st) for st in states], j
    return None, None


@app.function
def repete_sequencia(N, seed, sequencia):
    g = estado_inicial_py(N, seed)
    for t, ((i, j), depois) in enumerate(sequencia, start=1):
        if regra_py(g, i, j) != depois or g[i][j] == depois:
            return False, f"passo {t}: a célula ({i},{j}) não passa a {depois} pela regra"
        g[i][j] = depois
    instaveis = [(i, j) for (i, j) in normais(N) if regra_py(g, i, j) != g[i][j]]
    if instaveis:
        return False, f"o estado final não é estável ({len(instaveis)} células mudariam)"
    return True, [(i, j) for (i, j) in normais(N) if g[i][j]]


@app.function
def tabela_validacao(N):
    linhas = []
    for tipo, k in (("sync", 20), ("async", 6)):
        traco = gera_traco(*fots(N, tipo), k + 1, verbose=False)
        ok, msg = valida_traco(traco, N, tipo)
        linhas.append(f"| {tipo} | {k} | {'✔ ok' if ok else '✘ falhou'} | {msg} |")
    return mo.md("| modo | k | validação | detalhe |\n|---|---|---|---|\n" + "\n".join(linhas))


@app.function
def prop1_sync(N):
    orb = orbita_sync(N, seed=0)
    K = orb["estados distintos"]
    traco = bmc_always(*fots(N, "sync"), sem_erro(N), K)
    tab = pd.DataFrame(
        [{"semente": sd, **orbita_sync(N, seed=sd)} for sd in range(10)]
    )
    if traco is None:
        veredito = (
            f"BMC com {K} estados ($d={K - 1}$ passos) sem contra-exemplo. "
            f"Como a órbita só tem {K} estados, "
            "isto **prova** que o estado de erro nunca é alcançado no modo síncrono, "
            "para esta semente."
        )
    else:
        veredito = f"Contra-exemplo no modo síncrono com $d={len(traco) - 1}$ passos."
    return mo.vstack([
        mo.md(
            "### Modo síncrono: prova por exaustão\n\n"
            f"Com a semente 0, a simulação repete um estado ao fim de {orb['estados distintos']} "
            f"estados (pré-período {orb['pre-período']}, período {orb['período']}). " + veredito
            + "\n\nA tabela repete a simulação para outras sementes (simulação exaustiva, "
            "pelo mesmo argumento do determinismo):"
        ),
        mo.ui.table(tab, selection=None),
    ])


@app.function
def prop1_inducao(N):
    linhas = []
    cti1 = None
    for k in (1, 2, 3):
        cti = passo_inducao(N, k, "sync")
        if k == 1:
            cti1 = cti
        linhas.append({
            "k": k,
            "passo da k-indução": "falha (há contra-exemplo à indução)" if cti else "provado",
        })
    texto = ""
    if cti1 is not None:
        vivas0 = [(i, j) for (i, j) in normais(N) if cti1[0][i][j]]
        vivas1 = [(i, j) for (i, j) in normais(N) if cti1[1][i][j]]
        texto = (
            f"\n\nContra-exemplo à indução com $k=1$: um estado com as células vivas {vivas0} "
            f"(cada uma tem uma só vizinha viva) que passa a ter {len(vivas1)} células vivas, "
            "ou seja, a um estado de erro. Este estado pode nem ser alcançável a partir "
            "do estado inicial, mas a indução não tem como o excluir."
        )
    return mo.vstack([
        mo.md("### Modo síncrono: tentativa de *k*-indução"),
        mo.ui.table(pd.DataFrame(linhas), selection=None),
        mo.md(
            "A propriedade **não é *k*-indutiva** para estes $k$: existem sequências de estados "
            "sem erro que acabam num erro (padrões que levam vários passos a morrer, por exemplo "
            "uma linha diagonal de células, que perde as pontas a cada passo). "
            "Fora do notebook testou-se até $k=12$ sem sucesso (e o passo já demora cerca de "
            "220 s), por isso a *k*-indução não resolve este caso. O que prova a propriedade "
            "no síncrono é a exaustão acima." + texto
        ),
    ])


@app.function
def prop1_async(N):
    traco = bmc_always(*fots(N, "async"), sem_erro(N), 13)
    if traco is None:
        return mo.md("BMC até $d=12$ sem contra-exemplo no modo assíncrono.")
    d = len(traco) - 1
    ok, msg = valida_traco(traco, N, "async")
    tab = pd.DataFrame([
        {"passo": passo, "célula": f"({cel[0]},{cel[1]})", "antes": antes, "depois": depois}
        for (passo, cel, antes, depois) in mudancas(traco, N)
    ])
    return mo.vstack([
        mo.md(
            "### Modo assíncrono: contra-exemplo\n\n"
            f"O BMC encontra o estado de erro com **$d={d}$ passos**. "
            "Em cada passo transita uma só célula, e é esta a ordem:"
        ),
        mo.ui.table(tab, selection=None),
        mo.md(
            f"O traço foi confirmado pelo simulador em Python ({msg}). "
            f"É o contra-exemplo **mínimo**: o estado inicial tem 9 células normais vivas "
            f"(o bloco $3\\times3$) e só uma muda por passo, por isso são precisos pelo "
            f"menos 9 passos."
        ),
    ])


@app.function
def prop2_sync(N):
    linhas = []
    for sd in range(10):
        pre, ciclo = ciclo_sync(N, seed=sd)
        mortas, vivas = classifica_celulas(ciclo, N)
        linhas.append({
            "semente": sd,
            "pré-período": pre,
            "período": len(ciclo),
            "sempre mortas": len(mortas),
            "sempre vivas": len(vivas),
            "variam": N * N - len(mortas) - len(vivas),
        })
    confirmacoes = []
    for sd in (5, 2):
        traco, j = bmc_gf(*fots(N, "sync", sd), alguma_constante(N), 20)
        mortas, vivas = classifica_celulas(traco[j:], N)
        confirmacoes.append(
            f"- semente {sd}: o BMC encontra um *lasso* com {len(traco)} estados e ciclo a "
            f"começar no estado {j} (período {len(traco) - j}); no ciclo há "
            f"{len(mortas)} células sempre mortas e {len(vivas)} sempre vivas."
        )
    return mo.vstack([
        mo.md(
            "### Modo síncrono\n\n"
            "O sistema é determinístico, por isso a simulação dá o ciclo exato de cada execução. "
            "A tabela classifica as células normais ao longo desse ciclo:"
        ),
        mo.ui.table(pd.DataFrame(linhas), selection=None),
        mo.md(
            "O resultado é o mesmo para todas as sementes: o sistema acaba num ciclo (muitas vezes "
            "um estado parado) em que quase todas as células são constantes. Para duas sementes "
            "com ciclo curto confirma-se com o `bmc_gf`, a partir do modelo SMT:\n\n"
            + "\n".join(confirmacoes)
        ),
    ])


@app.function
def prop2_async_sem_fairness(N):
    traco, j = bmc_gf(*fots(N, "async"), alguma_constante(N), 3)
    return mo.md(
        "### Modo assíncrono, sem *fairness*\n\n"
        f"O BMC encontra um contra-exemplo logo com {len(traco)} estado(s): o ciclo é o próprio "
        f"estado inicial (de {len(traco) - 1} para {j}). Como a célula escolhida pode ficar com "
        "o mesmo valor (*stuttering*), o sistema pode ficar parado para sempre, e todas as células "
        "são constantes. É um contra-exemplo trivial: só mostra que sem exigir *fairness* a "
        "propriedade não pode valer."
    )


@app.function
def prop2_async_com_fairness(N):
    sequencia = [
        ((2, 3), 0), ((2, 2), 0), ((1, 8), 1), ((2, 1), 0), ((1, 9), 1),
        ((4, 2), 1), ((3, 1), 0), ((4, 3), 1), ((1, 8), 0),
    ]
    ok, res = repete_sequencia(N, 0, sequencia)
    if ok:
        texto = (
            "A sequência de 9 passos abaixo foi encontrada pelo BMC (uma pesquisa de cerca de "
            "4 minutos, por isso aqui só se repete) e é verificada em Python: cada passo aplica a "
            "regra B3|S23 a uma célula e o estado final é **estável**, isto é, nenhuma célula "
            f"mudaria se fosse escolhida. Células vivas no estado final: {res}."
        )
    else:
        texto = f"A verificação da sequência falhou: {res}"
    return mo.vstack([
        mo.md(
            "### Modo assíncrono, com *fairness*\n\n"
            "*Fairness* exige que cada célula seja escolhida infinitas vezes no ciclo. O modelo "
            "não guarda a célula escolhida, por isso não se codifica diretamente. Usa-se um "
            "argumento: se é alcançável um estado estável, escolher todas as células em ciclo é "
            "uma execução justa em que o estado nunca muda, e todas as células ficam constantes."
        ),
        mo.md(texto),
        mo.ui.table(
            pd.DataFrame(
                [{"passo": p, "célula": f"({c[0]},{c[1]})", "novo valor": v}
                 for p, (c, v) in enumerate(sequencia, start=1)]
            ),
            selection=None,
        ),
    ])


@app.cell
def _():
    modo = mo.ui.dropdown(
        options=["sync", "async"], value="sync", label="Modo de transição"
    )
    modo
    return (modo,)


@app.cell
def _(modo):
    gera_traco(*fots(17, modo.value), 6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validação cruzada

    O modelo SMT pode ter um erro que não se vê a olho (um índice trocado, uma regra mal escrita). Para o apanhar usa-se uma **segunda implementação**, escrita de outra forma e sem SMT: um simulador em Python puro (`regra_py`, `passo_sync_py`). A função `valida_traco` pega num traço gerado pelo SMT e verifica cada passo:

    - as células iniciais ficam constantes e as finais mortas;
    - **síncrono:** o estado seguinte é exatamente o que o simulador calcula;
    - **assíncrono:** muda no máximo uma célula normal e, se mudar, o novo valor é o da regra B3|S23 (zero mudanças é um passo de *stuttering*).

    O validador foi testado com traços alterados à mão e deteta-os.
    """)
    return


@app.cell
def _():
    tabela_validacao(17)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Propriedade 1: nunca se alcança o estado de erro

    **O que se pede.** O estado de erro é aquele em que todas as células normais estão mortas. A propriedade é $G\,\neg\mathit{erro}$ ("é sempre verdade que não há erro"), uma propriedade de segurança (um invariante). Há duas respostas possíveis: uma **prova**, ou um **contra-exemplo** (um traço que chega ao erro).

    **Ferramentas.**

    - **BMC (*bounded model checking*).** Para $d=0,1,2,\dots,K$ pergunta ao solver se existe um traço $s_0,\dots,s_d$ com $\mathit{init}(s_0)\wedge\mathit{trans}(s_0,s_1)\wedge\dots\wedge\mathit{erro}(s_d)$. Se for `sat`, o primeiro $d$ dá um contra-exemplo mínimo. Se for `unsat` para todo o $d\le K$, só sabemos que **até $K$ passos** não há erro, o que não é uma prova para sempre.
    - ***k*-indução.** Prova para todos os tempos. Caso base: o BMC não encontra erro até $k-1$. Passo: a partir de um estado **qualquer** (não precisa de ser alcançável), se $k$ estados consecutivos não são erro então o seguinte também não é. Se o passo for `unsat`, o erro nunca é alcançado.
    - **Código da ficha 7.** O BMC é o `bmc_always` do exercício 4 (invariante verificado com traços de tamanho crescente, devolvendo um contra-exemplo mínimo) e a geração de traços é o `gera_traco` do exercício 3. A única diferença é que o `bmc_always` usa um só solver incremental, em vez de criar um novo por cada tamanho, porque com um solver novo por tamanho o BMC assíncrono demorou cerca de 105 s (até 10 estados) e o síncrono 85 s só até 60 estados, contra 7 s e 8 s (até 157 estados) com o incremental.
    - **Prova por exaustão (só no síncrono).** O sistema síncrono é determinístico: cada estado tem um único sucessor. Simulando, a sequência acaba por repetir um estado $s_n=s_m$, e a partir daí só se repetem estados já vistos. Os estados alcançáveis são então exatamente os da órbita, e um BMC com profundidade igual ao seu tamanho cobre-os a todos. Isso é uma prova.
    """)
    return


@app.cell
def _():
    prop1_sync(17)
    return


@app.cell
def _():
    prop1_inducao(17)
    return


@app.cell
def _():
    prop1_async(17)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusão da propriedade 1

    | modo | resultado | como se obteve |
    |---|---|---|
    | síncrono | o erro **nunca** é alcançado (semente 0) | exaustão: a órbita tem 157 estados e o BMC até $d=156$ não encontra erro |
    | assíncrono | o erro **é** alcançado, em 9 passos | contra-exemplo do BMC, confirmado por simulação |

    A diferença vem do tipo de transição. No síncrono há uma única evolução possível, e a partir deste estado inicial ela estabiliza sem passar pelo erro. No assíncrono o solver escolhe a ordem em que as células transitam, e existe uma ordem que mata as 9 células do bloco inicial, uma de cada vez. A propriedade vale então num modo e falha no outro, o que é um resultado válido para o enunciado ("provar ou obter contra-exemplos").

    *Limitação:* a prova do síncrono é para a semente 0 (o estado inicial depende das células iniciais aleatórias). A tabela mostra que as outras sementes testadas também não chegam ao erro, mas para provar o resultado para todas as atribuições iniciais seria preciso deixá-las como variáveis livres.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Propriedade 2: nenhuma célula normal está permanentemente morta ou viva

    **O que se pede.** Lê-se "permanentemente" como "a partir de certo instante, para sempre". Uma célula $c$ não está permanentemente morta nem permanentemente viva se fica viva infinitas vezes e morta infinitas vezes:

    $$\bigwedge_{c\in\mathcal{N}}\ G\,F\,c\ \wedge\ G\,F\,\neg c$$

    É uma propriedade de animação (*liveness*), e a sua negação diz que existe uma célula que, a partir de certo ponto, é constante.

    **Como é um contra-exemplo.** Como o espaço de estados é finito, toda a execução infinita acaba num ciclo. Um contra-exemplo é um *lasso*: um prefixo $s_0,\dots,s_{j-1}$ seguido de um ciclo $s_j,\dots,s_{k-1}$ (com transição de $s_{k-1}$ para $s_j$) em que alguma célula normal tem o mesmo valor em todos os estados do ciclo.

    **Código da ficha 7.** O `bmc_gf` é o `bmc_eventually` dos exercícios 6 e 7, com duas alterações. A ficha verifica $F\,\phi$, em que o contra-exemplo é um ciclo onde $\phi$ nunca vale desde o início; aqui o contra-exemplo só precisa de a célula ser constante **dentro do ciclo**, não no prefixo. E o ciclo pode ser uma auto-transição do último estado (a ficha só testa $j<k-1$), o que é preciso para estados que ficam parados.
    """)
    return


@app.cell
def _():
    prop2_sync(17)
    return


@app.cell
def _():
    prop2_async_sem_fairness(17)
    return


@app.cell
def _():
    prop2_async_com_fairness(17)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusão da propriedade 2

    | modo | resultado | como se obteve |
    |---|---|---|
    | síncrono | **não vale**: quase todas as células ficam constantes no ciclo | análise exata do ciclo (sementes 0 a 9) e *lasso* do BMC (sementes 5 e 2) |
    | assíncrono, sem *fairness* | **não vale**, de forma trivial | *lasso* com 1 estado (o sistema fica parado) |
    | assíncrono, com *fairness* | **não vale** | estado estável alcançável em 9 passos, onde nenhuma célula muda |

    A propriedade falha em todos os casos porque o sistema tende a estabilizar: as células longe da atividade ficam mortas para sempre, e as que pertencem a padrões estáveis ficam vivas para sempre. O enunciado aceita este resultado ("provar ou obter contra-exemplos").

    *Limitações:* os resultados do síncrono são para as sementes testadas. No assíncrono, o argumento da *fairness* usa um estado estável, e na variante estrita (a célula escolhida tem de mudar) um estado estável não tem sucessores, por isso esse argumento deixaria de valer e a análise teria de ser refeita.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Testes de tempo (relatório)

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