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

    MODO = "sync"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Modelo FSM
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Game of Life com espaço de estados finito

    O autómato é modelado como um *first-order transition system* (FOTS): um predicado `init` sobre o estado inicial e um predicado `trans` sobre o par (pré-estado, pós-estado). Primeiro formula-se a relação de transição em lógica; depois implementa-se em pySMT, com o modo de transição (síncrono ou assíncrono) escolhido pela constante `MODO`, definida no início do notebook. A estrutura segue a da ficha 7 (FOTS, `declare`/`init`/`trans`, `gera_traco` e `bmc_always`) e, na *k*-indução, a da ficha 8 (`kinduction_always`); os exercícios de origem são indicados ao longo do texto.

    ## Formulação lógica

    **Estado.** Cada célula é uma variável booleana $c_{i,j}$ (e $c'_{i,j}$ no pós-estado), com $0 \le i,j \le N+1$. Os três tipos de célula são:

    - normais: $\mathcal{N}=\{1..N\}^2$
    - iniciais: $\mathcal{I}=\{(i,j)\in\{0..N\}^2 : i=0 \vee j=0\}$, em número $2N+1$
    - finais: $\mathcal{F}=\{(i,j)\in\{0..N+1\}^2 : i=N+1 \vee j=N+1\}$, em número $2N+3$ (não há células iniciais e finais em comum)

    **Regra B3|S23.** Seja $V(i,j)$ o conjunto das 8 vizinhas de $(i,j)$ e $n(i,j)=\sum_{(u,v)\in V(i,j)} [c_{u,v}]$ o número de vizinhas vivas:

    $$R(i,j)\;\equiv\; n(i,j)=3 \;\vee\; \big(c_{i,j}\wedge n(i,j)=2\big)$$

    **Manutenção da fronteira** (igual nos dois modos):

    $$\mathit{Fr}\;\equiv\;\bigwedge_{(i,j)\in\mathcal{I}}\big(c'_{i,j}\leftrightarrow c_{i,j}\big)\;\wedge\;\bigwedge_{(i,j)\in\mathcal{F}}\neg c'_{i,j}$$

    **Modo síncrono** (todas as células normais transitam em simultâneo):

    $$T_{\mathit{sync}}\;\equiv\;\mathit{Fr}\;\wedge\;\bigwedge_{(i,j)\in\mathcal{N}}\big(c'_{i,j}\leftrightarrow R(i,j)\big)$$

    **Modo assíncrono** (uma e só uma célula normal transita). Cada disjunto é um caso, com a sua atualização e manutenção:

    $$T_{\mathit{async}}\;\equiv\;\mathit{Fr}\;\wedge\;\bigvee_{(a,b)\in\mathcal{N}}\Big(\underbrace{c'_{a,b}\leftrightarrow R(a,b)}_{\text{Atualização}}\;\wedge\;\underbrace{\bigwedge_{(i,j)\in\mathcal{N}\setminus\{(a,b)\}}\big(c'_{i,j}\leftrightarrow c_{i,j}\big)}_{\text{Manutenção}}\Big)$$

    A estrutura de cada caso (pré-condição, atualização, manutenção) é a do exercício 2 da ficha 7. A pré-condição é vazia: a célula escolhida pode ficar com o mesmo valor (*stuttering*). É o comportamento que o enunciado descreve, porque a regra B3|S23 inclui como resultados possíveis "mantém-se viva" e "continua morta"; escolher uma célula cuja regra dá o valor atual é, por isso, uma transição válida. A variante estrita acrescentaria a cada disjunto a pré-condição $R(a,b)\not\leftrightarrow c_{a,b}$.
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
def trans(curr, prox, N, modo=MODO):
    if modo == "sync":
        return trans_sync(curr, prox, N)
    if modo == "async":
        return trans_async(curr, prox, N)
    raise ValueError("modo deve ser 'sync' ou 'async'")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Geração de traços
    """)
    return


@app.function
def fots(N, modo=MODO, seed=0):
    def declare_(i):
        return declare(i, N)

    def init_(state):
        return And(init(state, N), initial_final_cells(state, N, seed))

    def trans_(curr, prox):
        return trans(curr, prox, N, modo)

    return declare_, init_, trans_


@app.function
def grelha_modelo(model, state):
    N = max(i for (i, _) in state) - 1
    return [
        [1 if model.get_py_value(state[(i, j)]) else 0 for j in range(N + 2)]
        for i in range(N + 2)
    ]


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Simulação
    """)
    return


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
def trajetoria_sync(N, seed=0, limite=1_000_000):
    g = estado_inicial_py(N, seed)
    visto = {tuple(map(tuple, g)): 0}
    estados = [g]
    while len(estados) < limite:
        g = passo_sync_py(g, N)
        chave = tuple(map(tuple, g))
        if chave in visto:
            return visto[chave], estados
        visto[chave] = len(estados)
        estados.append(g)
    return None, None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Verificação formal
    """)
    return


@app.function
def erro(state, N):
    return And([Not(state[c]) for c in normais(N)])


@app.function
def sem_erro(N):
    return lambda state: Not(erro(state, N))


@app.function
def invariante_inducao(N):
    return lambda state: And(
        Not(erro(state, N)),
        And([Not(state[c]) for c in finais(N)]),
    )


@app.function
def bmc_always(declare, init, trans, inv, K):
    for k in range(1, K + 1):
        with Solver(name="z3") as s:
            states = [declare(i) for i in range(k)]
            constraints = []
        
            constraints.append(init(states[0]))
        
            for i in range(k - 1):
                constraints.append(trans(states[i], states[i + 1]))
            
            constraints.append(Not(inv(states[-1])))
        
            s.add_assertion(And(constraints))

            if s.solve():
                model = s.get_model()
                return [grelha_modelo(model, st) for st in states]
    return None


@app.function
def kinduction_always(declare, init, trans, inv, k):

    with Solver(name="z3") as solver:

        states = [declare(i) for i in range(k)]
        base = [init(states[0])]

        for i in range(k - 1):
            base.append(trans(states[i], states[i + 1]))

        caso_base = And(And(base), Or([Not(inv(s)) for s in states]))
        solver.add_assertion(caso_base)
    
        if solver.solve():
            model = solver.get_model()
            return "caso base", [grelha_modelo(model, s) for s in states]

        solver.reset_assertions()

        states = [declare(i) for i in range(k + 1)]
        inductive = []

        for i in range(k):
            inductive.append(trans(states[i], states[i + 1]))

        inductive.append(And([inv(states[i]) for i in range(k)]))
        inductive.append(Not(inv(states[k])))

        step_case = And(inductive)
        solver.add_assertion(step_case)    

        if solver.solve():
            model = solver.get_model()
            return "passo indutivo", [grelha_modelo(model, s) for s in states]

    return None, None


@app.function
def alguma_constante(N):
    cel = normais(N)
    return lambda estados: Or([
        Or(And([st[c] for st in estados]), And([Not(st[c]) for st in estados]))
        for c in cel
    ])


@app.function
def bmc_eventually(declare, init, trans, constante, bound):
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Análise de propriedades
    """)
    return


@app.function
def orbita_sync(N, seed=0):
    inicio, estados = trajetoria_sync(N, seed)
    return {
        "pre-período": inicio,
        "período": len(estados) - inicio,
        "estados distintos": len(estados),
        "erro na órbita": any(
            all(g[i][j] == 0 for (i, j) in normais(N)) for g in estados
        ),
    }


@app.function
def ciclo_sync(N, seed=0):
    inicio, estados = trajetoria_sync(N, seed)
    return inicio, estados[inicio:]


@app.function
def classifica_celulas(ciclo, N):
    cel = normais(N)
    mortas = [(i, j) for (i, j) in cel if all(g[i][j] == 0 for g in ciclo)]
    vivas = [(i, j) for (i, j) in cel if all(g[i][j] == 1 for g in ciclo)]
    return mortas, vivas


@app.function
def mudancas(traco, N):
    return [
        (t + 1, (i, j), traco[t][i][j], traco[t + 1][i][j])
        for t in range(len(traco) - 1)
        for (i, j) in normais(N)
        if traco[t][i][j] != traco[t + 1][i][j]
    ]


@app.function
def prop1_sync(N):

    orb = orbita_sync(N, seed=0)

    K = orb["estados distintos"]

    traco = bmc_always(*fots(N, "sync"), sem_erro(N), K)

    tab = pd.DataFrame(

        [{"semente": sd, **orbita_sync(N, seed=sd)} for sd in range(10)]

    )

    if traco is None:

        veredito = f"O BMC não encontrou contra-exemplo até d={K - 1}. Sendo K o tamanho total da órbita, prova-se por exaustão que o estado de erro é inalcançável (para a semente 0)."

    else:

        veredito = f"Contra-exemplo no modo síncrono atingido em d={len(traco) - 1} passos."

    

    return mo.vstack([

        mo.md(f"### Modo síncrono (Prova por Exaustão)\n\n{veredito}\n\nA tabela demonstra a estabilização para outras sementes:"),

        mo.ui.table(tab, selection=None),

    ])


@app.function
def prop1_inducao(N):

    linhas = []

    cti1 = None

    for k in (1, 2, 3):

        onde, ce = kinduction_always(*fots(N, "sync"), invariante_inducao(N), k)

        if k == 1 and onde == "passo indutivo":

            cti1 = ce

        linhas.append({

            "k": k,

            "resultado": f"falha no {onde}" if onde else "válida por k-indução",

        })

    texto = ""

    if cti1 is not None:

        texto = "\n\nO invariante falha no passo indutivo. O solver encontra estados (mesmo que inalcançáveis a partir da origem) onde células vivas transitam todas para zero num só passo, mostrando que a propriedade não é k-indutiva para estes valores."

    

    return mo.vstack([

        mo.md("### Análise por k-indução"),

        mo.ui.table(pd.DataFrame(linhas), selection=None),

        mo.md(texto),

    ])


@app.function
def prop1_async(N):

    traco = bmc_always(*fots(N, "async"), sem_erro(N), 13)

    if traco is None:

        return mo.md("Sem contra-exemplo até d=12 (assíncrono).")



    d = len(traco) - 1

    tab = pd.DataFrame([

        {"passo": passo, "célula": f"({cel[0]},{cel[1]})", "antes": antes, "depois": depois}

        for (passo, cel, antes, depois) in mudancas(traco, N)

    ])



    return mo.vstack([

        mo.md(f"### Modo assíncrono (Contra-exemplo)\n\nA propriedade falha no modo assíncrono. O BMC gerou um traço de d={d} passos (contra-exemplo mínimo) onde o estado de erro é forçado, alterando uma célula por passo:"),

        mo.ui.table(tab, selection=None)

    ])


@app.function
def prop2_sync(N, sementes_bmc=(5, 2)):
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
        })

    bmc = []
    for sd in sementes_bmc:
        K = orbita_sync(N, seed=sd)["estados distintos"]
        traco, j = bmc_eventually(*fots(N, "sync", sd), alguma_constante(N), K)
        bmc.append({
            "semente": sd,
            "lasso encontrado": traco is not None,
            "estados no traço": len(traco) if traco else None,
            "ciclo começa no estado": j,
        })

    return mo.vstack([
        mo.md("### Modo síncrono\n\nA propriedade falha. O sistema tende a estabilizar, com a esmagadora maioria das células a fixarem-se num estado permanente (ver primeira tabela). A segunda tabela é o BMC de ciclo (`bmc_eventually`) para as sementes de órbita curta, onde a profundidade necessária é pequena."),
        mo.ui.table(pd.DataFrame(linhas), selection=None),
        mo.ui.table(pd.DataFrame(bmc), selection=None),
    ])


@app.function
def prop2_async(N):
    traco, j = bmc_eventually(*fots(N, "async"), alguma_constante(N), 3)
    if traco is None:
        texto = "O BMC não encontrou nenhum *lasso* até k=3."
    else:
        texto = (
            "A propriedade é trivialmente falsa, pois a relação de transição permite *stuttering*. "
            f"O BMC encontra um *lasso* com {len(traco)} estado(s), com o ciclo a começar no estado {j}: "
            "o sistema fica parado e todas as células ficam permanentes."
        )
    return mo.md(f"### Modo assíncrono\n\n{texto}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Visualização e interface
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Geração de traços

    Mostra um traço de 5 passos para $N=17$ (semente 0), gerado com `gera_traco(..., 6)`: tal como na ficha, o último argumento é o número de estados, ou seja, $k-1$ passos. O modo de transição começa com o valor de `MODO` e pode trocar-se aqui. Em cada estado, a linha de cima e a coluna da esquerda são as células iniciais (constantes), a linha de baixo e a coluna da direita são as finais (sempre mortas), e o bloco do meio são as $N\times N$ células normais.

    A animação da geração de traços foi gerada pelo [chatGPT](https://chatgpt.com/share/6ae8ded7-2bdc-83eb-b76e-d93f407b779a).
    """)
    return


@app.cell
def _():
    modo = mo.ui.dropdown(
        options=["sync", "async"], value=MODO, label="Modo de transição"
    )
    modo
    return (modo,)


@app.cell
def _(modo):
    # A geração do traço mantém-se independente da animação.
    # verbose=False evita imprimir todas as grelhas em texto.
    traco_anim = gera_traco(*fots(17, modo.value), 6, verbose=False)
    return (traco_anim,)


@app.cell
def _():
    # Estado reativo: índice da grelha atualmente apresentada.
    get_indice, set_indice = mo.state(0)
    return get_indice, set_indice


@app.cell
def _(set_indice):
    # O próprio controlo refresh inicia/pausa a reprodução.
    # O menu permite escolher o intervalo entre estados.
    reproducao = mo.ui.refresh(
        options=[0.1, 0.2, 0.5, 1.0],
        default_interval=None,
        label="Reproduzir / Pausar — intervalo",
    )
    reiniciar = mo.ui.button(
        label="Reiniciar",
        on_click=lambda _: set_indice(0),
    )
    mo.hstack([reproducao, reiniciar], justify="start", align="center")
    return (reproducao,)


@app.cell
def _(reproducao, set_indice, traco_anim):
    # Cada atualização do temporizador avança exatamente um estado.
    # O índice fica limitado ao último estado do traço.
    if reproducao.value and traco_anim:
        set_indice(
            lambda atual: min(atual + 1, len(traco_anim) - 1)
        )
    return


@app.cell
def _(get_indice, traco_anim):
    if not traco_anim:
        saida = mo.md("**Não foi encontrado um traço válido para visualizar.**")
    else:
        indice = min(get_indice(), len(traco_anim) - 1)
        grelha = traco_anim[indice]
        altura = len(grelha)
        largura = len(grelha[0])

        # A figura adapta-se às dimensões da grelha recebida.
        fig, ax = plt.subplots(
            figsize=(min(12, max(4, largura * 0.35)),
                     min(12, max(4, altura * 0.35)))
        )
        ax.imshow(grelha, cmap="binary", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(largura))
        ax.set_yticks(range(altura))
        ax.set_xticks([x - 0.5 for x in range(1, largura)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, altura)], minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, largura - 0.5)
        ax.set_ylim(altura - 0.5, -0.5)
        ax.set_title(f"Estado {indice + 1} / {len(traco_anim)}")
        ax.set_xlabel("Coluna")
        ax.set_ylabel("Linha")
        fig.tight_layout()

        saida = mo.vstack([
            mo.md(f"**Estado {indice + 1} de {len(traco_anim)}**"),
            fig,
        ])
        plt.close(fig)
    saida
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7. Execução e apresentação dos resultados
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Propriedade 1: Invariante de Segurança

    **Propriedade:** O estado de erro (todas as células normais mortas) nunca é alcançado. Formalmente: $G\,\neg\mathit{erro}$.

    **Estratégia de Verificação:**
    - **Modo síncrono:** Como o sistema é determinístico e finito, utiliza-se BMC iterativo para provar a propriedade por exaustão (cobrindo o tamanho total da órbita) e $k$-indução para tentar a verificação generalizada no tempo.
    - **Modo assíncrono:** Utiliza-se BMC convencional para procurar um contra-exemplo até um limite de profundidade $K$.

    Foi utilizado o [ChatGPT](https://chatgpt.com/share/6ae9fb56-18ec-83eb-b384-f83453ce1ced) para a opção de execução da prova formal síncrona por exaustão.
    """)
    return


@app.cell
def _():
    correr_teste = mo.ui.checkbox(value=False, label="Executar prova formal síncrona no z3 (Aviso: demora ~18 minutos)")
    correr_teste
    return (correr_teste,)


@app.cell
def _(correr_teste):
    if correr_teste.value:
        # Executa a prova pesada apenas se a checkbox for ativada
        resultado_prop1 = prop1_sync(17)
    else:
        # Resultados guardados (cache) correspondentes ao seu output
        tab_hardcoded = pd.DataFrame([
            {"semente": 0, "pre-período": 156, "período": 1, "estados distintos": 157, "erro na órbita": False},
            {"semente": 1, "pre-período": 51, "período": 7, "estados distintos": 58, "erro na órbita": False},
            {"semente": 2, "pre-período": 9, "período": 7, "estados distintos": 16, "erro na órbita": False},
            {"semente": 3, "pre-período": 120, "período": 1, "estados distintos": 121, "erro na órbita": False},
            {"semente": 4, "pre-período": 41, "período": 6, "estados distintos": 47, "erro na órbita": False},
            {"semente": 5, "pre-período": 6, "período": 1, "estados distintos": 7, "erro na órbita": False},
            {"semente": 6, "pre-período": 31, "período": 1, "estados distintos": 32, "erro na órbita": False},
            {"semente": 7, "pre-período": 1023, "período": 6, "estados distintos": 1029, "erro na órbita": False},
            {"semente": 8, "pre-período": 998, "período": 1, "estados distintos": 999, "erro na órbita": False},
            {"semente": 9, "pre-período": 774, "período": 1, "estados distintos": 775, "erro na órbita": False}
        ])

        veredito = "O BMC não encontrou contra-exemplo até d=156. Sendo K o tamanho total da órbita, prova-se por exaustão que o estado de erro é inalcançável (para a semente 0)."

        resultado_prop1 = mo.vstack([
            mo.md(f"### Modo síncrono (Prova por Exaustão)\n\n{veredito}\n\nA tabela demonstra a estabilização para outras sementes:\n\n*(Resultados em cache. Para executar a prova formal no z3, assinale a caixa acima)*"),
            mo.ui.table(tab_hardcoded, selection=None)
        ])

    resultado_prop1
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
    ## Propriedade 2: Propriedade de Animação

    **Propriedade:** Nenhuma célula normal permanece infinitamente viva ou morta. Formalmente: $\bigwedge_{c\in\mathcal{N}}\ (G\,F\,c\ \wedge\ G\,F\,\neg c)$. A falsificação desta propriedade implica encontrar um traço em ciclo (*lasso*) onde pelo menos uma célula se mantenha constante.

    **Estratégia de Verificação:** Utiliza-se a função `bmc_eventually` para gerar traços até profundidade $K$ terminados num ciclo válido, validando a estabilização das células.
    """)
    return


@app.cell
def _():
    prop2_sync(17)
    return


@app.cell
def _():
    prop2_async(17)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusão da propriedade 2

    | modo | resultado | como se obteve |
    |---|---|---|
    | síncrono | **não vale**: quase todas as células ficam constantes no ciclo | análise exata do ciclo (sementes 0 a 9) e *lasso* do BMC (sementes 5 e 2) |
    | assíncrono | **não vale**, de forma trivial | *lasso* com 1 estado do BMC (o sistema fica parado, graças ao *stuttering*) |

    A propriedade falha nos dois modos. No síncrono, o sistema tende a estabilizar: as células longe da atividade ficam mortas para sempre, e as que pertencem a padrões estáveis ficam vivas para sempre. No assíncrono, o *stuttering* permite um *lasso* de um só estado. O enunciado aceita este resultado ("provar ou obter contra-exemplos").

    *Limitações:* os resultados do síncrono são para as sementes testadas. Na variante estrita do assíncrono (a célula escolhida tem de mudar), o *lasso* de um estado deixa de existir e a análise teria de ser refeita.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nota sobre a Revisão de Código

    A totalidade do código deste projeto foi minuciosamente revista, testada e alterada em conformidade com as análises detalhadas no [histórico da conversa com o ChatGPT](https://chatgpt.com/share/6aea0102-4e48-83eb-9826-5dad2803e8fb). Todas as propostas de otimização e verificações de propriedades foram validadas e adaptadas às necessidades do modelo FOTS.
    """)
    return


if __name__ == "__main__":
    app.run()
