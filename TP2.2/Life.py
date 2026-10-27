import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from random import Random, randint
    from z3 import (
        Solver, Bool, And, Or, Not, If, Sum,
        sat, is_true
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Com o núcleo do problema percebo, vamos passar à sua modelação.
    """)
    return


@app.function
def declare(t, N):
    if not N > 16:
        raise ValueError("N must be greater than 16")
    return {
        (i, j): Bool(f'cell_{i}_{j}_{t}')
        for i in range(0, N + 2)
        for j in range(0, N + 2)
    }


@app.function
def init(state, N):
    constraints = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
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
    for i in range(N + 1):
        constraints.append(
            state[(i, 0)] if rng.random() < 0.5 else Not(state[(i, 0)])
        )
    for j in range(1, N + 1):
        constraints.append(
            state[(0, j)] if rng.random() < 0.5 else Not(state[(0, j)])
        )

    # 2N+3 células finais: sempre mortas
    for i in range(N + 2):
        constraints.append(Not(state[(i, N + 1)]))
    for j in range(N + 1):
        constraints.append(Not(state[(N + 1, j)]))

    return And(constraints)


@app.function
def trans(curr, prox, N):
    constraints = []

    # Células iniciais: mantêm o seu valor
    for i in range(N + 1):
        constraints.append(prox[(i, 0)] == curr[(i, 0)])

    for j in range(1, N + 1):
        constraints.append(prox[(0, j)] == curr[(0, j)])

    # Células finais: permanecem mortas
    for i in range(N + 2):
        constraints.append(Not(prox[(i, N + 1)]))

    for j in range(N + 1):
        constraints.append(Not(prox[(N + 1, j)]))

    # Regra do Game of Life para as células normais
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            vizinhos = Sum([
                If(curr[(i + di, j + dj)], 1, 0)
                for di in range(-1, 2)
                for dj in range(-1, 2)
                if di != 0 or dj != 0
            ])

            constraints.append(
                prox[(i, j)] == Or(
                    vizinhos == 3,
                    And(curr[(i, j)], vizinhos == 2)
                )
            )

    return And(constraints)


@app.function
def gera(N, k, seed=0, verbose=True):
    with Solver() as s:
        states = [declare(t, N) for t in range(k + 1)]

        s.add(init(states[0], N))
        s.add(initial_final_cells(states[0], N, seed))
        for t in range(k):
            s.add(trans(states[t], states[t + 1], N))

        if s.check() != sat:
            if verbose:
                print(f"Sem traço válido (N={N}, k={k}, seed={seed}).")
            return None

        model = s.model()
        traco = [
            [[1 if is_true(model.eval(st[(i, j)], model_completion=True)) else 0
              for j in range(1, N + 1)]
             for i in range(1, N + 1)]
            for st in states
        ]
        if verbose:
            print(f"N={N}, k={k}, seed={seed}")
            for t, grelha in enumerate(traco):
                print(f"Estado {t}:")
                for linha in grelha:
                    print(linha)
                print()
        return traco


@app.cell
def _():
    gera(N=17, k=5, seed=0)
    return


if __name__ == "__main__":
    app.run()
