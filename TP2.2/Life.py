import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from random import random, randint
    from z3 import (
        Solver, Bool, And, Or, Not, If, Sum,
        sat, is_true
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Primeiro vou começar por uma visualização do Conway's game of life para melhor entender o problema. Vamos efetuar visualizações e perceber melhor como é que seria este problema, antes de o modelar logicamente.
    """)
    return


@app.function
def initW(n: int):
    if not n > 16:
        raise ValueError("N must be greater than 16")
    matriz = [[0] * n for _ in range(n)]
    matriz = bW(matriz)
    matriz = cel_finW(matriz)
    matriz = cel_iniW(matriz)
    return matriz


@app.function
def bW(matriz):
    for i in range(4):
        for j in range(4):
            matriz[i][j] = 1
    return matriz


@app.function
def cel_finW(matriz):
    n = len(matriz)
    matriz.append([0] * n)
    
    for linha in matriz:
        linha.append(0)
        
    return matriz


@app.function
def cel_iniW(matriz):
    n = len(matriz)
    counter = 0
    for i in range(n):
        for j in range(n):
            if (i == 0) or (j == 0):
                counter += 1
                matriz[i][j] = randint(0, 1)
    print(f"Deviamos ter 2N + 1 = {(2*n + 1)} células inicias e temos {counter}")
    return matriz


@app.function
def display_matrixW(matriz):
    n = len(matriz)
    width = len(str(n - 1)) + 2

    header = " " * width + "".join(f"{j:>{width}}" for j in range(n))
    separator = ""
    rows = [
        f"{i:>{width-1}} " + "".join(f"{v:>{width}}" for v in row)
        for i, row in enumerate(matriz)
    ]

    return mo.md(
        f"```\n{header}\n{separator}\n" + "\n".join(rows) + "\n```"
    )


@app.cell
def _():
    display_matrixW(initW(20))
    return


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
    for i in range(1, N + 2):
        for j in range(1, N + 2):
            if i <= 3 and j <= 3:
                constraints.append(state[(i, j)])
            else:
                constraints.append(Not(state[(i, j)]))
    return And(constraints)


@app.function
def initial_final_cells(state, N):
    constraints = []

    for i in range(N + 1):
        constraints.append(
            state[(i, 0)] if random() < 0.5
            else Not(state[(i, 0)])
        )

    for j in range(1, N + 1):
        constraints.append(
            state[(0, j)] if random() < 0.5
            else Not(state[(0, j)])
        )

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


@app.cell
def _():

    def gera(declare, init, initial_final_cells, trans, N, k):
        with Solver() as s:
            states = [declare(t, N) for t in range(k + 1)]

            constraints = []

            constraints.append(init(states[0], N))
            constraints.append(initial_final_cells(states[0], N))

            for t in range(k):
                constraints.append(trans(states[t], states[t + 1], N))

            s.add(And(constraints))

            if s.check() == sat:
                model = s.model()

                for t, state in enumerate(states):
                    print(f"Estado {t}:")
                    for i in range(1, N + 1):
                        linha = [
                            1 if is_true(model.eval(state[(i, j)], model_completion=True)) else 0
                            for j in range(1, N + 1)
                        ]
                        print(linha)
                    print()
            else:
                print("Não foi encontrado um traço válido.")

    gera(declare, init, initial_final_cells, trans, 17, 5)
    return


if __name__ == "__main__":
    app.run()
