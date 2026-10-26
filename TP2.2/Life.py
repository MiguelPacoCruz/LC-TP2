import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    from random import random,randint
    import pysmt.shortcuts 
    from pysmt.typing import INT


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Primeiro vou começar por uma visualização do Conway's game of life para melhor entender o problema. Vamos efetuar visualizações e perceber melhor como é que seria este problema, antes de o modelar logicamente.
    """)
    return


@app.function
def init(n: int):
    if not n > 16:
        raise ValueError("N must be greater than 16")
    matriz = [[0] * n for _ in range(n)]
    matriz = b(matriz)
    matriz = cel_fin(matriz)
    matriz = cel_ini(matriz)
    return matriz


@app.function
def b(matriz):
    for i in range(4):
        for j in range(4):
            matriz[i][j] = 1
    return matriz


@app.function
def cel_fin(matriz):
    n = len(matriz)
    matriz.append([0] * n)
    
    for linha in matriz:
        linha.append(0)
        
    return matriz


@app.function
def cel_ini(matriz):
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
def display_matrix(matriz):
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
    display_matrix(init(20))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Com o núcleo do problema percebo, vamos passar à sua modelação.
    """)
    return


@app.cell
def _(Bool):
    def declare(t, N):
        return {
            (i, j): Bool(f'cell_{i}_{j}_{t}')
            for i in range(1, N + 1)
            for j in range(1, N + 1)
        }

    return (declare,)


@app.cell
def _(And, Not):
    def init(state, N):
        return And(
            # Células (i,j) com i,j <= 3 estão vivas
            *[
                state[(i, j)] if i <= 3 and j <= 3
                else Not(state[(i, j)])
                for i in range(1, N + 1)
                for j in range(1, N + 1)
            ]
        )

    return (init,)


@app.function
def initial_cells(N):
    return {
        (i, j): random() < 0.5
        for i in range(N + 1)
        for j in range(N + 1)
        if (i == 0 or j == 0)
    }


@app.cell
def _(And, Solver, declare, i, j, states, trans):
    def gera(declare,init,trans,k):

        with Solver(name="z3") as s:
            # completar
            matriz = [i for i in range(k+2)]

            constraints = []
        
            constraints.append(init(matriz,i,j))

            for i in range(k-1):
                constraints.append(trans(states[i],states[i+1]))

            s.add_assertion(And(constraints))

            if s.solve():
                model = s.get_model()
                #print(model)
                for i, st in enumerate(states):
                    pc_val = model.get_value(st['pc'])
                    x_val = model.get_value(st['x'])
                    print(f'Estado {i}: pc = {pc_val}, x = {x_val}')
            else:
                print("Não foi encontrado um traço válido")
                
    gera(declare,init,trans,20)
    return


if __name__ == "__main__":
    app.run()
