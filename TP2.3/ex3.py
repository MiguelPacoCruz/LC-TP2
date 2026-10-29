import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Considere o problema descrito no documento [Lógica Computacional: Multiplicação de Inteiros](https://paper.dropbox.com/doc/n1G7pMihg2yJrMswfpBxr). Nesse documento usa-se um “Control Flow Automaton” como modelo do programa imperativo que calcula a multiplicação de inteiros positivos representados por vetores de bits.

    1. Construir um FOTS, usando BitVec’s de tamanho \(n\), que descreva o comportamento deste autómato; para isso, identifique e codifique em `Z3` as variáveis do modelo, o estado inicial, a relação de transição e o estado de erro.

    2. Usando \(k\)-indução, verifique nesse SFOTS se a propriedade
       \[
       (x * y + z = a * b)
       \]
       é um invariante do seu comportamento.

    3. Usando \(k\)-indução no FOTS acima e adicionando ao estado inicial a condição
       \[
       (a < 2^{n/2}) \land (b < 2^{n/2}),
       \]
       verifique a segurança do programa; nomeadamente, prove que, com tal estado inicial, o estado de erro nunca é acessível.

    4. Codifique a propriedade de animação que assegura que o programa termina sempre, quaisquer que sejam os valores iniciais de \(a\) e \(b\).
    """)
    return


@app.cell
def _():
    from z3 import *

    n = 16
    INIT, SKIP, LEFT, RIGHT, STOP, ERROR = 0, 1, 2, 3, 4, 5

    def declare(i):
        state = {}
        state['pc'] = Int(f'pc_{i}')
        state['a'] = BitVec(f'a_{i}', n)
        state['b'] = BitVec(f'b_{i}', n)
        state['x'] = BitVec(f'x_{i}', n)
        state['y'] = BitVec(f'y_{i}', n)
        state['z'] = BitVec(f'z_{i}', n)
        return state

    #decidi passar o valores de x y e z ja diretamente na funçao
    def init(state):
        return And(
            state['pc'] == INIT, 
            state['x'] == state['a'], 
            state['y'] == state['b'], 
            state['z'] == 0,
            state['a'] >= 0, 
            state['b'] >= 0
        )

    def trans(curr,prox):
        #entradas a e b nunca mudam
        keep_a_b = And(prox['a'] == curr['a'], prox['b'] == curr['b'])

        # 0. INIT para SKIP (Mantém os valores porque o init já os carregou)
        t_init = And(curr['pc'] == INIT, prox['pc'] == SKIP,
                     prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        
        # Na representação binária de números, qualquer valor 
        # cujo último bit seja 0 é obrigatoriamente um número par (1 para impar)
        is_even = Extract(0,0,'y') == 0
        
        return keep_a_b, t_init, is_even

    return ERROR, INIT, LEFT, RIGHT, SKIP, STOP, declare, init, n, trans


if __name__ == "__main__":
    app.run()