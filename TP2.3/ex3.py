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

        # 0. INIT para SKIP 
        t_init = And(curr['pc'] == INIT, prox['pc'] == SKIP,
                     prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])

        
        # 1. SKIP para LEFT, RIGHT ou STOP

        # Na representação binária de números, qualquer valor 
        # cujo último bit seja 0 é obrigatoriamente um número par (1 para impar)
        is_even = Extract(0,0,'y') == 0
        
                          # Pre-condiçao
        t_skip_stop = And(curr['pc'] == SKIP, curr['y'] == 0, 
                          # Atualização
                          prox['pc'] == STOP, 
                          #Manutenção
                          prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z']
                          )

        t_skip_left = And(curr['pc'] == SKIP, curr['y']!=0, is_even,
                          prox['pc'] == LEFT,
                          prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z']
                          )
        t_skip_right = And(curr['pc'] == SKIP, curr['y']!=0, Not(is_even), 
                           prox['pc'] == RIGHT,
                           prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z']
                           )
       
        # 2. Transições do LEFT
        
        # compara dois vetores de bits, se o primeiro é menor que o segundo,
        # caso se verifique é pq passou o tamanho do vetor 
        of_left = ULT(curr['x']<<1, curr['x'])

        t_left_err = And(curr['pc'] == LEFT, of_left,
                         prox['pc'] == ERROR,
                         prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z']
                         )
        t_left_ok = And(curr['pc']==LEFT, Not(of_left),
                        prox['pc'] == SKIP,
                        prox['x'] == curr['x'] << 1, prox['y'] == LShR(curr['y'], 1), prox['z'] == curr['z']
                        )

        # 3. Transições do RIGHT
        of_right = ULT(curr['z'] + curr['x'], curr['z'])

        t_right_err = And(curr['pc'] == RIGHT, of_right,
                         prox['pc'] == ERROR,
                         prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z']
                         )

        t_right_ok = And(curr['pc']==RIGHT, Not(of_right),
                        prox['pc'] == SKIP,
                        prox['x'] == curr['x'], prox['y'] == curr['y'] - 1, prox['z'] == curr['z'] + curr['x'] # Corrigido y = y - 1 para condizer com o habitual
                        )
        #Para garantir continuidade 
        t_stop = And(curr['pc'] == STOP, prox['pc'] == STOP,
                     prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_err = And(curr['pc'] == ERROR, prox['pc'] == ERROR,
                    prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])

        return And(keep_a_b, Or(t_init, t_skip_stop, t_skip_left, t_skip_right, 
                               t_left_err, t_left_ok, t_right_err, t_right_ok, t_stop, t_err))

    return ERROR, INIT, LEFT, RIGHT, SKIP, STOP, declare, init, n, trans


if __name__ == "__main__":
    app.run()