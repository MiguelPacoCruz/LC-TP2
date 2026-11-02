import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

@app.cell(hide_code=True)
def _():
    import marimo as mo
    return mo.md("ALÍNEA 1: SFOTS")

@app.cell
def _():
    from z3 import *
    n = 8
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

    def init(state, limitado=False):
        c = [state['pc'] == INIT,
             state['x'] == state['a'],
             state['y'] == state['b'],
             state['z'] == 0,
             state['a'] >= 0,                 
             state['b'] >= 0]
        if limitado:
            c += [ULT(state['a'], 2 ** (n // 2)),
                  ULT(state['b'], 2 ** (n // 2))]
        return And(c)

    def init_lim(state):
        return init(state, limitado=True)

    def erro(state):
        return state['pc'] == ERROR

    def trans(curr,prox):
        keep_a_b = And(prox['a'] == curr['a'], prox['b'] == curr['b'])
        t_init = And(curr['pc'] == INIT, prox['pc'] == SKIP, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        is_even = Extract(0,0,curr['y']) == 0
        t_skip_stop = And(curr['pc'] == SKIP, curr['y'] == 0, prox['pc'] == STOP, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_skip_left = And(curr['pc'] == SKIP, curr['y']!=0, is_even, prox['pc'] == LEFT, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_skip_right = And(curr['pc'] == SKIP, curr['y']!=0, Not(is_even), prox['pc'] == RIGHT, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        of_left = ULT(curr['x']<<1, curr['x'])
        t_left_err = And(curr['pc'] == LEFT, of_left, prox['pc'] == ERROR, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_left_ok = And(curr['pc']==LEFT, Not(of_left), prox['pc'] == SKIP, prox['x'] == curr['x'] << 1, prox['y'] == LShR(curr['y'], 1), prox['z'] == curr['z'])
        of_right = ULT(curr['z'] + curr['x'], curr['z'])
        t_right_err = And(curr['pc'] == RIGHT, of_right, prox['pc'] == ERROR, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_right_ok = And(curr['pc']==RIGHT, Not(of_right), prox['pc'] == SKIP, prox['x'] == curr['x'], prox['y'] == curr['y'] - 1, prox['z'] == curr['z'] + curr['x'])
        t_stop = And(curr['pc'] == STOP, prox['pc'] == STOP, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])
        t_err = And(curr['pc'] == ERROR, prox['pc'] == ERROR, prox['x'] == curr['x'], prox['y'] == curr['y'], prox['z'] == curr['z'])

        return And(keep_a_b, Or(t_init, t_skip_stop, t_skip_left, t_skip_right, t_left_err, t_left_ok, t_right_err, t_right_ok, t_stop, t_err))

    return ERROR, INIT, LEFT, RIGHT, SKIP, STOP, declare, erro, init, init_lim, n, trans

@app.cell
def _(declare, init, trans):
    from z3 import *
    def mostra(m, estados):
        for i, e in enumerate(estados):
            print('  ', i, {v: m.eval(e[v], model_completion=True) for v in e})
     
    def gera_traco(declare, init, trans, k, extra=None):
        s = Solver()
        estados = [declare(i) for i in range(k)]
        s.add(init(estados[0]))
        for i in range(k - 1):
            s.add(trans(estados[i], estados[i + 1]))
        if extra is not None:
            s.add(extra(estados[0]))
        if s.check() == sat:
            mostra(s.model(), estados)
        else:
            print('  não existe traço de tamanho', k)

    print('1) Traço para a=3, b=5 (esperado: STOP com z=15 no fim)')
    gera_traco(declare, init, trans, 12, extra=lambda st: And(st['a'] == 3, st['b'] == 5))
    return gera_traco, mostra

@app.cell(hide_code=True)
def _():
    import marimo as mo
    return mo.md("ALÍNEA 2: o invariante x*y + z = a*b")

@app.cell
def _(declare, init, trans):
    from z3 import *
    def kinduction_always(declare, init, trans, inv, k):
        s_base = Solver()
        states = [declare(i) for i in range(k)]
        s_base.add(init(states[0]))
        for i in range(k - 1):
            s_base.add(trans(states[i], states[i + 1]))
        
        s_base.add(Or([Not(inv(st)) for st in states]))
        if s_base.check() == sat:
            print(f"  Falha no caso base (k={k})")
            return False
        
        s_step = Solver()
        states = [declare(i) for i in range(k + 1)]
        s_step.add(And([trans(states[i], states[i + 1]) for i in range(k)]))
        s_step.add(And([inv(states[i]) for i in range(k)]))
        s_step.add(Not(inv(states[k])))
        
        if s_step.check() == sat:
            print(f"  Falha no passo indutivo (k={k})")
            return False
            
        print(f"  Propriedade válida por {k}-indução!")
        return True
     
    def inv_b(st):
        return st['x'] * st['y'] + st['z'] == st['a'] * st['b']
     
    print('\n2) x*y + z == a*b é invariante?')
    kinduction_always(declare, init, trans, inv_b, 2)
    return inv_b, kinduction_always

@app.cell(hide_code=True)
def _():
    import marimo as mo
    return mo.md("ALÍNEA 3: segurança com a, b < 2^(n/2)")

@app.cell
def _(declare, erro, init, mostra, trans):
    from z3 import *
    def nao_erro(st):
        return Not(erro(st))

    def bmc_always(declare, init, trans, inv, K):
        for k in range(1, K + 1):
            s = Solver()
            estados = [declare(i) for i in range(k)]
            s.add(init(estados[0]))
            for i in range(k - 1):
                s.add(trans(estados[i], estados[i + 1]))
            s.add(Not(inv(estados[k - 1])))
            if s.check() == sat:
                print('  contra-exemplo mínimo com', k, 'estados:')
                mostra(s.model(), estados)
                return False
        print('  sem contra-exemplo nos primeiros', K, 'estados')
        return True

    bmc_always(declare, init, trans, nao_erro, 12)
    return bmc_always, nao_erro

@app.cell
def _(declare, init_lim, kinduction_always, nao_erro, trans):
    from z3 import *
    print('\nDireta (sem reforço): \nCom a,b < 2^(n/2): pc != ERROR sozinho:')
    kinduction_always(declare, init_lim, trans, nao_erro, 32)
    return

@app.cell
def _(declare, init_lim, kinduction_always, n, nao_erro, trans):
    from z3 import *
    def ext(v):
        return ZeroExt(n, v)
     
    def inv_inteiro(st):
        return ext(st['x']) * ext(st['y']) + ext(st['z']) == ext(st['a']) * ext(st['b'])
     
    def limites(st):
        return And(ULT(st['a'], 2 ** (n // 2)), ULT(st['b'], 2 ** (n // 2)))
     
    def inv_seguro(st):
        return And(nao_erro(st), inv_inteiro(st), limites(st))

    print('\nReforçada com Invariante seguro: \nCom a,b < 2^(n/2)')
    kinduction_always(declare, init_lim, trans, inv_seguro, 2)
    return ext, inv_inteiro, inv_seguro, limites

@app.cell(hide_code=True)
def _():
    import marimo as mo
    return mo.md("o erro é inalcançável nos dois casos, e o reforço serve para encurtar muito a prova. Mesmo sem inv_nodos, o k=2 chega, porque o estado anterior (SKIP) já garante a paridade de y.")

@app.cell(hide_code=True)
def _():
    import marimo as mo
    return mo.md("ALÍNEA 4: terminação, F (pc = STOP  ou  pc = ERROR)")

@app.cell
def _(ERROR, SKIP, STOP, declare, trans):
    from z3 import *
    def k_lookahead_termination(declare, trans, V, k):
        s = Solver()
        states = [declare(i) for i in range(k + 1)]
        s.add(states[0]['pc'] == SKIP)          
        s.add(And([trans(states[i], states[i + 1]) for i in range(k)]))

        terminou = Or(states[-1]['pc'] == STOP, states[-1]['pc'] == ERROR)
        decresceu = ULT(V(states[-1]), V(states[0]))
        s.add(Not(Or(terminou, decresceu)))

        if s.check() != unsat:
            print(f"  Falha ao provar terminação (lookahead k={k})")
            return False
        print(f"  Terminação provada com lookahead de {k} passos!")
        return True

    def variante_y(state):
        return state['y']

    print('\nd) Verificação de terminação:')
    k_lookahead_termination(declare, trans, variante_y, 2)
    return k_lookahead_termination, variante_y


if __name__ == "__main__":
    app.run()