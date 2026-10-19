import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Este exercício é um problema de optimização não linear cónica que deve ser resolvido com o “solver” `Clarabel` e a interface `cvxpy`. Note que o `Clarabel` não lida com variáveis inteiras.

    Em criptografia pós-quântica é importante um tipo de subconjuntos do espaço métrico $\mathbb{R}^m$ designados por reticulados euclidianos. Cada espaço é determinado por uma base de $k$ vetores:

    $$
    \mathsf{G} \equiv \{g_1, g_2, \cdots, g_k \mid g_i \in \mathbb{R}^m\}
    $$

    linearmente independentes. O reticulado determinado por $\mathsf{G}$ é o conjunto de todas as combinações lineares inteiras dos vetores $g_i$. Isto é,

    $$
    \mathcal{L}_{\mathsf{G}} \equiv \{x = z_1 g_1 + z_2 g_2 + \cdots + z_\kappa g_\kappa \mid z_i \in \mathbb{Z}\}
    $$

    O problema difícil que está na base de muitas das técnicas criptográficas pós-quânticas designa-se por _Closest Vector Problem_ (CVP) e define-se:

    >Dado um vetor aleatório $t \in \mathbb{R}^m$ (o “target”) e um reticulado determinado pela base $\mathsf{G}$, calcular o vetor do reticulado
    $$
    x \in \mathcal{L}_{\mathsf{G}} = \sum_{i=1}^{\kappa} z_i g_i
    $$
    que está mais próximo de $t$.

    Neste exercício vamos considerar uma versão particular deste problema, caracterizada pelas seguintes restrições:

    - A proximidade define-se pela distância Euclidiana no espaço $\mathbb{R}^m$; isto é,
      $$
      \mathsf{dist}(x,t) \equiv \sqrt{\sum_{j=1}^{m}(x_j-t_j)^2}
      $$
      sendo $x_j$ a $j$-ésima componente do vetor $x$.

    - Os elementos $\{g_{i,j}\}$ do vetor $g_i$, assim como os elementos $t_j$ do “target” $t$, representam probabilidades; por isso, são sempre valores reais no intervalo
      $$
      0 \leq g_{i,j},\,t_j \leq 1
      $$
      que, aqui, são gerados aleatoriamente.

    - Existe um limite $\ell$ positivo não nulo nos coeficientes inteiros $z_i$, tal que, para todo $i = 1, \ldots, \kappa$, se verifica
      $$
      -\ell \leq z_i \leq \ell
      $$

    - É necessário verificar se os $\{g_i\}_{i=1}^{k}$ formam realmente uma base.

    Nestas circunstâncias, pretende-se usar programação cónica para resolver o CVP.

    > Em problemas CVP criptograficamente seguros, os valores típicos de $m$ e $k$ são da ordem das centenas. Aqui vamos considerar uma situação em que $m \sim 30$, $k \sim 10$ e $\ell = 1, \cdots, 5$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Justificação da escolha do exercício

    Optámos pelo Exercício 2 (CVP via programação cónica) por três razões.

    ### 1. Continuidade com a aula TP de 19/10

    Nessa aula, lecionada pelo professor Bruno, foi resolvido um problema de otimização cónica de portfólios com `cvxpy` e o solver `Clarabel`. O risco foi modelado como pertença a um cone de segunda ordem (`cp.SOC`), através de uma variável epigráfica $t \geq \|\Sigma^{1/2} w\|_2$. Foram ainda trabalhados a reutilização de um problema com `cp.Parameter` e a leitura das variáveis duais.

    O CVP tem a mesma estrutura de base: a distância Euclidiana $\|Gz - t\|_2$ é, tal como o risco no portfólio, uma norma que se escreve diretamente como restrição cónica. Isto permitiu-nos partir de uma formulação já conhecida.

    ### 2. Ligação ao problema inteiro tratado no fim da aula

    A secção final do notebook introduziu restrições de cardinalidade com variáveis booleanas, resolvidas por *branch-and-bound* sobre a relaxação SOCP contínua em cada nó, com o `SCIP` como solver.

    O Exercício 2 coloca um problema análogo: a condição $z_i \in \mathbb{Z}$ torna o problema difícil e, como o `Clarabel` não suporta variáveis inteiras, a integralidade tem de ser tratada por nós, resolvendo a relaxação contínua em cada nó. O exercício permite, assim, aprofundar uma ideia que na aula foi apenas apresentada com uma demonstração em ponto pequeno.

    ### 3. Objetividade na validação

    A condição de base reduz-se a $\operatorname{rank}(G) = k$ e, para valores pequenos de $k$ e $\ell$, a solução pode ser comparada com pesquisa exaustiva. Isto permite validar a implementação e analisar o desempenho (nós explorados, tempo de execução, efeito de $\ell$ e de $k$).

    ### Porque não o Exercício 1

    A segunda propriedade do Exercício 1 (nenhuma célula permanentemente morta ou viva) é uma propriedade de vivacidade, que exige raciocínio sobre ciclos e, no caso assíncrono, hipóteses de justiça. Somado à necessidade de modelar duas semânticas de transição distintas, isto tornava o âmbito do trabalho mais incerto.
    """)
    return


if __name__ == "__main__":
    app.run()
