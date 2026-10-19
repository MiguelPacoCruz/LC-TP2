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


if __name__ == "__main__":
    app.run()
