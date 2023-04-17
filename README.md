# naturalComputing

# 1 Exercício Computacional: otimização com restrição usando algoritmos evolutivos
Leandro Furlam Turi

## Métodos
### PSO
Utilizou-se o algoritmo PSO sugerido em aula, inicializado randomicamente dentro dos limites estabelecidos por cada exercício, e as velocidades sendo atualizadas através de uma constriction (omega), um componente histórico (v[i-1]), um componente cognitivo (phip relacionando os ótimos locais), um componente social (phig relacionando ótimo global), e respectivos valores randômicos (rp e rg). Ou seja, 

v[i+1] = omega * v[i] + phip * rp[i] * (local_best[i] - x[i]) + phig * rg[i] * (global_best - x[i])

Acerca dos valores destes componentes, realizou-se uma busca em grade com cada componente variando linearmente com passo 0.1 no intervalo [0.1, 1.0]. O melhor conjunto de hiperparâmetros, ou seja, o conjunto onde os resultados discorrem são apresentados problema a problema.


### Algoritmo Genético
Novamente, utilizou-se o algoritmo sugerido em sala de aula, com a (1) seleção via torneio: seleciona-se dois indivíduos randomicamente, e o melhor é passado a frente, até completar toda a população; (2) crossover aritmético (best(beta*x_1 + (1 - beta)*x_2, beta*x_2 + (1 - beta)*x_1)) com os valores de beta ({0.1, 0.25, 0.5, 0.75, 1.0}) e probabilidade de crossover (passo 0.1 no intervalo [0.4, 0.9]) a nível de cromossomo selecionados via busca em grade, em cada problema; (3) mutação randômica linear, com a respectiva probabilidade ({0.05, 0.1, 0.2, 0.3, 0.4}) a nível de cromossomo também selecionada via busca em grade, em cada problema. Ainda, ressalta-se que houve inserção de elitismo durante o operador de seleção, com a quantidade de melhores indivíduos ({0, 1, 2, 3}) mantidos na população posterior também obtido via busca em grade, para cada problema.


### Evolução Diferencial
Loren Ipsum


### Restrições
Mais uma vez seguindo o sugerido em sala de aula, trabalhou-se com as restrições como sendo penalidades. Uma vez que eram todas de desigualdades e sendo f(x) a função objetivo a ser minimizada e h(x) a nova função, agora com a restrição inclusa:

h(x) = f(x) + r*sum(phi_i^2), sendo phi_i cada uma das restrições dadas.


## Resultados

Todos os valores aqui apresentados estão serializados e entregue junto a este relatório.

Os melhores parâmetros encontrados pela busca em grade foram:

* PSO (omega, phip, phig)
    - Problema 1: 0.9, 0.5, 0.3
    - Problema 2: 0.9, 0.2, 1.0

* GA (pcrossover, beta, pmutacao, elitismo)
    - Problema 1: 0.6, 0.1, 0.05, 3
    - Problema 2: 0.4, 0.1, 0.05, 3

* DE (pcrossover, F, n)
    - Problema 1: 0.6, 0.5, 50
    - Problema 2: 0.8, 0.25, 50

Já uma tabela contendo uma sumarização dos resultados truncadis na terceira casa decimal é apresentada a seguir:

Método | min | max | mean | std | median | iter (mean)
--- | --- | --- | --- | --- | --- | --- |
PSO 1 | 2196.898 | 35410.023 | 14094.097 | 6728.772 | 13956.503 | 913.100 |
PSO 2 | 29536.605 | 125903.831 | 58943.809 | 25644.614 | 57385.182 | 3083.300 |
GA 1 | 393314.289 | 13131701.605 | 2842632.348 | 2848197.291 | 1380058.870 | 20.600 |
GA 2 | 434620.914 | 27189330.203 | 4969779.815 | 6022810.286 | 2388376.125 | 18.666 |
DE 1 | 6425.766 | 12239106.618 | 2120357.687 | 2900925.053 | 826770.566 | 89.000
DE 1 | 176057.720 | 758323139.924 | 80846299.043 | 141184495.461 | 38376538.612 | 20.966

Note que embora o GA tenha uma quantidade menor de iterações (apresentando assim soluções com maior velocidade), este resultou em resultados mais variados, conforme observa-se pelo desvio e intervalo min-max.

Ainda, destaca-se que nenhum dos métodos conseguiu encontrar uma solução que fosse totalmente factível ao problema proposto, isto é, que respeitasse todas as restrições dadas. Mesmo com alguns métodos propostos pela literatura e tentativas de seeding (inserir ao menos uma solução factível) não foi possível encontrar tal indivíduo.


## Conclusão

Embora a implementação proposta do Algoritmo Genético tenha sido veloz, esta resultou em resultados com maior variabilidade, o que torna o método susceptível ao acaso. O PSO, por sua vez, demonstrou-se ser mais fiel à solução encontrada, mesmo que com maior quantidade de iterações.
