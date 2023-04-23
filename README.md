# naturalComputing

# 1 Exercício Computacional: otimização com restrição usando algoritmos evolutivos
Leandro Furlam Turi

## Métodos
### PSO
Utilizou-se o algoritmo PSO sugerido em aula, inicializado randomicamente dentro dos limites estabelecidos por cada exercício, e as velocidades sendo atualizadas através de uma constriction (omega), um componente histórico (v[i-1]), um componente cognitivo (phip relacionando os ótimos locais), um componente social (phig relacionando ótimo global), e respectivos valores randômicos (rp e rg). Ou seja, v[i+1] = omega * v[i] + phip * rp[i] * (local_best[i] - x[i]) + phig * rg[i] * (global_best - x[i])


### Algoritmo Genético
Novamente, utilizou-se o algoritmo sugerido em sala de aula, com a (1) seleção via torneio: seleciona-se dois indivíduos randomicamente, e o melhor é passado a frente, até completar toda a população; (2) crossover aritmético (best(beta*x_1 + (1 - beta)*x_2, beta*x_2 + (1 - beta)*x_1)) com probabilidade de crossover a nível de cromossomo selecionados via busca em grade, em cada problema; (3) mutação randômica linear, com a respectiva probabilidade a nível de cromossomo, em cada problema. Ainda, ressalta-se que houve inserção de elitismo durante o operador de seleção.


### Evolução Diferencial
Inicializou-se a população randomicamente dentro dos limites estabelecidos por cada exercício, e seguindo as estratégias do método, como mutação foi dada v[i+1] = x_r1[i] + F(x_r2[i] - x_r3[i]), com x_ri escolhidos randomicamente dentro da população, e F denominado fator de mutação. Acerca da recombinação, escolheu-se aleatoriamente os cromossomos dentre x_r1[i] e v[i+1] com uma probabilidade de recombinação (pcrossover). Caso este novo individuo recombinado seja melhor que o indivíduo original, este é passado a nova geração, caso contrário, o indivíduo original que será passado adiante.


### Estratégias Evolutivas
Inicializou-se a população randomicamente dentro dos limites estabelecidos por cada exercício, e seguindo as estratégias do método, como recombinação foi escolhida uma local intermediária, com dois pais selecionados aleatoriamente e z_i = (x_i + y_i) / 2 como mutação foi dada a não correlatada com apenas um sigma (tamanho do passo de mutação), ou seja, sigma_i = sigma*exp(tau * N(0, 1)) e x[i+1] = x[i] + sigma_i*N(0, 1), com N(0, 1) um valor aleatório dentro da distribuição normal (0, 1). Em seguinda, selecionou-se mu pais e lambda filhos, implementando a estratégia ES(mu + lambda).


### Restrições
Mais uma vez seguindo o sugerido em sala de aula, trabalhou-se com as restrições como sendo penalidades. Uma vez que eram todas de desigualdades e sendo f(x) a função objetivo a ser minimizada e h(x) a nova função, agora com a restrição inclusa:

h(x) = f(x) + r*sum(phi_i^2), sendo phi_i cada uma das restrições dadas.


## Resultados

Os valores dos hiperparâmetros utilizados neste relatório foram

* PSO
    - n 100
    - omega 0.5
    - phip 0.5
    - phig0.5

* GA
    - n 50
    - pcrossover 0.8
    - beta 0.5
    - pmutacao 0.05
    - elitismo 3

* DE (pcrossover, F, n)
    - n 50
    - omega 0.8
    - F 0.5

* ES
    - n 50
    - tau 0.15
    - step_size 0.5
    - mu 1/7
    - lambda 6/7

Acerca do critério de parada, em todos os métodos utilizou-se no máximo 10e5 iterações e tolerância entre as soluções de 1e-5.

Uma tabela contendo uma sumarização dos resultados truncadis na terceira casa decimal é apresentada a seguir:

Método | min | max | mean | std | median | iter (mean) | % soluções válidas
--- | --- | --- | --- | --- | --- | --- | --- |
PSO 1 |1082380.127 | 16505705.851 | 5973667.844 | 3710118.878 | 5402894.703 | 100000.0 | 0%* |
PSO 2 | 408.939 | 484.412 | 430.432 | 17.297 | 428.765 | 96669.866 | 10%* |
GA 1 | -4.762 | -0.734 | -2.833 | 1.022 | -2.681 | 32.633 | 100% |
GA 2 | 542.294 | 726.169 | 632.566 | 38.565 | 642.330 | 22.4 | 100% |
DE 1 | -5.707 | 0.528 | -2.522 | 1.597 | -2.457 | 36.866 | 100% |
DE 2 | 688.550 | 101267.045 | 9046.649 | 18941.844 | 2076.678 | 6.733 | 100% |
ES 1 | 34995.454 | 9086739.810 | 3092228.905 | 2447197.241 | 2697194.721 | 100000.0 | 0%* |
ES 2 | 474.032 | 3498.404 | 734.434 | 544.858 | 577.224 | 4.7 | 100% |

* finalizou via iteração


## Conclusão

Os métodos que melhor se adaptaram aos problemas propostos foram o GA e DE, conforme pode ser observado pelos resultados. Isto pode ser dado pelas estratégias adotadas, sendo um pouco mais sofisticadas que às adotadas pelo PSO e ES. Destaca-se ainda a baixa vairabilidade dos resultados apresentados pelo GA no primeiro problema e a baixa quantidade média de iterações. Ainda ressalta-se a não convergência dos métodos PSO e ES em alguns problemas, em que ambos finalizaram via iteração. Isto pode retratar má ajuste dos parâmetros dos métodos.
