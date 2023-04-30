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

método | f_min | f_max | f_avg | f_std | f_median | n_gen (avg) | % soluções válidas
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



# 2 Exercício Computacional: implementing a Fuzzy Classifier and Improving its Accuracy using Genetic Algorithms
## Métodos

Utilizou-se os algoritmos PSO e GA padrão da biblioteca `pymoo`, de forma a explorar as ferramentas existentes.

## Resultados
### PSO
W = 0.99123903, 0.54933287, 0.24488458, 0.41248142

F = 0.95333333


n_gen  |  f     |    S    |    w    |    c1    |    c2    |     f_avg     |     f_min    |
--- | --- | --- | --- | --- | --- | --- | --- | 
 1 |        - |       - |  0.9000 |  2.00000 |  2.00000 |  1.5970556954 |  1.0563380281 |
 2 |  0.38805 |       2 |  0.6465 |  2.04061 |  1.95939 |  1.3061308372 |  1.0563380281 |
 3 |  0.11847 |       3 |  0.4757 |  2.04010 |  1.97248 |  1.1362607856 |  1.0563380281 |
 4 |  0.02640 |       3 |  0.4166 |  2.03298 |  1.99184 |  1.0966906735 |  1.0489510488 |
 5 | -5.3E-03 |       3 |  0.3967 |  2.01998 |  2.00904 |  1.0807979114 |  1.0489510488 |
 6 | -2.9E-03 |       3 |  0.3982 |  2.00536 |  2.01506 |  1.0712190777 |  1.0489510488 |
 7 | -5.5E-05 |       3 |  0.4000 |  1.99523 |  2.02239 |  1.0667597188 |  1.0489510488 |
 8 |  0.00177 |       3 |  0.4011 |  1.98672 |  2.03535 |  1.0631396258 |  1.0489510488 |
 9 |  0.00439 |       3 |  0.4027 |  1.97613 |  2.04217 |  1.0606901232 |  1.0489510488 |
10 |  0.00216 |       3 |  0.4013 |  1.96780 |  2.05725 |  1.0587125513 |  1.0489510488 |
11 |  0.00284 |       3 |  0.4018 |  1.95656 |  2.07249 |  1.0573377888 |  1.0489510488 |
12 |  0.00144 |       3 |  0.4009 |  1.94352 |  2.08253 |  1.0556575297 |  1.0489510488 |
13 |  0.00152 |       3 |  0.4009 |  1.93182 |  2.08743 |  1.0543862978 |  1.0489510488 |
14 |  0.00026 |       3 |  0.4002 |  1.92421 |  2.10199 |  1.0539388878 |  1.0489510488 |
15 |  0.00227 |       3 |  0.4014 |  1.91309 |  2.10936 |  1.0532001899 |  1.0489510488 |
16 |  0.00227 |       3 |  0.4014 |  1.90400 |  2.11796 |  1.0532001899 |  1.0489510488 |
17 |  0.00194 |       3 |  0.4012 |  1.89583 |  2.13070 |  1.0521267191 |  1.0489510488 |
18 |  0.00175 |       3 |  0.4011 |  1.88525 |  2.13782 |  1.0518270038 |  1.0489510488 |
19 |  0.00129 |       3 |  0.4008 |  1.87649 |  2.14559 |  1.0516771686 |  1.0489510488 |
20 |  0.00242 |       3 |  0.4015 |  1.86813 |  2.15191 |  1.0502891324 |  1.0489510488 |
21 |  0.00262 |       3 |  0.4016 |  1.86158 |  2.16302 |  1.0496918424 |  1.0489510488 |
22 |  0.00229 |       3 |  0.4014 |  1.85375 |  2.17616 |  1.0495441028 |  1.0489510488 |
23 |  0.00229 |       3 |  0.4014 |  1.84294 |  2.18293 |  1.0495441028 |  1.0489510488 |
24 |  0.00229 |       3 |  0.4014 |  1.83360 |  2.18761 |  1.0495441028 |  1.0489510488 |
25 |  0.00230 |       3 |  0.4014 |  1.82690 |  2.19522 |  1.0493963632 |  1.0489510488 |
26 |  0.00230 |       3 |  0.4014 |  1.82080 |  2.20624 |  1.0493963632 |  1.0489510488 |
27 |  0.00230 |       3 |  0.4014 |  1.81143 |  2.21013 |  1.0493963632 |  1.0489510488 |
28 |  0.00230 |       3 |  0.4014 |  1.80588 |  2.22043 |  1.0493963632 |  1.0489510488 |
29 |  0.00230 |       3 |  0.4014 |  1.79723 |  2.22450 |  1.0493963632 |  1.0489510488 |
30 |  0.00264 |       3 |  0.4016 |  1.79160 |  2.23291 |  1.0492465280 |  1.0489510488 |
31 |  0.00264 |       3 |  0.4016 |  1.78531 |  2.24173 |  1.0492465280 |  1.0489510488 |
32 |  0.00264 |       3 |  0.4016 |  1.77636 |  2.24360 |  1.0492465280 |  1.0489510488 |
33 |  0.00225 |       3 |  0.4014 |  1.77114 |  2.24915 |  1.0490987884 |  1.0489510488 |
34 |  0.00225 |       3 |  0.4014 |  1.76644 |  2.25625 |  1.0490987884 |  1.0489510488 |

### GA
W = 0.00446343, 0.4301238, 0.09728874, 0.41168106

F = 0.96

n_gen  |     f_avg     |     f_min     | 
--- | --- | --- |
 1 | 1.6345743828 |  1.0714285713 |
 2 | 1.2278453536 |  1.0714285713 |
 3 | 1.1344640581 |  1.0714285713 |
 4 | 1.0991645541 |  1.0638297871 |
 5 | 1.0782945192 |  1.0563380281 |
 6 | 1.0716089048 |  1.0563380281 |
 7 | 1.0687036891 |  1.0489510488 |
 8 | 1.0665813807 |  1.0489510488 |
 9 | 1.0619685528 |  1.0489510488 |
10 | 1.0575513350 |  1.0416666666 |
11 | 1.0539064807 |  1.0416666666 |
12 | 1.0516944908 |  1.0416666666 |
13 | 1.0471299533 |  1.0416666666 |
14 | 1.0445804195 |  1.0416666666 |
15 | 1.0416666666 |  1.0416666666 |
16 | 1.0416666666 |  1.0416666666 |


## Conclusão
Notou-se que o GA obteve um resultado ligeriamente melhor quando comparado ao PSO, com o parâmetro w4 mantendo-se semelhante entre as duas técnicas (não haviam intersecções nas regras fuzzy relacionadas à este parâmetro), e observando maior variabilidade em w1.
