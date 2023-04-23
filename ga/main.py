import numpy as np
# import matplotlib.pyplot as plt


###############################################################################
# GA ##########################################################################
###############################################################################


def selecao(x, elitismo, fun): # torneio
    xnew = []
    x_sorted = sorted(x, key=fun)
    for i in range(elitismo):
        xnew.append(x_sorted[i])
    while len(xnew) < n:
        idxs = np.random.randint(0, n, 2)
        if fun(x[idxs[0]]) < fun(x[idxs[1]]):
            xnew.append(x[idxs[0]])
        else:
            xnew.append(x[idxs[1]])
    return xnew


def crossover(x, pcrossover, beta): # aritmetico
    xnew = []
    while len(xnew) < 2*n:
        idxs = np.random.randint(0, n, 2)
        c1 = beta*x[idxs[0]] + (1 - beta)*x[idxs[1]]
        c1 = [c1[i] if (np.random.random(1)[0] < pcrossover) and (c1[i] >= xmin[i]) and (c1[i] <= xmax[i]) 
              else x[idxs[0]][i] for i in range(len(c1))]
        c2 = beta*x[idxs[1]] + (1 - beta)*x[idxs[0]]
        c1 = [c2[i] if (np.random.random(1)[0] < pcrossover) and (c2[i] >= xmin[i]) and (c2[i] <= xmax[i]) 
              else x[idxs[1]][i] for i in range(len(c2))]
        xnew.append(np.array(c1))
        xnew.append(np.array(c2))
    return xnew


def mutacao(x, pmutacao):
    for _ in range(n):
        idx = np.random.randint(0, n, 1)[0]
        x[idx] = np.array([x[idx][i] if np.random.random(1)[0] > pmutacao else np.random.random(1)[0] for i in range(len(x[idx]))])


def ga(pcrossover, beta, pmutacao, elitismo, fun, fun_penalizacao):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([fun(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    while (c > eps) and (count < max_iter):
        x = crossover(x, pcrossover, beta)
        mutacao(x, pmutacao)
        x = selecao(x, elitismo, fun)
        f_x = [fun(x_i) for x_i in x]
        if np.isnan(f_x).all():
            break
        new_global_best = x[np.nanargmin(f_x)]
        c = sum(abs(new_global_best - global_best))
        global_best = new_global_best
        if fun_penalizacao(global_best) > 0:
            c = c + r
        solutions.append(fun(global_best))
        color_plot.append('red' if fun_penalizacao(global_best) > 0 else 'green')
        count = count + 1

    print(count, fun(global_best), fun_penalizacao(global_best) < eps, all(global_best >= xmin) and all(global_best <= xmax))

    # plt.scatter(range(count), solutions, c=color_plot)
    # plt.show()

    return (count, global_best, fun(global_best))


###############################################################################
# problema 1 ##################################################################
###############################################################################

# parametros
n = 50
xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
pcrossover = 0.8
beta = 0.5
pmutacao = 0.05
elitismo = 3
r = 1000 # fator de penalizacao
eps = 1e-5
max_iter = 10000

# penalizacao
def penalizacao1(x):
    phi1 = max(0, (2*x[0] + 2*x[1] + x[9] + x[10] - 10))
    phi2 = max(0, (2*x[0] + 2*x[2] + x[9] + x[11] - 10))
    phi3 = max(0, (2*x[1] + 2*x[2] + x[10] + x[11] - 10))
    phi4 = max(0, (-8*x[0] + x[9]))
    phi5 = max(0, (-8*x[1] + x[10]))
    phi6 = max(0, (-8*x[2] + x[11]))
    phi7 = max(0, (-2*x[3] - x[4] + x[9]))
    phi8 = max(0, (-2*x[5] - x[6] + x[10]))
    phi9 = max(0, (-2*x[7] - x[8] + x[11]))
    return r*(phi1*phi1 + phi2*phi2 + phi3*phi3 + phi4*phi4 + phi5*phi5 + phi6*phi6 + phi7*phi7 + phi8*phi8 + phi9*phi9)

# funcao para minimizacao
def f1(x):
    return (5*x[0] + 5*x[1] + 5*x[2] + 5*x[3] +
           - 5*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]) + 
           - x[4]- x[5] - x[6] - x[7] - x[8]- x[9]- x[11] - x[12] +
           penalizacao1(x))

bests = [ga(pcrossover, beta, pmutacao, elitismo, f1, penalizacao1) for _ in range(30)]
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))


###############################################################################
# problema 2 ##################################################################
###############################################################################


# parametros
n = 50
xmin = np.array([-10, -10, -10, -10, -10, -10, -10])
xmax = np.array([10, 10, 10, 10, 10, 10, 10])
pcrossover = 0.8
beta = 0.5
pmutacao = 0.05
elitismo = 3
r = 1000 # fator de penalizacao
eps = 1e-5
max_iter = 10000

# penalizacao
def penalizacao2(x):
    phi1 = max(0, -127 + 2*x[0]*x[0] + 3*x[1]*x[1]*x[1]*x[1] + x[2] + 4*x[3]*x[3] + 5*x[4])
    phi2 = max(0, -282 + 7*x[0] + 3*x[1] + 10*x[2]*x[2] + x[3] - x[4])
    phi3 = max(0, -196 + 23*x[0] + x[1]*x[1] + 6*x[5]*x[5] - 8*x[6])
    phi4 = max(0, 4*x[0]*x[0] + x[1]*x[1] - 3*x[0]*x[1] + 2*x[2]*x[2] + 5*x[5] - 11*x[6])
    return r*(phi1*phi1 + phi2*phi2 + phi3*phi3 + phi4*phi4)

# funcao para minimizacao
def f2(x):
    return ((x[0] - 10)*(x[0] - 10) + 5*(x[1] - 12)*(x[1] - 12) + x[2]*x[2]*x[2]*x[2] +
            3*(x[3] - 11)*x[3] - 11 + 10*x[5]*x[5]*x[5]*x[5]*x[5]*x[5] +
            7*x[5]*x[5] + x[6]*x[6]*x[6]*x[6] - 4*x[5]*x[6] - 10*x[5] - 8*x[6] +
           penalizacao2(x))

bests = [ga(pcrossover, beta, pmutacao, elitismo, f2, penalizacao2) for _ in range(30)]
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))
