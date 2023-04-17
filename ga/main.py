# import pickle
import numpy as np
# import matplotlib.pyplot as plt


def selecao(x): # torneio
    xnew = []
    x_sorted = sorted(x, key=f)
    for i in range(elitismo):
        xnew.append(x_sorted[i])
    while len(xnew) < n:
        idxs = np.random.randint(0, n, 2)
        if f(x[idxs[0]]) < f(x[idxs[1]]):
            xnew.append(x[idxs[0]])
        else:
            xnew.append(x[idxs[1]])
    return xnew


def crossover(x): # aritmetico
    xnew = []
    while len(xnew) < n:
        idxs = np.random.randint(0, n, 2)
        c1 = beta*x[idxs[0]] + (1 - beta)*x[idxs[1]]
        c1 = [c1[i] if np.random.random(1) < pcrossover else x[idxs[0]][i] for i in range(len(c1))]
        c2 = beta*x[idxs[1]] + (1 - beta)*x[idxs[0]]
        c1 = [c2[i] if np.random.random(1) < pcrossover else x[idxs[1]][i] for i in range(len(c2))]
        if f(c1) < f(c2):
            xnew.append(np.array(c1))
        else:
            xnew.append(np.array(c2))
    return xnew


def mutacao(x):
    xnew = []
    while len(xnew) < n:
        idx = np.random.randint(0, n, 1)
        c = [x[idx[0]][i] if np.random.random(1) > pmutacao else np.random.random(1)[0] for i in range(len(x[idx[0]]))]
        xnew.append(np.array(c))
    return xnew


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

np.random.seed(42)


# penalizacao
def penalizacao(x):
    return r*(
        max(0, (2*x[0] + 2*x[1] + x[9] + x[10] - 10)*(2*x[0] + 2*x[1] + x[9] + x[10] - 10)) +
        max(0, (2*x[0] + 2*x[2] + x[9] + x[11] - 10)*(2*x[0] + 2*x[2] + x[9] + x[11] - 10)) +
        max(0, (2*x[1] + 2*x[2] + x[10] + x[11] - 10)*(2*x[1] + 2*x[2] + x[10] + x[11] - 10)) +
        max(0, (-8*x[0] + x[9])*(-8*x[0] + x[9])) +
        max(0, (-8*x[1] + x[10])*(-8*x[1] + x[10])) +
        max(0, (-8*x[2] + x[11])*(-8*x[2] + x[11])) +
        max(0, (-2*x[3] - x[4] + x[9])*(-2*x[3] - x[4] + x[9])) +
        max(0, (-2*x[5] - x[6] + x[10])*(-2*x[5] - x[6] + x[10])) +
        max(0, (-2*x[7] - x[8] + x[11])*(-2*x[7] - x[8] + x[11]))
    )


# funcao para minimizacao
def f(x):
    return (5*x[0] + 5*x[1] + 5*x[2] + 5*x[3] +
           - 5*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]) + 
           - x[4]- x[5] - x[6] - x[7] - x[8]- x[9]- x[11] - x[12] +
           penalizacao(x))


def ga_1(pcrossover, beta, pmutacao, elitismo):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    x = crossover(x)
    x = mutacao(x)
    while (c > eps) and (count < max_iter):
        x = selecao(x)
        x = crossover(x)
        x = mutacao(x)
        f_x = [f(x_i) for x_i in x]
        if np.isnan(f_x).all():
            break
        new_global_best = x[np.nanargmin(f_x)]
        c = sum(abs(new_global_best - global_best))
        global_best = new_global_best
        solutions.append(f(global_best) - penalizacao(global_best))
        color_plot.append('red' if penalizacao(global_best) > 0 else 'green')
        count = count + 1

    print(count, f(global_best) - penalizacao(global_best), f(global_best))

    # plt.scatter(range(count), solutions, c=color_plot)
    # plt.show()

    return (count, global_best, f(global_best), pcrossover, beta, pmutacao, elitismo, solutions, color_plot)


# R = []
# for pcrossover in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     for beta in[0.1, 0.25, 0.5, 0.75, 1.0]:
#         for pmutacao in [0.05, 0.1, 0.2, 0.3, 0.4]:
#             for elitismo in [0, 1, 2, 3]:
#                 R.append(ga_1(pcrossover, beta, pmutacao, elitismo))

# file = open('ga1', 'wb')
# pickle.dump(R, file)
# file.close()

# file = open('ga1', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# pcrossover, beta, pmutacao, elitismo = best[3], best[4], best[5], best[6]
pcrossover, beta, pmutacao, elitismo = 0.6, 0.1, 0.05, 3
print(pcrossover, beta, pmutacao, elitismo)
bests = [ga_1(pcrossover, beta, pmutacao, elitismo) for _ in range(30)]
# file = open('ga11', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('ga11', 'rb')
# bests = pickle.load(file)
# file.close()
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))


###############################################################################
# problema 2 ##################################################################
###############################################################################


# parametros
n = 50
xmin = np.array([-10, -10, -10, -10, -10, -10, -10])
xmax = np.array([10, 10, 10, 10, 10, 10, 10, ])
pcrossover = 0.8
beta = 0.5
pmutacao = 0.05
elitismo = 3
r = 1000 # fator de penalizacao
eps = 1e-5
max_iter = 10000

np.random.seed(42)


# penalizacao
def penalizacao(x):
    return r*(
        max(0, (-127 + 2*x[0]*x[0] + 3*x[1]*x[1]*x[1]*x[1] + x[2] + 4*x[3]*x[3] + 5*x[4])*(-127 + 2*x[0]*x[0] + 3*x[1]*x[1]*x[1]*x[1] + x[2] + 4*x[3]*x[3] + 5*x[4])) +
        max(0, (-282 + 7*x[0] + 3*x[1] + 10*x[2]*x[2] + x[3] - x[4])*(-282 + 7*x[0] + 3*x[1] + 10*x[2]*x[2] + x[3] - x[4])) +
        max(0, (-196 + 23*x[0] + x[1]*x[1] + 6*x[5]*x[5] - 8*x[6])*-196 + 23*x[0] + x[1]*x[1] + 6*x[5]*x[5] - 8*x[6]) +
        max(0, (4*x[0]*x[0] + x[1]*x[1] - 3*x[0]*x[1] + 2*x[2]*x[2] + 5*x[5] - 11*x[6])*(4*x[0]*x[0] + x[1]*x[1] - 3*x[0]*x[1] + 2*x[2]*x[2] + 5*x[5] - 11*x[6]))
    )


# funcao para minimizacao
def f(x):
    return ((x[0] - 10)*(x[0] - 10) + 5*(x[1] - 12)*(x[1] - 12) + x[2]*x[2]*x[2]*x[2] +
            3*(x[3] - 11)*x[3] - 11 + 10*x[5]*x[5]*x[5]*x[5]*x[5]*x[5] +
            7*x[5]*x[5] + x[6]*x[6]*x[6]*x[6] - 4*x[5]*x[6] - 10*x[5] - 8*x[6] +
           penalizacao(x))


def ga_2(pcrossover, beta, pmutacao, elitismo):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    x = crossover(x)
    x = mutacao(x)
    while (c > eps) and (count < max_iter):
        x = selecao(x)
        x = crossover(x)
        x = mutacao(x)
        f_x = [f(x_i) for x_i in x]
        if np.isnan(f_x).all():
            break
        new_global_best = x[np.nanargmin(f_x)]
        c = sum(abs(new_global_best - global_best))
        global_best = new_global_best
        solutions.append(f(global_best) - penalizacao(global_best))
        color_plot.append('red' if penalizacao(global_best) > 0 else 'green')
        count = count + 1

    print(count, f(global_best) - penalizacao(global_best), f(global_best))

    # plt.scatter(range(count), solutions, c=color_plot)
    # plt.show()

    return (count, global_best, f(global_best), pcrossover, beta, pmutacao, elitismo, solutions, color_plot)


# R = []
# for pcrossover in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     for beta in[0.1, 0.25, 0.5, 0.75, 1.0]:
#         for pmutacao in [0.05, 0.1, 0.2, 0.3, 0.4]:
#             for elitismo in [0, 1, 2, 3]:
#                 R.append(ga_2(pcrossover, beta, pmutacao, elitismo))

# file = open('ga2', 'wb')
# pickle.dump(R, file)
# file.close()


# file = open('ga2', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# pcrossover, beta, pmutacao, elitismo = best[3], best[4], best[5], best[6]
pcrossover, beta, pmutacao, elitismo = 0.4, 0.1, 0.05, 3
print(pcrossover, beta, pmutacao, elitismo)
bests = [ga_2(pcrossover, beta, pmutacao, elitismo) for _ in range(30)]
# file = open('ga21', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('ga21', 'rb')
# bests = pickle.load(file)
# file.close()
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))