# import pickle
import numpy as np
# import matplotlib.pyplot as plt


def selecao(x): # torneio
    xnew = []
    while len(xnew) < n:
        idxs = np.random.randint(0, n, 2)
        if f(x[idxs[0]]) < f(x[idxs[1]]):
            xnew.append(x[idxs[0]])
        else:
            xnew.append(x[idxs[1]])
    return xnew


def crossover_individual(x, x_mutated):
    ps = np.random.rand(len(x_mutated))
    idx = np.random.randint(0, n, 1)[0]
    xs = x[idx]
    u = [x_mutated[i] if ps[i] < pcrossover else xs[i] for i in range(len(x_mutated))]
    return np.array(u)


def mutacao_individual(x):
    idxs = np.random.randint(0, n, 3)
    idxs_std = np.argsort([f(x[idxs[0]]), f(x[idxs[1]]), f(x[idxs[2]])])
    return np.array(x[idxs_std[0]] + F*(x[idxs_std[1]] - x[idxs_std[2]]))


def crossover_plus_mutacao(x):
    xnew = []
    while len(xnew) < n:
        x_mutated = mutacao_individual(x)
        u = crossover_individual(x, x_mutated)
        xnew.append(u)
    return xnew


###############################################################################
# problema 1 ##################################################################
###############################################################################


# parametros
n = 50
xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
pcrossover = 0.8
F = 0.5
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


def de_1(pcrossover, F, n):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    while (c > eps) and (count < max_iter):
        x = crossover_plus_mutacao(x)
        x = selecao(x)
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

    return (count, global_best, f(global_best), pcrossover, F, n, solutions, color_plot)


R = []
for pcrossover in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for F in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for n in [4, 8, 10, 20, 50]:
                R.append(de_1(pcrossover, F, n))

# file = open('de1', 'wb')
# pickle.dump(R, file)
# file.close()

# file = open('de1', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# pcrossover, F, n = best[3], best[4], best[5]
pcrossover, F, n = 0.6, 0.5, 50
print(pcrossover, F, n)
bests = [de_1(pcrossover, F, n) for _ in range(30)]
# file = open('de11', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('de11', 'rb')
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
F = 0.5
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


def de_2(pcrossover, F, n):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    while (c > eps) and (count < max_iter):
        x = crossover_plus_mutacao(x)
        x = selecao(x)
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

    return (count, global_best, f(global_best), pcrossover, F, n, solutions, color_plot)


R = []
for pcrossover in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for F in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for n in [4, 8, 10, 20, 50]:
                R.append(de_2(pcrossover, F, n))

# file = open('de2', 'wb')
# pickle.dump(R, file)
# file.close()


# file = open('de2', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# pcrossover, F, n = best[3], best[4], best[5]
pcrossover, F, n = 0.8, 0.25, 50
print(pcrossover, F, n)
bests = [de_2(pcrossover, F, n) for _ in range(30)]
# file = open('de21', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('de21', 'rb')
# bests = pickle.load(file)
# file.close()
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))
