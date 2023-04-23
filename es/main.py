import numpy as np
# import matplotlib.pyplot as plt


###############################################################################
# ES ##########################################################################
###############################################################################


def selecao(x, fun, n):
    xnew = []
    x_sorted = sorted(x, key=fun)
    for i in range(n):
        xnew.append(x_sorted[i])
    return xnew


def recombinacao(x, fun, fun_penalizacao):
    xnew = []
    while len(xnew) < n:
        idxs = np.random.randint(0, n, 2)
        c = np.array([min(max((x[idxs[0]][i] + x[idxs[1]][i]) / 2, xmin[i]), xmax[i]) for i in range(len(x[idxs[0]]))])
        if fun_penalizacao(c) <= 0:
            xnew.append(c)
        elif (fun(c) < fun(x[idxs[0]])) or (fun_penalizacao(c) < fun_penalizacao(x[idxs[0]])):
            xnew.append(c)
        else:
            xnew.append(x[idxs[0]])
    return xnew


def mutacao(x, tau, step_size):
    xnew = []
    for idx in range(n):
        ps = np.random.random(len(x[idx]))
        sigmal = step_size*np.exp(tau*np.random.random(len(x[idx])))
        xnew.append(np.array([min(max(x[idx][i] + ps[i]*sigmal[i], xmin[i]), xmax[i]) for i in range(len(x[idx]))]))
    return xnew


def es(tau, step_size, mues, lambdaes, fun, fun_penalizacao):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    global_best = x[np.nanargmin([fun(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []

    while (c > eps) and (count < max_iter):
        x_children = recombinacao(x, fun, fun_penalizacao)
        x_children = mutacao(x_children, tau, step_size)
        x_mu = selecao(x, fun, int(n*mues))
        x_lambda = selecao(x_children, fun, max(n - int(n*mues), int(n*lambdaes)))
        x = x_mu + x_lambda
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
tau = 0.15
step_size = 0.5
mues = 1/7 # numero de pais selecionados
lambdaes = 6/7 # numero de filhos gerados pelos pai
r = 1000 # fator de penalizacao
eps = 1e-5
max_iter = 100000

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

bests = [es(tau, step_size, mues, lambdaes, f1, penalizacao1) for _ in range(30)]
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
tau = 0.15
step_size = 0.5
mues = 1/7 # numero de pais selecionados
lambdaes = 6/7 # numero de filhos gerados pelos pai
r = 1000 # fator de penalizacao
eps = 1e-5
max_iter = 100000

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

bests = [es(tau, step_size, mues, lambdaes, f2, penalizacao2) for _ in range(30)]
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))
