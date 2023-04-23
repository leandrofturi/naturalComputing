import numpy as np
# import matplotlib.pyplot as plt


###############################################################################
# PSO #########################################################################
###############################################################################


np.random.seed(42)

def pso(omega, phip, phig, fun, fun_penalizacao):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    local_best = x
    global_best = x[np.nanargmin([fun(x_i) for x_i in x])]
    vmin = xmin - xmax
    vmax = - vmin
    v = [np.random.random(len(vmin))*(vmax - vmin) + vmin for _ in range(n)]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []
    while (c > eps) and (count < max_iter):
        rp = np.random.random(n*n)
        rg = np.random.random(n*n)
        v = [np.array([min(max(omega * v[i][j] + phip * rp[i*n+j] * (local_best[i][j] - x[i][j]) + phig * rg[i*n+j] * (global_best[j] - x[i][j]), xmin[j]), xmax[j]) 
                      for j in range(len(x[i]))]) for i in range(n)]
        x = [x[i] + v[i] for i in range(n)]
        local_best = [x[i] if ((fun(x[i]) < fun(local_best[i])) and all(x[i] >= xmin) and all(x[i] <= xmax) or
                                ((fun_penalizacao(x[i]) < fun_penalizacao(local_best[i])) and all(x[i] >= xmin) and all(x[i] <= xmax)))
                      else local_best[i] for i in range(n)]
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
n = 100
xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
omega = 0.5
phip = 0.5
phig = 0.5
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

bests = [pso(omega, phip, phig, f1, penalizacao1) for _ in range(30)]
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))


###############################################################################
# problema 2 ##################################################################
###############################################################################


# parametros
n = 100
xmin = np.array([-10, -10, -10, -10, -10, -10, -10])
xmax = np.array([10, 10, 10, 10, 10, 10, 10])
omega = 0.5
phip = 0.5
phig = 0.5
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

bests = [pso(omega, phip, phig, f2, penalizacao2) for _ in range(30)]
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))
