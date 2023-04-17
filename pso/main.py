# import pickle
import numpy as np
# import matplotlib.pyplot as plt


###############################################################################
# problema 1 ##################################################################
###############################################################################


# parametros
n = 50
xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
omega = 0.9
phip = 0.5
phig = 0.3
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


def pso_1(omega, phip, phig):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    vmin = xmin - xmax
    vmax = - vmin
    v = [np.random.random(len(vmin))*(vmax - vmin) + vmin for _ in range(n)]
    local_best = x
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []
    while (c > eps) and (count < max_iter):
        rp = np.random.random(n)
        rg = np.random.random(n)
        v = [omega * v[i] + phip * rp[i] * (local_best[i] - x[i]) + phig * rg[i] * (global_best - x[i]) for i in range(n)]
        x = [x[i] + v[i] for i in range(n)]
        local_best = [x[i] if (f(x[i]) < f(local_best[i])) else local_best[i] for i in range(n)]
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

    return (count, global_best, f(global_best), omega, phip, phig, solutions, color_plot)


# R = []
# for omega in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for  phip in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         for phig in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#             R.append(pso_1(omega, phip, phig))

# file = open('pso1', 'wb')
# pickle.dump(R, file)
# file.close()

# file = open('pso1', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# omega, phip, phig = best[3], best[4], best[5]
omega, phip, phig = 0.9, 0.5, 0.3
print(omega, phip, phig)
bests = [pso_1(omega, phip, phig) for _ in range(30)]
# file = open('pso11', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('pso11', 'rb')
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
omega = 0.9
phip = 0.5
phig = 0.3
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


def pso_2(omega, phip, phig):
    # inicializacao
    x = [np.random.random(len(xmin))*(xmax - xmin) + xmin for _ in range(n)]
    vmin = xmin - xmax
    vmax = - vmin
    v = [np.random.random(len(vmin))*(vmax - vmin) + vmin for _ in range(n)]
    local_best = x
    global_best = x[np.nanargmin([f(x_i) for x_i in x])]

    # loop principal
    c = 1e9
    count = 0
    solutions = []
    color_plot = []
    while (c > eps) and (count < max_iter):
        rp = np.random.random(n)
        rg = np.random.random(n)
        v = [omega * v[i] + phip * rp[i] * (local_best[i] - x[i]) + phig * rg[i] * (global_best - x[i]) for i in range(n)]
        x = [x[i] + v[i] for i in range(n)]
        local_best = [x[i] if (f(x[i]) < f(local_best[i])) else local_best[i] for i in range(n)]
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

    return (count, global_best, f(global_best), omega, phip, phig, solutions, color_plot)

# R = []
# for omega in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for  phip in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         for phig in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#             R.append(pso_1(omega, phip, phig))

# file = open('pso2', 'wb')
# pickle.dump(R, file)
# file.close()

# file = open('pso2', 'rb')
# R = pickle.load(file)
# file.close()
# best = R[np.nanargmin([r[2] if np.isfinite(r[2]) else np.nan for r in R])]
# omega, phip, phig = best[3], best[4], best[5]
omega, phip, phig = 0.9, 0.2, 1.0
print(omega, phip, phig)
bests = [pso_2(omega, phip, phig) for _ in range(30)]
# file = open('pso21', 'wb')
# pickle.dump(bests, file)
# file.close()

# file = open('pso21', 'rb')
# bests = pickle.load(file)
# file.close()
f_xs = [r[2] for r in bests]
print([np.min(f_xs), np.max(f_xs), np.mean(f_xs), np.std(f_xs), np.median(f_xs)])
print(np.mean([r[0] for r in bests]))
