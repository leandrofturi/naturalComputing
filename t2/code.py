import sys
import time
import random
import pickle

import numpy as np
from datetime import timedelta
# import matplotlib.pyplot as plt


"""# Bee Colony"""

# maximization problem

def initialize(PS, bee_size):
    beemin = -1
    beemax = 1
    hive = [beemin + np.random.random(bee_size)*(beemax - beemin) for _ in range(PS)]
    return hive


def evaluate(hive, fitness):
    F = [{"X": bee, "F": fitness(bee)} for bee in hive]
    return F


def employed_bees(hive, fitness):
    V = [bee["X"] + random.uniform(-1, 1)*(bee["X"] - random.choice(hive)["X"]) for bee in hive]
    V = evaluate(V, fitness)
    has_improved = False
    E = []
    for bee, v in zip(hive, V):
        if v["F"] > bee["F"]:
            E.append(v)
            has_improved = True
        else:
            E.append(bee)
    return E, has_improved


def onlooker_bees(hive, fitness):
    V = [bee["X"] + random.uniform(-1, 1)*(bee["X"] - random.choice(hive)["X"]) for bee in hive]
    V = evaluate(V, fitness)
    sum_fitness = sum([bee["F"] for bee in hive])
    Pbees = [bee["F"]/sum_fitness for bee in hive]
    sum_fitness = sum([v["F"] for v in V])
    PV = [v["F"]/sum_fitness for v in V]
    has_improved = False
    O = []
    for bee, v, pbee, pv in zip(hive, V, Pbees, PV):
        if pv > pbee:
            O.append(v)
            has_improved = True
        else:
            O.append(bee)
    return O, has_improved


def scout_bees(bees, fitness):
    _PS = len(bees)
    _bee_size = len(bees[0]["X"])
    S = initialize(_PS, _bee_size)
    S = evaluate(S, fitness)
    return S

def ABC(X, Y, target_map, PS, bee_size, Lit, max_it, fitness):
    K = len(target_map)
    _fitness = lambda x: fitness(X, Y, target_map, K, x)
    hive = initialize(PS, bee_size)
    hive = evaluate(hive, _fitness)
    gbest = hive[np.argmax([bee["F"] for bee in hive])]
    history = []
    n_iter = 1
    Lit_count = 0
    while (n_iter < max_it):
        sys.stdout.write("\r%d" % n_iter)
        sys.stdout.flush()

        hive, employed_has_improved = employed_bees(hive, _fitness)
        hive, onlooker_has_improved = onlooker_bees(hive, _fitness)
        if (not employed_has_improved) and (not onlooker_has_improved):
            Lit_count = Lit_count + 1
        else:
            Lit_count = 0
        if Lit_count >= Lit:
            hive = scout_bees(hive, _fitness)
            Lit_count = 0
        n_iter = n_iter + 1
        best = hive[np.argmax([bee["F"] for bee in hive])]
        if best["F"] > gbest["F"]:
            gbest = best
        history.append(gbest["F"])
    sys.stdout.write("\n")
    return gbest["X"], gbest["F"], history


def fitness(X, Y, target_map, K, bee):
    # Y_pred = []
    # for row in X:
    #     distances = [np.linalg.norm(row - centroid) for centroid in np.array_split(bee, K)]
    #     Y_pred.append(target_map[np.argmax(distances)])
    Y_pred = [target_map[np.argmax([np.linalg.norm(row - centroid) for centroid in np.array_split(bee, K)])] for row in X]
    return sum(Y_pred == Y)/len(Y)


"""# Train

## Data
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split

# data = datasets.load_breast_cancer()
# data = datasets.load_diabetes()
# data = datasets.load_wine()

def prepare_dataset(data):
    X = data["data"]
    Y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size=0.25, random_state=42)
    # normalization
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train_scl = [(x - mu)/sigma for x in X_train]
    X_test_scl = [(x - mu)/sigma for x in X_test]

    target_map = {i: y for i, y in zip(range(len(np.unique(Y))), np.unique(Y))}

    return X_train_scl, X_test_scl, y_train, y_test, target_map, mu, sigma


"""# Train

### wine
"""

X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_wine())

PS = 50
Lit = 30
max_it = 500
K = len(target_map)
bee_size = K * len(X[0])

start = time.time()
X_ABC, f_ABC, hist_ABC = ABC(X, Y, target_map, PS, bee_size, Lit, max_it, fitness)
end = time.time()
print(str(timedelta(seconds=end - start)))
print(f_ABC)
print(fitness(X_test, Y_test, target_map, K, X_ABC))

file = open('data/wine.dat', 'wb')
d = {"X_ABC": X_ABC, "f_ABC": f_ABC, "hist_ABC": hist_ABC, 
     "X": X, "X_test": X_test, "Y": Y, "Y_test": Y_test, "target_map": target_map, "mu": mu, "sigma": sigma }
pickle.dump(d, file)
file.close()

# plt.plot(range(len(hist_ABC)), hist_ABC)
# plt.xlabel("iter")
# plt.ylabel("acc")
# plt.show()


"""# Train

### breast_cancer
"""

X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_breast_cancer())

PS = 50
Lit = 30
max_it = 500
K = len(target_map)
bee_size = K * len(X[0])

start = time.time()
X_ABC, f_ABC, hist_ABC = ABC(X, Y, target_map, PS, bee_size, Lit, max_it, fitness)
end = time.time()
print(str(timedelta(seconds=end - start)))
print(f_ABC)
print(fitness(X_test, Y_test, target_map, K, X_ABC))

file = open('data/breast_cancer.dat', 'wb')
d = {"X_ABC": X_ABC, "f_ABC": f_ABC, "hist_ABC": hist_ABC, 
     "X": X, "X_test": X_test, "Y": Y, "Y_test": Y_test, "target_map": target_map, "mu": mu, "sigma": sigma }
pickle.dump(d, file)
file.close()

# plt.plot(range(len(hist_ABC)), hist_ABC)
# plt.xlabel("iter")
# plt.ylabel("acc")
# plt.show()


"""# Train

### diabetes
"""

# X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_diabetes())
import datapackage
import pandas as pd

data_url = 'https://datahub.io/machine-learning/diabetes/datapackage.json'
package = datapackage.Package(data_url)
resources = package.resources
data = []
for resource in resources:
    if resource.tabular:
        data.append(pd.read_csv(resource.descriptor['path']))
diabetes = pd.concat(data)
data = {"target": np.array([1 if c == "tested_positive" else 0 for c in diabetes["class"]]),
        "data": diabetes[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].to_numpy()}
X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(data)

PS = 50
Lit = 30
max_it = 500
K = len(target_map)
bee_size = K * len(X[0])

start = time.time()
X_ABC, f_ABC, hist_ABC = ABC(X, Y, target_map, PS, bee_size, Lit, max_it, fitness)
end = time.time()
print(str(timedelta(seconds=end - start)))
print(f_ABC)
print(fitness(X_test, Y_test, target_map, K, X_ABC))

file = open('data/diabetes.dat', 'wb')
d = {"X_ABC": X_ABC, "f_ABC": f_ABC, "hist_ABC": hist_ABC, 
     "X": X, "X_test": X_test, "Y": Y, "Y_test": Y_test, "target_map": target_map, "mu": mu, "sigma": sigma }
pickle.dump(d, file)
file.close()

# plt.plot(range(len(hist_ABC)), hist_ABC)
# plt.xlabel("iter")
# plt.ylabel("acc")
# plt.show()


"""# Comparison"""


import numpy as np
from pymoo.core.problem import ElementwiseProblem


class problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=bee_size, n_obj=1, n_constr=0, xl=[-100 for _ in range(bee_size)], xu=[100 for _ in range(bee_size)])

    def _evaluate(self, x, out, *args, **kwargs):
        bee = x
        Y_pred = [target_map[np.argmax([np.linalg.norm(row - centroid) for centroid in np.array_split(bee, K)])] for row in X]
        out["F"] = -sum(Y_pred == Y)/len(Y)


"""# Comparison

# GA
"""


from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA


X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_wine())
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = GA(pop_size=50)
d_wine = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_wine.F, len(d_wine.history))
print(fitness(X_test, Y_test, target_map, K, d_wine.X))


X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_breast_cancer())
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = GA(pop_size=50)
d_breast_cancer = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_breast_cancer.F, len(d_breast_cancer.history))
print(fitness(X_test, Y_test, target_map, K, d_breast_cancer.X))


# X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_diabetes())
data_url = 'https://datahub.io/machine-learning/diabetes/datapackage.json'
package = datapackage.Package(data_url)
resources = package.resources
data = []
for resource in resources:
    if resource.tabular:
        data.append(pd.read_csv(resource.descriptor['path']))
diabetes = pd.concat(data)
data = {"target": np.array([1 if c == "tested_positive" else 0 for c in diabetes["class"]]),
        "data": diabetes[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].to_numpy()}
X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(data)
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = GA(pop_size=50)
d_diabetes = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_diabetes.F, len(d_diabetes.history))
print(fitness(X_test, Y_test, target_map, K, d_diabetes.X))


file = open('data/GA.dat', 'wb')
d = {"wine": d_wine, "breast_cancer": d_breast_cancer, "diabetes": d_diabetes}
pickle.dump(d, file)
file.close()


"""# Comparison

# ES
"""


from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.es import ES


X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_wine())
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = ES(pop_size=50)
d_wine = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_wine.F, len(d_wine.history))
print(fitness(X_test, Y_test, target_map, K, d_wine.X))


X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_breast_cancer())
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = ES(pop_size=50)
d_breast_cancer = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_breast_cancer.F, len(d_breast_cancer.history))
print(fitness(X_test, Y_test, target_map, K, d_breast_cancer.X))


# X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(datasets.load_diabetes())
data_url = 'https://datahub.io/machine-learning/diabetes/datapackage.json'
package = datapackage.Package(data_url)
resources = package.resources
data = []
for resource in resources:
    if resource.tabular:
        data.append(pd.read_csv(resource.descriptor['path']))
diabetes = pd.concat(data)
data = {"target": np.array([1 if c == "tested_positive" else 0 for c in diabetes["class"]]),
        "data": diabetes[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].to_numpy()}
X, X_test, Y, Y_test, target_map, mu, sigma = prepare_dataset(data)
K = len(target_map)
bee_size = K * len(X[0])
p = problem()
algorithm = ES(pop_size=50)
d_diabetes = minimize(p, algorithm, seed=1, verbose=False, save_history=True)
print(d_diabetes.F, len(d_diabetes.history))
print(fitness(X_test, Y_test, target_map, K, d_diabetes.X))


file = open('data/ES.dat', 'wb')
d = {"wine": d_wine, "breast_cancer": d_breast_cancer, "diabetes": d_diabetes}
pickle.dump(d, file)
file.close()


# file = open('data/diabetes.dat', 'rb')
# abc = pickle.load(file)
# file.close()
# file = open('data/GA.dat', 'rb')
# ga = pickle.load(file)
# file.close()
# file = open('data/ES.dat', 'rb')
# es = pickle.load(file)
# file.close()

# plt.plot(range(len(abc["hist_ABC"])), abc["hist_ABC"], label="abc")

# hist_F = []
# for algo in ga["diabetes"].history:
#     opt = algo.opt
#     feas = np.where(opt.get("feasible"))[0]
#     hist_F.append(-opt.get("F")[feas][0][0])
# plt.plot(range(len(hist_F)), hist_F, label="ga")

# hist_F = []
# for algo in es["diabetes"].history:
#     opt = algo.opt
#     feas = np.where(opt.get("feasible"))[0]
#     hist_F.append(-opt.get("F")[feas][0][0])
# plt.plot(range(len(hist_F)), hist_F, label="es")

# plt.xlabel("iter")
# plt.ylabel("acc")
# plt.legend()
# plt.show()

