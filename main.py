from GA import *

with open('input.txt') as fin:
    n = int(fin.readline().strip())
    categories = np.array(list(map(int, fin.readline().strip().split()))) - 1
    time = np.array(list(map(float, fin.readline().strip().split())))
    m = int(fin.readline().strip())
    coefficients = np.zeros((m, 4))
    for i, line in enumerate(fin.readlines()):
        coefficients[i] = list(map(float, line.strip().split()))

n_categories = 4
ga = GA(n, m, n_categories, categories, time, coefficients,
        population_size=15, top=0.5, genes_mutate=0.4, select_policy='roulette wheel')

while ga.best_fitness < 1650:
    ga.step()
