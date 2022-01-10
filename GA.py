from typing import Union
import numpy as np


class GA:
    rng = np.random.default_rng()

    def __init__(self, n: int, m: int, n_categories: int, tasks_categories: np.ndarray, estimated_time: np.ndarray, coefficients: np.ndarray,
                 population_size: int=10, top: float = 0.1, genes_mutate: float = 0.02, select_policy: str='elitism'):
        """
        Initiates all class fields, also creates initial population and calculates its fitness

        :param n: number of tasks
        :param m: number of developers
        :param n_categories: number of categories
        :param tasks_categories: category of each task
        :param estimated_time: estimated time to complete each task
        :param coefficients: coefficients of real/estimated time relation for every task category of every developer
        :param population_size: population size
        :param top: proportion of individuals to select
        :param genes_mutate: proportion of genes to mutate
        :param select_policy: selection policy, 'elitism', 'roulette wheel' available
        """
        self.n_categories = n_categories
        self.n = n
        self.m = m
        self.population_size = population_size
        self.estimated_time = estimated_time
        self.coefficients = coefficients
        self.tasks_categories = tasks_categories
        self.top = max(int(np.floor(top * self.population_size)), 2)
        self.genes_mutate = max(int(np.floor(genes_mutate * self.n)), 1)
        self.select_policy = select_policy

        self.fitness = None
        self.population = None
        self.best_fitness = np.inf
        self.best_individual = None
        self.best_index = None

        self.create_population()
        self.population_fitness()

    def create_population(self):
        """ Creates the initial population of `self.size` elements """
        self.population = np.empty((self.population_size, self.n))
        for individual in self.population:
            individual[:] = GA.rng.choice(self.m, size=self.n)


    def population_fitness(self):
        """ Calculate fitness of every individual in the population """
        self.fitness = np.array(list(map(self.individual_fitness, self.population)))


    def individual_fitness(self, individual: np.ndarray) -> float:
        """
        Returns the fitness of an individual, e.g. max time to finish all tasks
        :param individual: a distribution of developers among tasks (`i`-th task is taken by developer `individual[i]`)
        :return: fitness of an `individual`
        """
        devs_time = np.zeros(self.m)
        for dev in range(self.m):
            inds = np.where(individual == dev)[0]
            tasks_categories = self.tasks_categories[inds]
            est_time = self.estimated_time[inds]
            difficulty = self.coefficients[dev]
            devs_time[dev] = sum(map(lambda cat: sum(difficulty[cat] * est_time[np.where(tasks_categories == cat)]),
                                     range(self.n_categories)))
        return np.max(devs_time)

    def selection(self):
        """ Selects elements to create next generation using chosen selection policy
        :raise: NotImplemented if `self.select_policy` is invalid
        """
        if self.select_policy == 'elitism':
            inds = self.elitism_selection()
        elif self.select_policy == 'roulette wheel':
            inds = self.roulette_wheel_selection()
        else:
            raise NotImplementedError('no such selection policy')
        self.population = self.population[inds].copy()

    def elitism_selection(self) -> np.ndarray:
        """ Performs selection elitism policy, number of remaining individuals is `self.top`
        :return: indices of selected individuals from `self.population`
        """
        return np.argsort(self.fitness)[self.top]


    def roulette_wheel_selection(self) -> np.ndarray:
        """ Performs selection roulette wheel policy, number of remaining individuals is `self.top`
         :return: indices of selected individuals from `self.population`
         """
        probs = self.fitness / np.sum(self.fitness)
        return GA.rng.choice(self.population_size, size=self.top, p=probs)


    def crossover(self):
        """ Performs crossover of remaining (after selection) individuals and creates new individuals """
        self.new = np.zeros((self.population_size - len(self.population), self.n))
        for i in range(len(self.new)):
            a, b = GA.rng.choice(self.population.shape[0], size=2, replace=False)
            j = GA.rng.integers(self.n)
            self.new[i, :j] = self.population[a, :j]
            self.new[i, j:] = self.population[b, j:]

    def mutation(self):
        """ Mutates new individuals, number of new individuals is defined by `self.genes_mutate` """
        for i in range(len(self.new)):
            j = GA.rng.choice(self.n, size=self.genes_mutate)
            self.new[i, j] = GA.rng.choice(self.m, size=self.genes_mutate)

    def step(self):
        """ Performs a step of a genetic algorithm, finds best individual of all the previous steps and current one """
        self.selection()
        self.crossover()
        self.mutation()
        self.population = np.vstack((self.population, self.new))
        self.population_fitness()

        new_best_ind = np.argmin(self.fitness)
        if self.fitness[new_best_ind] < self.best_fitness:
            self.best_fitness = self.fitness[new_best_ind]
            self.best_individual = self.population[new_best_ind]

    def score(self) -> float:
        """
        Calculates task-specific score of algorithm performance up till last step
        :return: the best score
        """
        if self.best_fitness < np.inf:
            return 1e6 / self.best_fitness
        return 0.

