import random
import numpy as np
import functools

import co_functions as cf
import utils

DIMENSION = 10 # dimension of the problems
POP_SIZE = 100 # population size
MAX_GEN = 500 # maximum number of generations
CX_PROB = 1 # crossover probability
CR_PROB = 1/DIMENSION # probability of flipping each bit in mutation
MUT_PROB = 0.2 # mutation probability
MUT_STEP = 0.5 # size of the mutation steps
MUT_STEP_FINAL = 0.01 # final step size of the mutation
REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'continuous' # output directory for logs
EXP_ID = 'differential_complete' # the ID of this experiment (used to create log names)


def linear_scaling(current, min_value, max_value, start_value=0.5, end_value=0.01):
    ratio = (current - min_value) / (max_value - min_value)
    return start_value + ratio*(end_value - start_value)

def exp_decay(current, min_value, max_value, start_value=0.5, end_value=0.01):
    ratio = (current - min_value) / (max_value - min_value)
    return start_value * (end_value/start_value)**ratio

# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have 
# negative fitness)
def tournament_selection(pop, fits, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(np.copy(pop[p1]))
        else:
            selected.append(np.copy(pop[p2]))

    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return o1, o2


# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class Mutation:

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, ind):
        return ind + self.step_size*np.random.normal(size=ind.shape)
    
class ArithmeticCrossover:
    def __init__(self, f1=1, f2=1, fitnesses=None):
        self.f1 = f1
        self.f2 = f2
        
        self.fitnesses = fitnesses
        self.counter = 0
        
    @staticmethod
    def arithmetic_cross(p1, p2, f1=1, f2=1):
        w1, w2 = f1/(f1+f2), f2/(f1+f2)
        assert abs(1 - (w1 + w2)) < 1e-5, f'w1 + w2 = {w1 + w2}; should add up to 1'

        o1 = (w1*p1 + w2*p2)
        o2 = (w1*p2 + w2*p1)
        return o1, o2
    
    def reset(self):
        self.counter = 0
        self.fitnesses = None

    def __call__(self, p1, p2):
        if self.fitnesses is None:
            f1, f2 = self.f1, self.f2
        else:
            f1, f2 = self.fitnesses[self.counter*2], self.fitnesses[self.counter*2+1]
        self.counter += 1

        return ArithmeticCrossover.arithmetic_cross(p1, p2, f1, f2)
    
class DifferentialMutation:
    def __init__(self, step_size, pop=None, k=1):
        self.step_size = step_size # F
        self.pop = pop #reference to the population
        self.k = k # how many pairs of individuals to use for mutation
        self.donors = []
        self.counter = 0

    def reset(self):
        self.donors = []
        self.counter = 0

    def __call__(self, ind):
        if self.pop is None:
            raise ValueError('Population not set')
        
        # idx_pool = [i for i in range(len(self.pop)) if i != self.counter]
        # if not self.use_as_base:
        #     selected_idx = np.random.choice(len(self.pop), self.k*2 + 1, replace=False)
        #     base = self.pop[selected_idx[-1]]
        #     selected_idx = selected_idx[:-1]

        #     selected = np.array([self.pop[i] for i in selected_idx])
        # else:
        selected_idx = np.random.choice(len(self.pop), self.k*2, replace=False)
        selected = np.array([self.pop[i] for i in selected_idx])
        base = ind

        diffs = selected[0::2] - selected[1::2]
        direction = np.sum(diffs, axis=0)

        self.donors.append(base + self.step_size*direction)
        return ind[:]
    
class DifferentialCrossover:
    def __init__(self):
        self.candidates = []
        self.counter = 0

    def reset(self):
        self.counter = 0
        self.candidates = []

    def __call__(self, p1, p2, cr_prob, mutate_ind):
        for ind in [p1, p2]:
            ind = ind[:]
            for i in range(len(ind)):
                if random.random() < cr_prob:
                    ind[i] = mutate_ind.donors[self.counter][i]
            self.candidates.append(ind)
            self.counter += 1
        return p1, p2


def differential_selection(pop, fits, k, mutate_ind, cross_ind, fit):
    selected = []
    candidate_fits = list(map(fit, cross_ind.candidates))
    for i in range(k):
        if fits[i] > candidate_fits[i]:
            selected.append(np.copy(pop[i]))
        else:
            selected.append(np.copy(cross_ind.candidates[i]))
    return selected

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, cross_ind, mutate_ind, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mutate_ind.reset()
        cross_ind.reset()
        # mutate_ind.step_size = linear_scaling(G, 0, max_gen, MUT_STEP, MUT_STEP_FINAL)
        # mutate_ind.step_size = exp_decay(G, 0, max_gen, MUT_STEP, MUT_STEP_FINAL)
        mutate_ind.pop = pop

        cross_ind.counter = 0
        cross_ind.fitnesses = fits

        # mating_pool = mate_sel(pop, fits, POP_SIZE)
        # offspring = mate(mating_pool, operators)
        pool = mate(pop, operators)
        offspring = mate_sel(pool, fits, POP_SIZE)
        pop = offspring[:]

    return pop

if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        # mutate_ind = Mutation(step_size=MUT_STEP)
        mutate_ind = DifferentialMutation(step_size=MUT_STEP)
        # cross_ind = ArithmeticCrossover()
        cross_ind = DifferentialCrossover()
        # xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        xover = functools.partial(crossover,
                                  cross=functools.partial(cross_ind,
                                                          mutate_ind=mutate_ind,
                                                          cr_prob=CR_PROB), 
                                  cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)
        # sel = functools.partial(tournament_selection)
        sel = functools.partial(differential_selection, mutate_ind=mutate_ind, cross_ind=cross_ind, fit=fit)

        # run the algorithm `REPEATS` times and remember the best solutions from 
        # last generations
    
        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name , run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [mut, xover], sel, cross_ind, mutate_ind, map_fn=map, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()
        print(EXP_ID)
        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)