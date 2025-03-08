import random
import numpy as np
import functools

import utils

K = 10 #number of piles
POP_SIZE = 100 # population size
MAX_GEN = 4000 # maximum number of generations
CX_PROB = 0.35 # crossover probability
MUT_PROB = 0.75 # mutation probability
MUT_FLIP_PROB = 2/500 # probability of chaninging value during mutation / 1/ind_len
REPEATS = 10 # number of runs of algorithm (should be at least 10)
ELITE = 0.05
OUT_DIR = 'partition' # output directory for logs

SWAPS = 1
EXP_ID = f'cx{CX_PROB}-m{MUT_PROB}-f{MUT_FLIP_PROB}-e{ELITE}-G{MAX_GEN}-diff-rule-smart2-s{SWAPS}' # the ID of this experiment (used to create log names)  

# class for hall of fame
class HallOfFame:
    def __init__(self, size):
        self.size = size
        self.hof = []
        self.hof_fits = []

    # updates the hall of fame with new individuals
    # arguments:
    #   pop - new individuals
    #   fits - fitness values of the new individuals
    def update(self, pop, fits):
        is_candidate = [True if pop[i] not in self.hof else False for i in range(len(pop))]
        pop = [pop[i] for i in range(len(pop)) if is_candidate[i]]
        fits = [fits[i] for i in range(len(fits)) if is_candidate[i]]
        self.hof.extend(pop)
        self.hof_fits.extend(fits)
        new_hof, new_hof_fits = zip(*sorted(zip(self.hof, self.hof_fits), key=lambda x: x[1], reverse=True))
        self.hof = list(new_hof)[:self.size]
        self.hof_fits = list(new_hof_fits)[:self.size]
    
    # returns the hall of fame
    def get(self):
        return self.hof, self.hof_fits
    
    # changes the size of the hall of fame
    # arguments:
    #   new_size - new size of the hall of fame
    def change_size(self, new_size):
        if new_size < self.size:
            self.hof = self.hof[:new_size]
            self.hof_fits = self.hof_fits[:new_size]
        self.size = new_size

# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

# the fitness function
def fitness_diff(ind, weights):
    bw = bin_weights(weights, ind)
    fitness = 1/(max(bw) - min(bw) + 1)
    return utils.FitObjPair(fitness=fitness, 
                            objective=max(bw) - min(bw))

def fitness_std(ind, weights):
    bw = bin_weights(weights, ind)
    fitness = 1/(np.std(bw) + 1)
    return utils.FitObjPair(fitness=fitness, 
                            objective=max(bw) - min(bw))

def fitness_mean(ind, weights):
    bw = bin_weights(weights, ind)
    fitness = np.mean(bw)
    return utils.FitObjPair(fitness=fitness, 
                            objective=max(bw) - min(bw))

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]
    
# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    min_f = min(fits)
    fits = [f - min_f for f in fits]
    return random.choices(pop, fits, k=k)

def stochastic_universal_sampling(pop, fits, k):
    step = sum(fits)/k
    start = random.uniform(0, step)
    pointers = [start + i*step for i in range(k)]
    fits_summed = [sum(fits[:i+1]) for i in range(len(fits))]
    selected = []
    for i, ind in enumerate(pop):
        while pointers and fits_summed[i] >= pointers[0]:
            selected.append(ind)
            pointers.pop(0)
    return selected

        
# the rank selection
def rank_selection(pop, fits, k):
    sp = 2
    ranks = (len(fits) - np.argsort(fits)).tolist()
    ranks = [(sp - (2*sp-2)*(i-1)/(len(ranks)-1))/len(ranks) for i in ranks]

    return random.choices(pop, ranks, k=k)

# the tournament selection
def tournament_selection(pop, fits, k):
    touples = list(zip(pop, fits))
    return [max(random.sample(touples, 2), key=lambda x: x[1])[0] for _ in range(k)]

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

# implements the "swap" mutation of one individual
# arguments:
#   p - individual to be mutated
#   swaps - number of swaps to be performed
#   weights - list of weights of objects
#   K - number of bins
def smart_swap_mutate(p, swaps, weights, fit_fnc):
    # TODO implement the smart swap mutation
    ind = p[:]
    bw = bin_weights(weights, ind)
    best_fit = fit_fnc(p)
    for _ in range(swaps):
        max_w, max_i = max(tmp_list := list(zip(bw, range(len(bw)))), key=lambda x: x[0])
        min_w, min_i = min(tmp_list, key=lambda x: x[0])
        optimal_item_weight = (max_w - min_w)/2

        max_idx = (i for i in range(len(ind)) if ind[i] == max_i)
        min_idx = (i for i in range(len(ind)) if ind[i] == min_i)
        best_fit_i = min(list(max_idx), key=lambda x: abs(weights[x] - optimal_item_weight))

        bw[max_i] -= weights[best_fit_i]
        bw[min_i] += weights[best_fit_i]
        1/(max(bw) - min(bw) + 1)
        # TODO if the swap would not improve the fitness, do not perform it
        # TODO update bw to see if the swap improved the fitness
        # swap the item
        ind[best_fit_i] = min_i

def random_smart_swap_mutate(p, swaps, weights):
    ind = p[:]
    bw = bin_weights(weights, ind)
    for _ in range(swaps):
        max_i = max(pile_indicies := list(range(len(bw))), key=lambda x: bw[x])
        # min_i = min(pile_indicies, key=lambda x: bw[x])
        max_idx = (i for i in range(len(ind)) if ind[i] == max_i)
        
        probs = [abs(bw[i] - bw[max_i]) for i in pile_indicies]
        selected_pile_index = random.choices(pile_indicies, probs)[0]
        selected_item_index = random.choices(list(max_idx))[0]

        bw[max_i]               -= weights[selected_item_index]
        bw[selected_pile_index] += weights[selected_item_index]

        ind[selected_item_index] = selected_pile_index
    return ind

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
#   hof       - hall of fame structure (default `None`)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, hof:HallOfFame=None, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)

        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        if hof:
            hof.update(pop, fits)

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)

        pop = offspring[:]
        if hof:
            for e in hof.hof:
                if e not in pop:
                    # random_index = random.randint(0, len(pop))
                    pop[random.randint(0, len(pop)-1)] = e[:]
                

        # pop = offspring[:] + hof.hof if hof is not None else offspring[:]

    return pop

if __name__ == '__main__':
    # read the weights from input
    weights = read_weights('inputs/partition-easy.txt')

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=len(weights))
    # cr_ind = functools.partial(create_ind_lightest_first, ind_len=len(weights), weights=weights)
    # fit = functools.partial(fitness_std, weights=weights)
    fit = functools.partial(fitness_diff, weights=weights)
    xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
    # mut = functools.partial(mutation, mut_prob=MUT_PROB, 
    #                         mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K))
    # mut = functools.partial(mutation, mut_prob=MUT_PROB,
    #                         mutate=functools.partial(smart_swap_mutate, swaps=1, weights=weights, fit_fnc=fit))
    mut = functools.partial(mutation, mut_prob=MUT_PROB,
                            mutate=functools.partial(random_smart_swap_mutate, swaps=1, weights=weights))

    # we can use multiprocessing to evaluate fitness in parallel
    import multiprocessing
    pool = multiprocessing.Pool()

    import matplotlib.pyplot as plt

    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations
    best_inds = []
    for run in range(REPEATS):
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run, 
                        write_immediately=True, print_frequency=5)
        hof = HallOfFame(int(ELITE*POP_SIZE)) if ELITE else None
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        # pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], stochastic_universal_sampling, hof=hof, map_fn=pool.map, log=log)
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], roulette_wheel_selection, hof=hof, map_fn=pool.map, log=log)
        # pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, hof=hof, map_fn=pool.map, log=log)
        # pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], rank_selection, hof=hof, map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            for w, b in zip(weights, bi):
                f.write(f'{w} {b}\n')
        
        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')

    print(EXP_ID)
    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)

    # read the summary log and plot the experiment
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
    plt.figure(figsize=(12, 8))
    utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
    plt.legend()
    plt.show()

    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned') 