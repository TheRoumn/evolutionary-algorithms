import functools
import itertools
import math
import numpy as np
import random

import utils
# random.seed(42)

POP_SIZE = 100 # population size
MAX_GEN = 4000 # maximum number of generations
CX_PROB = 0.5 # crossover probability
MUT_PROB = 0.5 # mutation probability
MUT_MAX_LEN = 10 # maximum lenght of the swapped part
REPEATS = 10 # number of runs of algorithm (should be at least 10)
SUFFIX = 'std'
# SUFFIX = 'test'
INPUT = f'inputs/tsp_{SUFFIX}.in' # the input file
OUT_DIR = 'tsp' # output directory for logs
CROSS = 'er'
MUT = 'opt'
EXP_ID = f'{SUFFIX}-cx{CX_PROB}-m{MUT_PROB}-ml{MUT_MAX_LEN}-{CROSS}-{MUT}-long' # the ID of this experiment (used to create log names)



# reads the input set of values of objects
def read_locations(filename):
    locations = []
    with open(filename) as f:
        for l in f.readlines():
            tokens = l.split(' ')
            locations.append((float(tokens[0]), float(tokens[1])))
    return locations

@functools.lru_cache(maxsize=None) # this enables caching of the values
def distance(loc1, loc2):
    # based on https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [loc1[1], loc1[0], loc2[1], loc2[0]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371.01 * c
    return km

# the fitness function
def fitness(ind, cities):
    
    # quickly check that ind is a permutation
    num_cities = len(cities)
    assert len(ind) == num_cities
    assert sum(ind) == num_cities*(num_cities - 1)//2

    dist = 0
    for a, b in zip(ind, ind[1:]):
        dist += distance(cities[a], cities[b])

    dist += distance(cities[ind[-1]], cities[ind[0]])

    return utils.FitObjPair(fitness=-dist, 
                            objective=dist)

# creates the individual (random permutation)
def create_ind(ind_len):
    ind = list(range(ind_len))
    random.shuffle(ind)
    return ind

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection
def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(pop[p1][:])
        else:
            selected.append(pop[p2][:])

    return selected

# implements the order crossover of two individuals
def order_cross(p1, p2):
    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    start = min(point1, point2)
    end = max(point1, point2)

    # swap the middle parts
    o1mid = p2[start:end]
    o2mid = p1[start:end]

    # take the rest of the values and remove those already used
    restp1 = [c for c in p1[end:] + p1[:end] if c not in o1mid]
    restp2 = [c for c in p2[end:] + p2[:end] if c not in o2mid]

    o1 = restp1[-start:] + o1mid + restp1[:-start]
    o2 = restp2[-start:] + o2mid + restp2[:-start]

    return o1, o2

def pmx_cross(p1, p2):
    def use_mapping(value, mapping):
        return mapping[value] if value in mapping.keys() else value

    def create_mapping(p1, p2):
        result_map = {}
        helper_map = {p1[i]: p2[i] for i in range(len(p1))}
        for i, n in enumerate(p1):
            # identical values
            if p1[i] == p2[i]:
                continue

            # mapping propagation
            local_n = n
            visited = set()
            while local_n in helper_map.keys():
                # cycle detected
                if local_n in visited:
                    break
                visited.add(local_n)

                local_n = helper_map[local_n]

            else:
                result_map[n] = local_n
                continue
                
        return result_map

    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    start = min(point1, point2)
    end = max(point1, point2)

    # swap the middle parts
    p1mid = p1[start:end]
    p2mid = p2[start:end]
    o1mid = p2mid
    o2mid = p1mid

    p1p2_map = create_mapping(p1mid, p2mid)
    p2p1_map = create_mapping(p2mid, p1mid)
    o1_use_mapping = functools.partial(use_mapping, mapping=p2p1_map)
    o2_use_mapping = functools.partial(use_mapping, mapping=p1p2_map)

    o1 = list(map(o1_use_mapping, p1[:start])) + o1mid + list(map(o1_use_mapping, p1[end:]))
    o2 = list(map(o2_use_mapping, p2[:start])) + o2mid + list(map(o2_use_mapping, p2[end:]))
    assert len(o1) == len(p1)
    assert len(np.unique(o1)) == len(o1) 
    assert len(np.unique(o2)) == len(o2) 

    return o1, o2

def edge_recombination(p1, p2):
    def get_neighbors(n, container=set):
        result = []
        for ind in [p1, p2]:
            index = ind.index(n)
            off_range = [-1, 1] if index != len(p1) - 1 else [-1, -len(p1) + 1]
            result.extend([ind[index + offset] for offset in off_range])
        return container(result)

    def update_neighbor_map(n_to_remove):
        if n_to_remove not in neighbor_map.keys():
            return
        neighbor_map.pop(n_to_remove)
        to_remove = set()
        for k, v in neighbor_map.items():
            if n_to_remove in v:
                neighbor_map[k].remove(n_to_remove)
            if len(v) == 0:
                to_remove.add(k)

        for k in to_remove:
            neighbor_map.pop(k)
    
    def get_mins(neighbor_map:dict, current=None):
        if len(neighbor_map) == 0:
            return []
        if current is not None and current in neighbor_map.keys():
            neighbor_map = {k: v for k, v in neighbor_map.items() if k in neighbor_map[current]}
        min_length = min([len(v) for _, v in neighbor_map.items()])
        return [k for k, v in neighbor_map.items() if len(v) == min_length]

    neighbor_map = {p1[i]: get_neighbors(p1[i], container=set) for i in range(len(p1))}

    result = []
    key = None
    while len(neighbor_map) > 0:
        if key is None:
            min_keys = get_mins(neighbor_map, key)
        key = min_keys[0] if len(min_keys) == 1 else random.choice(min_keys)
        result.append(key) 
        min_keys = get_mins(neighbor_map, key)
        if len(neighbor_map) > 0:
            update_neighbor_map(key)

    if len(result) < len(p1):
        rest = [i for i in p1 if i not in result]

    result += rest

    assert len(np.unique(result)) == len(result)
    assert len(result) == len(p1)

    return result, swap_mutate(result, MUT_MAX_LEN)

# implements the swapping mutation of one individual
def swap_mutate(p, max_len):
    source = random.randrange(1, len(p) - 1)
    dest = random.randrange(1, len(p))
    lenght = random.randrange(1, min(max_len, len(p) - source))

    o = p[:]
    move = p[source:source+lenght]
    o[source:source + lenght] = []
    if source < dest:
        dest = dest - lenght # we removed `lenght` items - need to recompute dest
    
    o[dest:dest] = move
    
    return o

def invert_mutate(p, max_len):
    source = random.randrange(1, len(p) - 1)
    lenght = random.randrange(1, min(max_len, len(p) - source))

    o = p[:]
    o[source:source+lenght] = o[source:source+lenght][::-1]

    return  o

def opt_mutate(p, max_len, fit_fnc=None):
    o = invert_mutate(p, max_len)

    if random.random() < 0.1:
        return o

    return p[:] if fit_fnc is not None and fit_fnc(o).fitness < fit_fnc(p).fitness else o

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
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)

        pop = offspring[:-1] + [max(list(zip(fits, pop)), key = lambda x: x[0])[1]]

    return pop

cx_fnc = {"order": order_cross,
          "pmx": pmx_cross,
          "er": edge_recombination}
mut_fnc = {"swap": swap_mutate,
           "invert": invert_mutate,
           "opt": functools.partial(opt_mutate, fit_fnc=functools.partial(fitness, cities=read_locations(INPUT))),
           }
assert CROSS in cx_fnc.keys()
assert MUT in mut_fnc.keys()

if __name__ == '__main__':
    # read the locations from input
    locations = read_locations(INPUT)

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=len(locations))
    fit = functools.partial(fitness, cities=locations)
    
    # xover = functools.partial(crossover, cross=order_cross, cx_prob=CX_PROB)
    # xover = functools.partial(crossover, cross=pmx_cross, cx_prob=CX_PROB)
    # xover = functools.partial(crossover, cross=edge_recombination, cx_prob=CX_PROB)
    xover = functools.partial(crossover, cross=cx_fnc[CROSS], cx_prob=CX_PROB)
    # mut = functools.partial(mutation, mut_prob=MUT_PROB, 
    #                         mutate=functools.partial(swap_mutate, max_len=MUT_MAX_LEN))
    mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                            mutate=functools.partial(mut_fnc[MUT], max_len=MUT_MAX_LEN))

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
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        best_template = '{individual}'
        with open('resources/kmltemplate.kml') as f:
            best_template = f.read()

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            f.write(str(bi))

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best.kml', 'w') as f:
            bi_kml = [f'{locations[i][1]},{locations[i][0]},5000' for i in bi]
            bi_kml.append(f'{locations[bi[0]][1]},{locations[bi[0]][0]},5000')
            f.write(best_template.format(individual='\n'.join(bi_kml)))
        
        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: difference = {fit(bi).objective}')

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