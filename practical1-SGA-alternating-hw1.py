
import random
import numpy as np

IND_LENGTH = 25
POP_COUNT = 100
GEN_COUNT = 100
SEL_COUNT = 50
MUT_RATE = 1 / IND_LENGTH

ZERO_START_IND = [ x % 2 for x in range(IND_LENGTH)]
ONE_START_IND  = [ 1 - x for x in ZERO_START_IND]

def get_target_individuals():
    return [ZERO_START_IND, ONE_START_IND]

def fitness_function(individual):
    comparisons = [[int(individual_value == target_value) for individual_value, target_value in zip(individual, target) ] for target in get_target_individuals()]
    return max([sum(comparison) for comparison in comparisons])

def selection(population, fitness_values, selection_count=None):
    probs = [fitness / sum(fitness_values) for fitness in fitness_values]
    return random.choices(population, probs, k=POP_COUNT)

def cross(individual1, individual2):
    crossover_point = random.randint(0, IND_LENGTH - 1)
    
    return [individual2[:crossover_point] + individual1[crossover_point:],
            individual1[:crossover_point] + individual2[crossover_point:]]

def mutate(individual):

    return [1 - gene if random.random() < MUT_RATE else gene for gene in individual]

def mutation(population):
    return [mutate(individual) for individual in population]

def create_individual():
    return [random.randint(0, 1) for _ in range(IND_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POP_COUNT)]

def evolutionary_algorithm(population):

    best_individual, best_fitness = None, 0
    for gen in range(GEN_COUNT):
        fitness_values = [fitness_function(individual) for individual in population]
        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > best_fitness:
            best_individual = population[best_index]
            best_fitness = fitness_values[best_index]


        mating_pool = selection(population, fitness_values, selection_count=SEL_COUNT)

        for ind1, ind2 in zip(mating_pool[::2], mating_pool[1::2]):
            population.extend(cross(ind1, ind2))

        o = mutation(population)
        
        population = o[:-1] + [best_individual] if best_individual else o[:]
    return population

if __name__ == '__main__':
    population = create_population()
    init_fitness_values = [fitness_function(individual) for individual in population]
    print(max(init_fitness_values))
    population = evolutionary_algorithm(population)
    fitness_values = [fitness_function(individual) for individual in population]
    print(max(fitness_values))

        

        