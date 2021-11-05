"""Grupo:
Carlos Gabriel
Isaque Almeida
Luis Fernando
"""
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np

CALORIES = 2500

foods = (
    [('Pão Integral 100g', 246),
    ('Cereal Flocos de Milho 100g', 357),
    ('Salada Verde com Abacate 250g', 135),
    ('Batata Frita 200g', 321),
    ('Creme de Papaia 150g', 177),
    ('Big Mac', 740),
    ('Leite Semi-Desnatado 100g', 42),
    ('Frango grelhado 200g', 318),
    ('Torta de Legumes 250g', 224),
    ('Ovo Cozido 2un', 143),
    ('Queijo Branco 100g', 345),
    ('Coca-Cola 300ml', 313),
    ('Vitamina de Banana 250g', 288),
    ('Arroz Integral 200g', 222),
    ('Uva-passa 100g', 299),
    ('Abacaxi 100g', 50),
    ('Sorvete 100g', 207),
    ('Lasanha 100g', 135),
    ('Feijoada 100g', 305),
    ('Estrogonofe 100g', 98),
    ('Sopa de Vegetais 500g', 159)]
)

def generate_chromosome(length):
    return random.choices([0,1], k=length)

def generate_population(size, chromosome_length):
    return ([generate_chromosome(chromosome_length) for _ in range(size)])

def gen_pop_w_limit(size, chromosome_length, minimum_distance):
    pop = generate_population(size, chromosome_length)
    pop_s = full_population_fitness(pop)
    print(pop_s)

    while True:
        for i, score in enumerate(pop_s):
            if score < minimum_distance:
                pop[i] = generate_chromosome(len(foods))
                pop_s = full_population_fitness(pop)
        
        pop_s = full_population_fitness(pop)
        x = 0
        for i, score in enumerate(pop_s):
            if score < minimum_distance:
                x += 1
        if x == 0:
            break

    pop_s = full_population_fitness(pop)
    
    return pop


def fitness(chromosome):
    """Calcula a pontuação de um cromossomo baseado na distancia que ele se
       encontra da resposta esperada.
    """
    chromosome_value = sum([foods[x][1] for x, y in enumerate(chromosome) if y > 0])

    score = abs(CALORIES - chromosome_value)

    return score

def full_population_fitness(population):
    """Retorna um array com a pontuação da população inteira na mesma ordem."""
    return ([fitness(chromosome) for chromosome in population])

def selection_pair(population, pop_scores):
    """Cria uma roleta para escolher aleatoriamente dois individuos da população"""
    while True:
        sp = random.choices(
            population=population,
            k=2,
            weights=pop_scores)

        if sp[0] != sp[1]:
            return sp
        else:
            continue 

def single_point_crossover(a, b):
    if len(a) != len(b):
        raise ValueError("Genome 'a' and 'b' must be of the same length!")

    length = len(a)
    if length < 2:
        return a, b

    p = random.randint(1, length-1)

    offspring_a = a[:p] + b[p:]
    offspring_b = b[:p] + a[p:]

    return offspring_a, offspring_b

def mutation(chromosome, probability = 0.05):
    r = np.random.random_sample(size=None)

    if r <= probability:
        index = np.random.randint(low=0,high=(len(chromosome) - 1))
        chromosome[index] = abs(chromosome[index] - 1)

    return chromosome

def run(generation_limit = 100, population_size = 100):
    worst = []
    average  = []
    best  = []

    foods_len = len(foods)
    pop = (gen_pop_w_limit(population_size, foods_len, 1500))


    for g in range(generation_limit):
        sorted_pop = sorted(pop, key=lambda chromosome: fitness(chromosome))
        fpf = full_population_fitness(sorted_pop)
        print(sorted_pop)

        worst.append(fitness( sorted_pop[-1]))
        average.append(fitness( sorted_pop[ len(sorted_pop) // 2 ] ))
        best.append(fitness( sorted_pop[0] ))

        if best[len(best) - 1] < 1:
            break

        new_generation = sorted_pop[:2]
        off_a, off_b = single_point_crossover(new_generation[0], new_generation[1])
        off_a = mutation(off_a)
        off_b = mutation(off_b)

        print(new_generation)
        print(type(new_generation))
        print(type(new_generation[0]))

        new_generation.append(off_a)
        new_generation.append(off_b)
        print(f"New Generation: {new_generation}")

        for _ in range( int(population_size / 2) - 1 ):
            parents = selection_pair(pop, fpf)
            off_a, off_b = single_point_crossover(parents[0], parents[1])
            off_a = mutation(off_a)
            off_b = mutation(off_b)
            new_generation.append(off_a)
            new_generation.append(off_b)

        pop = new_generation

        print(f"WORST: {fitness( sorted_pop[-1] )}")
        print(f"AVERAGE: {fitness( sorted_pop[ len(sorted_pop) // 2 ] )}")
        print(f"BEST: {fitness( sorted_pop[0] )}")

    return g, sorted_pop, best, average, worst

def chromosome_to_foods(chromosome):
    result = []
    total = 0
    for i, gene in enumerate(chromosome):
        if gene == 1:
            result.append(foods[i][0])
            total += foods[i][1]

    return result, total

s = time()
g, pop, best, average, worst = run(generation_limit=40000, population_size=len(foods)*2)
e = time()

print(f"GENERATIONS: {g}")
print(f"BEST: {best}")
print(f"AVERAGE: {average}")
print(f"WORST: {worst}")
print(full_population_fitness(pop))

result, total = chromosome_to_foods(pop[0])
print(result)
print(f"Total de calorias: {total}")

generations_arr = [x for x in range(g + 1)]

plt.plot(generations_arr, best, label="best")
plt.plot(generations_arr, average, label="average")
plt.plot(generations_arr, worst, label="worst")
plt.ylabel("Distance to 2500kcal\n(lower is better)")
plt.xlabel("Number of generations")
plt.legend()
plt.show()
print(f"Elapsed: {e-s}")
