"""Grupo:
Carlos Gabriel
Isaque Almeida
Luis Fernando
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
from time import time

CALORIES = 7683

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
    ('Pastel de queijo', 301),
    ('Pizza Quatro Queijos 1 fatia', 420),
    ('Nhoque 100g', 364),
    ('Coxinha 150g', 274),
    ('Abacaxi 100g', 50),
    ('Sorvete 100g', 207),
    ('Lasanha 100g', 135),
    ('Picanha acebolada', 1200),
    ('Feijoada 100g', 305),
    ('Estrogonofe 100g', 98),
    ('Sopa de Vegetais 500g', 159)]
)

def generate_chromosome(length):
    return random.choices([0,1], k=length)

def generate_population(size, chromosome_length):
    return ([generate_chromosome(chromosome_length) for _ in range(size)])

def gen_pop_w_limit(size, chromosome_length, minimum_distance):
    """Gera a população com cada individuo estando a no minimo @minimum_distance
    do limite de @CALORIES.

    size: tamanho da população
    chromosome_length: tamanho de cada individuo da população
    minimum_distance: distancia minima que a pontuação do individuo deve estar
                      do numero maximo de calorias.
    """
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

def fitness(chromosome, score_or_value=0):
    chromosome_value = sum([foods[x][1] for x, y in enumerate(chromosome) if y > 0])

    score = abs(CALORIES - chromosome_value)

    if score_or_value == 1:
        return chromosome_value

    return score

def full_population_fitness(population, score_or_value=0):
    if score_or_value == 1:
        return ([fitness(chromosome, score_or_value=1) for chromosome in population])
    return ([fitness(chromosome) for chromosome in population])

def selection_pair(population, pop_scores):
    sp = []

    while len(sp) < 2:
        n1 = n2 = 0
        while n1 == n2:
            n1 = random.randint(0, len(population) - 1)
            n2 = random.randint(0, len(population) - 1)

        if pop_scores[n1] >= pop_scores[n2]:
            sp.append(population[n1])
        else:
            sp.append(population[n2])

    return sp

def single_point_crossover(a, b):
    if len(a) != len(b):
        raise ValueError("Genome 'a' and 'b' must be of the same length!")

    length = len(a)
    if length < 2:
        return a, b

    p = random.randint(0, length-1)

    offspring_a = a[:p] + b[p:]
    offspring_b = b[:p] + a[p:]

    return offspring_a, offspring_b

def mutation(chromosome, probability = 0.25):
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

        worst.append(fitness( sorted_pop[-1]))
        average.append(fitness( sorted_pop[ len(sorted_pop) // 2 ] ))
        best.append(fitness( sorted_pop[0] ))

        if best[len(best) - 1] < 1:
            break

        new_generation = sorted_pop[:2]
        fpv = full_population_fitness(sorted_pop, 1)
        # new_generation = []

        for _ in range( int(population_size / 2) - 1 ):
        # for _ in range( population_size ):
            parents = selection_pair(pop, fpv)
            off_a, off_b = single_point_crossover(parents[0], parents[1])
            off_a = mutation(off_a)
            off_b = mutation(off_b)
            new_generation.append(off_a)
            new_generation.append(off_b)

        pop = new_generation

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
g, pop, best, average, worst = run(generation_limit=1000, population_size=len(foods)*2)
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
plt.ylabel(f"Distance to {CALORIES}kcal\n(lower is better)")
plt.xlabel("Number of generations")
plt.legend()
plt.show()

print(f"Elapsed: {e-s}")
