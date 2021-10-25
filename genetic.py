"""Grupo:
Carlos Gabriel
Isaque Almeida
Luis Fernando
"""
from typing import List, Callable, Tuple
from collections import namedtuple
import random
from functools import partial

# Quantidade de calorias desejada por dia
max_calories = 2500

"""
    Genome -- Cria um individuo
    Population -- Conjunto de individuos
    FitnessFunction -- Calcula o 'score' de um individuo
    PopulateFunction -- Gera aleatoriamente a população
    SelectionFunc -- Seleciona dois individuos em uma população
    CrossoverFunc -- Gera os filhos de dois individuos
    MutationFunc -- Gera uma mutaçao aleatoria em algum individuo
"""
Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
Food = namedtuple('Food', ['name', 'calories'])

foods = [
    Food('Double Quarter', 742),
    Food('Pão Integral 100g', 246),
    Food('Cereal Flocos de Milho 100g', 357),
    Food('Salada Verde com Abacate 250g', 135),
    Food('Batata Frita 200g', 321),
    Food('Creme de Papaia 150g', 177),
    Food('Leite Semi-Desnatado 100g', 42),
    Food('Frango grelhado 200g', 318),
    Food('Torta de Legumes 250g', 224),
    Food('Ovo Cozido 2un', 143),
    Food('Queijo Branco 100g', 345),
    Food('Coca-Cola 300ml', 313),
    Food('Vitamina de Banana 250g', 288),
    Food('Arroz Integral 200g', 222),
    Food('Uva-passa 100g', 299),
    Food('Abacaxi 100g', 50),
    Food('Sopa de Vegetais 500g', 159)
]


def generate_genome(length: int) -> Genome:
    """Gera um individuo

    Keyword arguments:
    length -- tamanho do individuo a ser gerado
    """
    return random.choices([0,1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    """
    size: Quantidade de soluções a ser gerada
    genome_length: Tamanho de cada solução
    """
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, foods: List[Food], calories_limit: int) -> int:
    """Calcula a pontuação de cada individuo

    Keyword arguments:
    genome -- individuo de uma população
    foods -- lista de alimentos
    calories_limit -- limite de calorias
    """
    if len(genome) != len(foods):
        raise ValueError("Genome and Food must be of the same length!")

    total_cal = 0

    for i, food in enumerate(foods):
        if genome[i] == 1:
            total_cal += food.calories

            if total_cal > calories_limit:
                return 0

    return total_cal


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """Seleciona dois individuos de uma população
    
    Keyword arguments:
    population -- população onde os individuos serão escolhidos
    fitness_func -- função de fitness, usada para calcular a pontuação de cada
    individuo para definir quem será escolhido (a escolha é aleatoria, porém quem
    possuir maior pontuação terá maior chance de ser escolhido)
    """
    return random.choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """Gera filhos de dois individuos

    Keyword arguments:
    a -- individuo 1
    b -- individuo 2
    """
    if len(a) != len(b):
        raise ValueError("a and b must be of the same length!")
    
    length = len(a)
    if length < 2:
        return a, b

    p = random.randint(1, length-1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    """Gera uma mutação aleatoria em um individuo
    
    Keyword arguments:
    genome -- individuo a ser mutado
    num -- numero de vezes que a mutação ira ocorrer
    probability -- probabilidade da mutação ocorrer em %
    """
    for _ in range(num):
        index = random.randrange(len(genome))
        genome[index] = genome[index] if random.random() > probability else abs(genome[index] - 1)

    return genome


def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    """Inicia o processo de evolução

    Keyword arguments:
    populate_func -- função que gera a população
    fitness_func -- calcula a pontuação de um individuo
    fitness_limit -- pontuação maxima esperada de um individuo
    selection_func -- seleciona dois individuos de uma população
    crossover_func -- cria dois filhos de uma população
    mutation_func -- gera uma mutação em um individuo
    generation_limit -- limite de gerações
    """
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

        if fitness_func(population[0]) == fitness_limit:
            break

        next_generation = population[0:4]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
    
    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )

    return population, i


population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=14, genome_length=len(foods)
    ),
    fitness_func=partial(
        fitness, foods=foods, calories_limit=max_calories
    ),
    fitness_limit=2500,
    generation_limit=1000
)


def genome_to_foods(genome: Genome, foods: List[Food]) -> List[Food]:
    """Função auxiliar para converter os resultados finais nas
    subsequentes comidas com base em seu id.
    """
    result = []
    total = 0
    for i, gene in enumerate(genome):
        if gene == 1:
            result.append(foods[i].name)
            total += foods[i].calories

    return result, total


print(f"Number of generations: {generations}")
results = genome_to_foods(population[0], foods)
print(f"Best solution: {results[0]}")
print(f"Calories: {results[1]}")
