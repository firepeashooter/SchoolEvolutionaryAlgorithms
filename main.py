import random

#We want this function to generate a random permutation of a sequence 1-8 in an array
def generate_individual():

    individual = []

    while len(individual) < 8:

        rand_int = random.randint(1, 8)

        if rand_int in individual:
            continue
        else:
            individual.append(rand_int);

    return individual


#Generates an initial population of size n
#Returns an array of individuals
def generate_initial_population(n):
    population = []

    for i in range(n):
        individual = generate_individual()
        population.append(individual)

    return population



#Checks the fitness of a certain individual
def check_fitness(individual):

    ####Could we also check for like bad queens individual instead of looking at pairs?

    #Look at every pair of queens
    #Count how many pairs are diagonally attacking each other

    n = len(individual)
    conflicts = 0

    for col1 in range(n):
        for col2 in range(col1 + 1, n):
            row1 = individual[col1]
            row2 = individual[col2]
            if abs(row1 - row2) == abs(col1 - col2):
                conflicts += 1;

    return conflicts

#Mutates a single individual by swapping two values 
def mutate(individual):

    #Generate two random number between 0 and 7 for the possible positions
    pos1 = random.randint(0, 7)
    
    #Generates a random number that's not pos1
    pos2 = random.randint(0, 7)

    while pos2 == pos1:
        pos2 = random.randint(0, 7)

    print("positions:", pos1, pos2)

    #I love python
    individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

#Creates two new individuals via cross filling
def cross_over(individual1, individual2):

    offspring1 = []
    offspring2 = []

    crossover_point = random.randint(1, 7)

    individual1_first_half = individual1[0:crossover_point]
    individual2_first_half = individual2[0:crossover_point]

    individual1_last_half = individual1[crossover_point:]
    individual2_last_half = individual2[crossover_point:]

    #The first part of individual 1
    offspring1 += individual1_first_half
    #The first part of individual 2
    offspring2 += individual2_first_half
    
    #Last part of individual 2
    for num in individual2_last_half:
        if num not in offspring1:
            offspring1.append(num)

    #Add any remaining stuff from the first half that we need
    while len(offspring1) < 8:
        for num in individual2_first_half:
            if num not in offspring1:
                offspring1.append(num)

    for num in individual1_last_half:
        if num not in offspring2:
            offspring2.append(num)

    #Add any remaining stuff from the first half that we need
    while len(offspring2) < 8:
        for num in individual1_first_half:
            if num not in offspring2:
                offspring2.append(num)

    return offspring1, offspring2



def select_parents(population, pool, best):

    pop_size = len(population)
    #Select Parents (Best "best" out of random "pool")
    parents = []

    #Generate 5 random parents
    indices = random.choices(range(pop_size), k=pool)
    for index in indices:
        parents.append(population[index])

    #Evaluate their fitness and find the two best
    top_parents = sorted(parents, key=check_fitness)[:best]

    return top_parents




def main():

    #Initialization
    population_size = 100
    population = generate_initial_population(population_size)

    generations = 1000


    print(f"Initializing Population of size {population_size} for {generations} generations")
    print("--------------------")

    for i in range(generations):    
        print(f"BEGINNING GENERATION {i}")
        print("--------------------")
        #Parent Selection
        top_parents = select_parents(population, 5, 2)
        print(f"Top 2 parents picked from 5 random ones are: {top_parents[0]} and {top_parents[1]}")
        print(f"Fitness for the respective parents are: ")
        print()
        print(f"Fitness for Parent 1: {check_fitness(top_parents[0])}")
        print(f"Fitness for Parent 2: {check_fitness(top_parents[1])}")
        print("--------------------")

        
        #Crossover Parents
        offspring = cross_over(top_parents[0], top_parents[1])
        print("Crossing over Parents and generating offspring...")
        print(f"Offspring are: {offspring[0]} and {offspring[1]}")
        print("--------------------")



        #Mutate
        mutation_rate = 0.8
        print(f"Mutation rate is {mutation_rate} mutating offspring.....")

        for child in offspring:
            if random.random() < mutation_rate:
                print("Mutated: ", child)
                mutate(child)
        print("--------------------")



        print("Offspring and Parents added back to population, worst 2 populi removed")
        #Survival Selection (delete the worst 2 guys)
        population.sort(key=check_fitness, reverse=True)
        population = population[2:]

        for child in offspring:
            population.append(child)
        print("--------------------")



        best = 20
        print("--------------------")
        print(f"GENERATION COMPLETE: CHOOSING THE BEST {best} solutions")
        print("--------------------")

        top_ten = sorted(population, key=check_fitness)[:best]

        for ind in top_ten:
            print("--------------------")
            print(f"Parent: {ind} with a fitness of {check_fitness(ind)}")
            print("--------------------")

main()
