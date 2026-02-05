"""
CISC455/851 coding practice - Evolution Strategy with one mutation_step_size self-adaptation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    Computes the Ackley function for an n-dimensional input vector x.

    Parameters:
        x (np.array): Input vector of shape (n,).
        a (float): Parameter controlling the depth of the basin (default: 20).
        b (float): Parameter controlling the width of the basin (default: 0.2).
        c (float): Parameter controlling the frequency of the cosine term (default: 2π).

    Returns:
        float: The value of the Ackley function at x.
    """
    n = len(x)
    sum_sq = np.sum(x ** 2)  # Sum of squares
    sum_cos = np.sum(np.cos(c * x))  # Sum of cosines

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.exp(1)


def plot_ackley(trajectory=None):
    # creating a linspace for graphing
    x = np.linspace(-32, 32, 80)
    y = np.linspace(-32, 32, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley(np.array([xi, yi])) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet')

    # plotting the trajectory of the best solution throughout the run (only really works for dim=2)
    if trajectory is not None:
        trajectory = np.array([x[0] for x in trajectory])
        trajectory_ackley = np.array([x[1] for x in trajectory])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory_ackley, color='g', marker='o')

    plt.title("Ackley Function Landscape")
    plt.show()


def initialize_population(mu, dim, bounds, sigma_init):
    """
    Initializes the population

    Parameters:
        mu (int): Number of individuals in population.
        dim (int): Dimension of each individual.
        bounds (float, float): Min and max value that an individual can contain.
        sigma_init (float): Initial value of sigma (mutation step size).

    Returns:
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
    """
    # generate individuals as np.arrays of size (mu,dim) with values between the given bounds
    population = np.random.uniform(bounds[0], bounds[1], (mu, dim))
    # same for the sigma values (one sigma per individual)
    sigma = np.full((mu,), sigma_init)
    return population, sigma


def mutate(parent, parent_sigma, dim, bounds, lr, sigma_bound):
    """
    Generate a mutated child using Gaussian perturbation and self-adaptation.

    Parameters:
        parent (np.array): A single solution.
        parent_sigma (float): Sigma corresponding to the parent.
        dim (int): Dimension of each individual.
        bounds (float, float): Min and max value that an individual can contain.
        lr (float): learning rate.
        sigma_bound (float): Lower boundary for sigma.

    Returns:
        child (np.array): A new mutated solution.
        child_sigma: Sigma corresponding to the child.
    """
    # student code begins
    child_sigma = parent_sigma * (math.exp(lr * np.random.lognormal()))

    child = []

    for i in range(len(parent)):
        child.append(parent[i] * child_sigma * np.random.lognormal())

    # student code ends

    return child, child_sigma


def generate_offspring(population, sigma, mu, lambd, dim, bounds, lr, sigma_bound, ackley):
    """
    Generate offspring from the current population.

    Parameters:
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
        mu (int): Number of individuals in population.
        lambd (int): Number of offspring to be created.
        dim (int): Dimension of each individual.
        bounds (float, float): Min and max value that an individual can contain.
        lr (float): learning rate.
        sigma_bound (float): Lower boundary for sigma.
        ackley (func): The function to be optimized.

    Returns:
        offspring (np.array): Generated offspring. A list of lambd lists of dim floats.
    """
    offspring = []
    for _ in range(lambd):
        parent_idx = np.random.randint(mu)  # Random parent selection
        # generate the child and its sigma
        child, child_sigma = mutate(population[parent_idx], sigma[parent_idx], dim, bounds, lr, sigma_bound)
        fitness = ackley(child)  # Evaluate fitness
        offspring.append((child, child_sigma, fitness))
    return offspring


def select_survivors(offspring, mu, population, sigma, strategy='comma'):
    """
    Select the top mu individuals from offspring based on fitness.

    Parameters:
        offspring (np.array): Generated offspring. A list of lambd lists of dim floats.
        mu (int): Number of individuals in population.
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
        strategy (string): A flag that chooses (mu, lambd) or (mu + lambd). Only values are "comma" and "plus".

    Returns:
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
        best_solution (np.array): The solution with the best fitness.
        best_fitness (float): The best fitness.
    """
    if strategy == 'comma':

        # student code begins

        #Pick the best solutions from the children

        #To determine the fitness of a solution we call the ackley function
        temp_offspring = offspring.copy()

        temp_offspring = sorted(temp_offspring, key = ackley)

        population = temp_offspring[:mu]

        best_solution = temp_offspring[0]
        best_solution = ackley(best_solution)

        # student code ends

    elif strategy == 'plus':

        # student code begins
        population_offspring = population + offspring


        sorted_pool = sorted(population_offspring, key = ackley)

        population = sorted_pool[:mu]

        best_solution = sorted_pool[0]
        best_solution = ackley(best_solution)


        

        # student code ends


def evolution_strategy(ackley, dim=2, mu=30, lambd=200, sigma_init=3.0, bounds=(-30, 30), sigma_bound=0.01,
                       max_evals=100000):
    """
    Runs the (30,200) Evolution Strategy to minimize the Ackley function.

    Parameters:
        ackley (func): The function to be optimized.
        offspring (np.array): Generated offspring. A list of lambd lists of dim floats.
        mu (int): Number of individuals in population.
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
        strategy (string): A flag that chooses (mu, lambd) or (mu + lambd). Only values are "comma" and "plus".

    Returns:
        population (np.array): Initialized population. A list of mu lists of dim floats.
        sigma (np.array): Initialized sigmas. A single int which pairs with population via index.
        best_solution (np.array): The solution with the best fitness.
        best_fitness (float): The best fitness.
    """
    eval_count = 0
    # initialize population and sigmas and best solution
    population, sigma = initialize_population(mu, dim, bounds, sigma_init)
    best_solution, best_fitness = None, float('inf')
    trajectory = []  # for plotting
    lr = 1 / np.sqrt(dim)  # learning rate for mutating sigma

    # main loop
    while eval_count < max_evals:
        # generate lambda offspring

        # student code begins

        
        # student code ends


        # select mu survivors based on plus or comma strategy

        # student code begins


        # student code ends


        # Update best solution
        if new_best_fitness < best_fitness:
            best_solution, best_fitness = new_best_solution, new_best_fitness
            trajectory.append((best_solution, best_fitness))

        # for printing progress
        if eval_count % 1000 == 0:
            print(f"Eval count: {eval_count}, Best fitness: {best_fitness}, Best solution: {best_solution}")

        # termination condition
        if eval_count >= max_evals:
            break

    return best_solution, best_fitness, trajectory


def main():
    best_sol, best_fit, trajectory = evolution_strategy(ackley, dim=2)
    print(f"Best solution: {best_sol}")
    print(f"Best fitness: {best_fit}")
    print(trajectory)

    plot_ackley(trajectory)


# if __name__ == '__main__':
    # main()
