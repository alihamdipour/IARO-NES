
import numpy
import random
from eqs import *
import time
import sys


def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    # initialize a new population
    newPopulation = numpy.empty_like(population)
    newPopulation[0:keep] = population[0:keep]
    # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
    for i in range(keep, popSize, 2):
        # pair of parents selection
        parent1, parent2 = pairSelection(population, scores, popSize)
        crossoverLength = min(len(parent1), len(parent2))
        parentsCrossoverProbability = np.random.uniform(0.0, 1.0)
        if parentsCrossoverProbability < crossoverProbability:
            offspring1, offspring2 = crossover(crossoverLength, parent1, parent2)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        # Add offsprings to population
        newPopulation[i] = numpy.copy(offspring1)
        newPopulation[i + 1] = numpy.copy(offspring2)

    return newPopulation


def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
    for i in range(keep, popSize):
        # Mutation
        offspringMutationProbability = np.random.uniform(0.0, 1.0)
        if offspringMutationProbability < mutationProbability:
            mutation(population[i], len(population[i]), lb, ub)


def elitism(population, scores, bestIndividual, bestScore):
    worstFitnessId = selectWorstIndividual(scores)

    # replace worst cromosome with best one from previous generation if its fitness is less than the other
    if scores[worstFitnessId] > bestScore:
        population[worstFitnessId] = numpy.copy(bestIndividual)
        scores[worstFitnessId] = numpy.copy(bestScore)


def selectWorstIndividual(scores):
    maxFitnessId = numpy.where(scores == numpy.max(scores))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId


def pairSelection(population, scores, popSize):
    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = rouletteWheelSelectionId(scores, popSize)
    parent2 = population[parent2Id].copy()

    return parent1, parent2


def rouletteWheelSelectionId(scores, popSize):
    ##reverse score because minimum value should have more chance of selection
    reverse = max(scores) + min(scores)
    reverseScores = reverse - scores.copy()
    sumScores = sum(reverseScores)
    pick = np.random.uniform(0, sumScores)
    current = 0
    for individualId in range(popSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId


def crossover(individualLength, parent1, parent2):

    # The point at which crossover takes place between two parents.
    crossover_point = np.random.randint(0, individualLength - 1)
    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
    offspring1 = numpy.concatenate(
        [parent1[0:crossover_point], parent2[crossover_point:]]
    )
    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
    offspring2 = numpy.concatenate(
        [parent2[0:crossover_point], parent1[crossover_point:]]
    )

    return offspring1, offspring2


def mutation(offspring, individualLength, lb, ub):

    mutationIndex = np.random.randint(0, individualLength - 1)
    mutationValue = np.random.uniform(lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue


def clearDups(Population, lb, ub):

    newPopulation = numpy.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newPopulation = numpy.append(
            newPopulation,
            numpy.random.uniform(0, 1, (nDuplicates, len(Population[0])))
            * (numpy.array(ub) - numpy.array(lb))
            + numpy.array(lb),
            axis=0,
        )

    return newPopulation


def calculateCost(fun_index, population, popSize, lb, ub):

    scores = numpy.full(popSize, numpy.inf)

    # Loop through individuals in population
    for i in range(0, popSize):
        # Return back the search agents that go beyond the boundaries of the search space
        population[i] = numpy.clip(population[i], lb, ub)

        # Calculate objective function for each search agent
        scores[i] = ben_functions(population[i, :],fun_index)

    return scores


def sortPopulation(population, scores):

    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]

    return population, scores


def GA(fun_index, lb,ub ,dim, iters, popSize, pop_pos, pop_fit, best_f, best_x):

    cp = 1  # crossover Probability
    mp = 0.01  # Mutation Probability
    keep = 2
    # elitism parameter: how many of the best individuals to keep from one generation to the next


    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = best_x
    scores = pop_fit
    bestScore =best_f

    ga = pop_pos
    convergence_curve = [best_f]


    for l in range(iters):

        # crossover
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)

        # mutation
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)

        ga = clearDups(ga, lb, ub)

        scores = calculateCost(fun_index, ga, popSize, lb, ub)

        bestScore = min(scores)

        # Sort from best to worst
        ga, scores = sortPopulation(ga, scores)

        convergence_curve.append(bestScore)

        if l % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(bestScore)
                ]
            )

    return bestIndividual, bestScore, convergence_curve