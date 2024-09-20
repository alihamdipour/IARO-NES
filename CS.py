import math
import numpy
import random
from eqs import *

def get_cuckoos(nest, best, lb, ub, n, dim):

    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.array(nest)
    beta = 3 / 2
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    s = numpy.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = numpy.random.randn(len(s)) * sigma
        v = numpy.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * numpy.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = numpy.clip(s[k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = ben_functions(newnest[j, :],objf)
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = numpy.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim):

    # Discovered or not
    tempnest = numpy.zeros((n, dim))

    K = numpy.random.uniform(0, 1, (n, dim)) > pa

    stepsize = np.random.rand() * (
        nest[numpy.random.permutation(n), :] - nest[numpy.random.permutation(n), :]
    )

    tempnest = nest + stepsize * K

    return tempnest


##########################################################################


def CS(fun_index, lb,ub ,dim, N_IterTotal, n, pop_pos, pop_fit, best_f, best_x):

    pa = 0.25
    nd = dim
    convergence = [best_f]

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # RInitialize nests randomely
    nest =  pop_pos

    new_nest = numpy.zeros((n, dim))
    new_nest = numpy.copy(nest)

    bestnest = best_x

    fitness = pop_fit

    fmin, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, fun_index)
    # Main loop counter
    for iter in range(N_IterTotal):
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, fun_index)

        new_nest = empty_nests(new_nest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, fun_index)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        if iter % 10 == 0:
            print(["At iteration " + str(iter) + " the best fitness is " + str(fmin)])
        convergence.append(fmin)

    return best_x, fmin, convergence