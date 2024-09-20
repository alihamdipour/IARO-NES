import math
import numpy
import random
import time
from  eqs import *


def BAT(fun_index,lb,ub,dim ,Max_iteration, N, Positions, pop_fit, best_f,best_x):

    n = N
    # Population size

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    N_gen = Max_iteration  # Number of generations

    A = 0.5
    # Loudness  (constant or decreasing)
    r = 0.5
    # Pulse rate (constant or decreasing)

    Qmin = 0  # Frequency minimum
    Qmax = 2  # Frequency maximum

    d = dim  # Number of dimensions

    # Initializing arrays
    Q = numpy.zeros(n)  # Frequency
    v = numpy.zeros((n, d))  # Velocities
    Convergence_curve = [best_f]
    Sol = Positions
    Fitness = pop_fit
    best = best_x
    fmin = best_f

    S = numpy.zeros((n, d))
    S = numpy.copy(Sol)
    # Main loop
    for t in range(0, N_gen):

        # Loop over all bats(solutions)
        for i in range(0, n):
            Q[i] = Qmin + (Qmin - Qmax) * np.random.rand()
            v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
            S[i, :] = Sol[i, :] + v[i, :]

            # Check boundaries
            for j in range(d):
                Sol[i, j] = numpy.clip(Sol[i, j], lb[j], ub[j])

            # Pulse rate
            if np.random.rand() > r:
                S[i, :] = best + 0.001 * numpy.random.randn(d)

            # Evaluate new solutions
            Fnew = ben_functions(S[i, :],fun_index)

            # Update if the solution improves
            if (Fnew <= Fitness[i]) and (np.random.rand() < A):
                Sol[i, :] = numpy.copy(S[i, :])
                Fitness[i] = Fnew

            # Update the current best solution
            if Fnew <= fmin:
                best = numpy.copy(S[i, :])
                fmin = Fnew

        # update convergence curve
        Convergence_curve.append(fmin)

        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(fmin)])

    return best, fmin, Convergence_curve