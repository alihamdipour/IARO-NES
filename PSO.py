import random
import numpy
import time
from  eqs import *



def PSO(fun_index, lb,ub ,dim, iters, PopSize, pop_pos, pop_fit, best_f, best_x):

    # PSO parameters

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("inf"))
    pBest = numpy.zeros((PopSize, dim))

    gBest = best_x
    gBestScore = best_f

    pos = pop_pos
    convergence_curve = [best_f]



    for l in range(0, iters):
        for i in range(0, PopSize):
             
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])

            fitness = ben_functions(pos[i, :],fun_index)

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax

                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve.append(gBestScore)

        if l % 1 == 0:
            print(["At iteration " + str(l + 1) + " the best fitness is " + str(gBestScore)])

    return gBest, gBestScore, convergence_curve