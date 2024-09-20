import random
import numpy
import math
import time
from  eqs import *


def WOA(fun_index, lb,ub ,dim, Max_iter, SearchAgents_no, pop_pos, pop_fit, best_f, best_x):

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Leader_pos = best_x
    Leader_score = best_f
    Positions = pop_pos

    convergence_curve =[best_f]

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):


            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            fitness = ben_functions(Positions[i, :],fun_index)

            # Update the leader
            if fitness < Leader_score:  # Change this to > for maximization problem
                Leader_score = fitness
                # Update alpha
                Leader_pos = Positions[
                    i, :
                ].copy()  # copy current whale position into the leader position

        a = 2 - t * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0 in Eq. (2.3)

        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / Max_iter)

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            r1 = np.random.rand()  # r1 is a random number in [0,1]
            r2 = np.random.rand()  # r2 is a random number in [0,1]

            A = 2 * a * r1 - a  # Eq. (2.3) in the paper
            C = 2 * r2  # Eq. (2.4) in the paper

            b = 1
            #  parameters in Eq. (2.5)
            l = (a2 - 1) * np.random.rand() + 1  #  parameters in Eq. (2.5)

            p = np.random.rand()  # p in Eq. (2.6)

            for j in range(0, dim):

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(
                            SearchAgents_no * np.random.rand()
                        )
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand

                    elif abs(A) < 1:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader

                elif p >= 0.5:

                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    # Eq. (2.5)
                    Positions[i, j] = (
                        distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                        + Leader_pos[j]
                    )

        convergence_curve.append(Leader_score)
        if t % 1 == 0:
            print(
                ["At iteration " + str(t) + " the best fitness is " + str(Leader_score)]
            )
        t = t + 1

    return Leader_pos, Leader_score, convergence_curve