import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *

from eqs import *

def space_bound(X, Up, Low):
    dim = len(X)
    S = (X > Up) + (X < Low)
    res = (np.random.rand(dim) * (np.array(Up) - np.array(Low)) + np.array(Low)) * S + X * (~S)
    return res

def IARO(fun_index, lb,ub ,dim, max_it, npop, pop_pos, pop_fit, best_f, best_x):


    visit_table = np.zeros((npop, npop))
    diag_ind = np.diag_indices(npop)
    visit_table[diag_ind] = float('nan')

    his_best_fit = [best_f]

    for it in range(max_it):
        visit_table[diag_ind] = float('-inf')
        direct1=np.zeros((npop, dim))
        direct2=np.zeros((npop, dim))
        theta = 2 * (1 - (it+1) / max_it)
        for i in range(npop):
            L = (np.e - np.exp((((it+1) - 1) / max_it) ** 2)) * (np.sin(2 * np.pi * np.random.rand())) # Eq.(3)
            rd = np.floor(np.random.rand() * (dim))
            rand_dim = randperm(dim)
            direct1[i, rand_dim[:int(rd)]] = 1
            c = direct1[i,:]  #Eq.(4)
            R = L * c # Eq.(2)
            A = 2 * np.log(1 / np.random.rand()) * theta #Eq.(15)
            if A>1:

                MaxUnvisitedTime = max(visit_table[i, :])
                TargetFoodIndex = visit_table[i, :].argmax()
                MUT_Index = np.where(visit_table[i, :] == MaxUnvisitedTime)
                if len(MUT_Index[0]) > 1:
                    Ind = pop_fit[MUT_Index].argmin()
                    TargetFoodIndex = MUT_Index[0][Ind]
                RandInd=TargetFoodIndex
              
                newPopPos = pop_pos[RandInd, :] + R * (pop_pos[i, :] - pop_pos[RandInd, :])+round(0.5 * (0.05 +np.random.rand())) * np.random.randn() # Eq.(1)
                newPopPos = space_bound(newPopPos, ub, lb)
                newPopFit = ben_functions(newPopPos, fun_index)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
            else: 

                ttt=int(np.floor(np.random.rand() * dim))
                direct2[i, ttt] = 1
                gr = direct2[i,:] #Eq.(12)
                H = ((max_it - (it+1) + 1) / max_it) * np.random.randn() # % Eq.(8)
                b = pop_pos[i,:]+H * gr * pop_pos[i,:] # % Eq.(13)
                newPopPos = pop_pos[i][:]+ R* (np.random.rand() * b - pop_pos[i][:]) #Eq.(11)

                newPopPos = space_bound(newPopPos, ub, lb)
                newPopFit = ben_functions(newPopPos, fun_index)
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos


            if pop_fit[i] < best_f:
               best_f = pop_fit[i]
               best_x = pop_pos[i, :]
        his_best_fit.append(best_f)
    return best_x, best_f, his_best_fit
