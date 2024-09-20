import numpy as np
from matplotlib.pyplot import *
from pylab import *
import timeit
import os
from copy import deepcopy
import arabic_reshaper
from bidi.algorithm import get_display
from IARO import *
from ARO import *
from GWO import *
from BAT import *
from WOA import *
from PSO import *
from GA import *
from CS import *
from eqs import *

def main():
    folder='cyclik1'
    max_it = 1000
    max_repeat=5
    npop = 100
    createResultPath(folder)
    algorthims=['ARO','GA','WOA','GWO','BAT','CS','IARO']
    #algorthims=['GA','WOA','BAT','ARO','GWO',,'PSO']
    # eqs=['a','b','c','d','e','f','g','h','i','j','k','l','m']
        

    for fun_index in arange(11,12):
        avg_result=[[[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]],
                    [[],[],[],[]]]
        lb,ub,dim=fun_range(fun_index)
        pop_pos = np.zeros((npop, dim))
        for mr in range(max_repeat):
            for i in range(dim):
                pop_pos[:, i] = np.random.rand(npop) * (ub[i] - lb[i]) + lb[i]
            pop_fit = np.zeros(npop)
            for i in range(npop):
                pop_fit[i] = ben_functions(pop_pos[i, :], fun_index)
            best_f = float('inf')
            best_x = []
            for i in range(npop):
                if pop_fit[i] <= best_f:
                    best_f = pop_fit[i]
                    best_x = pop_pos[i, :]
    
            for al in range(len(algorthims)):    
                start = timeit.default_timer()     
                function = globals()[algorthims[al]]
                r_best_x, r_best_f, his_best_fit = function(fun_index,lb,ub ,dim,max_it, npop, deepcopy(pop_pos), deepcopy(pop_fit), deepcopy(best_f), deepcopy(best_x))
                stop = timeit.default_timer()
                avg_result[al][0].append(r_best_x)
                avg_result[al][1].append(r_best_f)
                avg_result[al][2].append(his_best_fit)
                avg_result[al][3].append(stop-start)

        f_fit = open(folder+"/fit.txt", "a")
        f_bestX=open(folder+"/bestX.txt", "a")
        f_time=open(folder+"/time.txt", "a")
        f_cave=open(folder+'/caves/'+str(fun_index)+'.txt', "a")
        f_fit.write(str(fun_index)+',')
        f_bestX.write(str(fun_index)+',')
        f_time.write(str(fun_index)+',')

        for i in range(len(algorthims)):
            f_bestX.write(str(avg_result[i][0]) +'\n')
            f_fit.write(str(np.average(avg_result[i][1]))+',')
            f_time.write(str(np.average(avg_result[i][3]))+',')
            f_cave.write (str(np.average(avg_result[i][2], axis=0)) +'\n')

        f_fit.write('\n')
        f_bestX.write('\n')
        f_cave.write('\n')
        f_fit.close
        f_bestX.close
        f_time.write('\n')
        f_cave.close

        color_labels=['b','darkorange','brown','c','r','m','g','teal','y','g']
        for i in range(len(algorthims)):
            print(algorthims[i])
            plot(arange(0,max_it+1), np.average(avg_result[i][2], axis=0), color_labels[i], label=algorthims[i])              

        yscale('log')        
        plt.xlim([0, max_it + 1])
        plt.xlabel(get_display(arabic_reshaper.reshape(u'%s' % str("تکرار") )))
        plt.ylabel(get_display(arabic_reshaper.reshape(u'%s' % str('خطای برازش'))),rotation=90)
        plt.title(str(fun_index))       
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [6,0,1,2,3,4,5]
        #legend()
        legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        savefig(folder+'\images\\'+str(fun_index)+'.png')
        clf()
def createResultPath(path):
    resultPath =  path
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    if not os.path.exists(resultPath+'/images'):
        os.makedirs(resultPath+'/images')
    if not os.path.exists(resultPath+'/caves'):
        os.makedirs(resultPath+'/caves')
if __name__ == '__main__':
    main()
