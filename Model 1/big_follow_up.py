import matplotlib.pyplot as plt
import matplotlib.animation as animation
from classes import *
from parameters import *

## Paramters
# Measures = clustering, size of groupes...
#nbr_measures = 1

direction = "big_simu/"+sys.argv[1]+"/"

## Figure
#fig,ax= plt.subplots(nbr_measures, sharex=True, figsize=(10,4))
fig,ax= plt.subplots(nt, sharex=True, figsize=(10,15))

## Functions

def read(dir,time):
    pickleFile = open(dir + 'time' + str(time).rjust(nbr_size,"0") + '.pkl',"rb")
    pop_lue = pickle.load(open(dir + 'time' + str(time).rjust(nbr_size,"0") + '.pkl',"rb"))
    return(pop_lue)

## Main

def make_stats():
    clusts = []
    for th in thresholds:
        clust_per_thr = []
        for k in range(nbr_simus):
            clust_per_simu = []
            dir = direction+'Threshold_' + str(th).rjust(nbr_thresholds_size,"0") + "/Simu_" + str(k).rjust(nbr_simu_size,"0") + '/'
            for t in range(max_time+1):
                pop_lue = read(dir,t)
                clust_per_simu.append(pop_lue.clustering_coef())
            clust_per_thr.append(clust_per_simu)
        clusts.append(clust_per_thr)
    return(clusts)

def make_plot(clusts) :
    for i in range(nt):
        threshold = thresholds[i]
        clust_th = clusts[i]
        ax[i].set_ylabel("Threshold: "+f'{threshold}')
        ax[i].set_ylim(0,1.1)
        #ax.set_xlim(0,max_time)
        #ax.legend()
        for k in range(nbr_simus):
            clust_simu = clust_th[k]
            ax[i].plot(clust_simu)

    ax[nt-1].set_xlabel('Time')

    fig.savefig(direction+'big_follow_up.pdf')

# execute from shell
if __name__=="__main__":
    clusts = make_stats()
    make_plot(clusts)
