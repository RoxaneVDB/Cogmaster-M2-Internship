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
fig,ax= plt.subplots(1, sharex=True, figsize=(10,4))

## Functions

def average(l):
    n = len(l)
    s = sum(l)
    return(s/n)

def read(dir,time):
    pickleFile = open(dir + 'time' + str(time).rjust(nbr_size,"0") + '.pkl',"rb")
    pop_lue = pickle.load(open(dir + 'time' + str(time).rjust(nbr_size,"0") + '.pkl',"rb"))
    return(pop_lue)

## Main

def make_stats():
    clust_average = []
    prop_empty = []
    prop_quasi_empty = []
    for th in thresholds:
        clust = []
        nbr_empty = 0
        nbr_quasi_empty = 0
        for k in range(nbr_simus):
            dir = direction+'Threshold_' + str(th).rjust(nbr_thresholds_size,"0") + "/Simu_" + str(k).rjust(nbr_simu_size,"0") + '/'
            pop_lue = read(dir,max_time)

            clust.append(pop_lue.clustering_coef())

            if (pop_lue.nbr_connexions() == 0):
                nbr_empty += 1
            if (pop_lue.nbr_of_pairs() == 0):
                nbr_quasi_empty += 1

        prop_empty.append(nbr_empty/nbr_simus)
        prop_quasi_empty.append(nbr_quasi_empty/nbr_simus)
        clust_average.append(average(clust))
    return(clust_average,prop_empty,prop_quasi_empty)

def make_plot(clust_average,prop_empty,prop_quasi_empty) :

    trust = thresholds.tolist()
    clust_average.reverse()
    prop_empty.reverse()
    prop_quasi_empty.reverse()
    trust.reverse()

    ax.set_title("Average for "+f'{nbr_simus}'+" simulations")
    ax.set_ylabel("Clustering coefficient")
    #ax[1].set_ylabel("Mesure 2")
    #ax[2].set_ylabel("Mesure 3")

    ax.plot(trust,clust_average,'ro',label = "Average of clustering")


    ax.plot(trust,prop_quasi_empty,label = "Proportion of quasi collapse")
    ax.plot(trust,prop_empty,label = "Proportion of collapse")
    ax.set_xlim(max(thresholds),0) # changes nothing visible but keeps the thing coherent
    ax.set_ylim(0,1.1)
    ax.legend()

    ax.set_xlabel('Trust Threshold')



    fig.savefig(direction+'big_stats.pdf')

# execute from shell
if __name__=="__main__":
    clust_average,prop_empty,prop_quasi_empty = make_stats()
    make_plot(clust_average,prop_empty,prop_quasi_empty)
