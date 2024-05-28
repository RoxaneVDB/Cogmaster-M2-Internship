import numpy as np
import pymetis
import random

def to_adj_list(g):
    (n,n_) = np.shape(g)
    res = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if g[i][j] == 1:
                (res[i]).append(j)
    return(res)

def partition(g,k):
    adj_list = to_adj_list(g)
    n_cuts, membership = pymetis.part_graph(k, adjacency=adj_list)
    return(n_cuts, membership)

def best_partition0(g,k_inf,k_max):
    n_cuts, membership = partition(g,k_max)
    best_ks = [k_max]
    for k in range(k_inf,k_max): # from k_inf to k_max-1 but k_max is already done
        print("k:",k)
        n_c, memb = partition(g,k)
        if n_c < n_cuts:
            print("oui: ", n_c)
            best_ks = [k]
            membership = memb
            n_cuts = n_c
        elif n_c == n_cuts:
            print("non: ", n_c)
            best_ks.append(k)
    print(best_ks)
    if len(best_ks) == 1:
        best_k = best_ks[0]
        return(best_k,membership)
    else:
        best_k = random.choice(best_ks)
        n_cuts, membership = partition(g,best_k)
        return(best_k,membership)

g = np.eye(5)
g[1][2] = 1
g[2][3] = 1
g[3][1] = 1


print(g)
k,memb = best_partition0(g,2,5) # -> renvoie toujours 2 groupes
print(k)

'''
nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel()
nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel()
'''
