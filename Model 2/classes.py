import numpy as np
import random
import pickle
# for plots:
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import sys
from parameters import *
from math import *

## Comments

# Graphs must all be symetric (not oriented)
# Diagonal must be empty (no i-i connexion)
# np.copy(g) gives an independant copy :)

## Initialisation

def positions():
    indiv_positions = []
    for i in range(number_of_people):
        indiv_positions.append( (random.randint(margin, width-margin),random.randint(top_margin,height-margin)) )
    return(indiv_positions)

def local_rand_graph(p,n,indiv_positions): # initial connexions can exist only if the distance between and j is smaller
    g = np.zeros((n,n))
    individuals = [i for i in range(number_of_people)]
    for i in range(n):
        added = 0
        random.shuffle(individuals)
        pos = 0
        while pos < n and added < max_friends//2: # not i+1 because we don't want a connexion i-i
            j = individuals[pos]
            if j != i:
                (x_i,y_i) = indiv_positions[i]
                (x_j,y_j) = indiv_positions[j]
                if (x_i-x_j)**2 + (y_i-y_j)**2 <= max_dist**2:
                    if random.random() <= p:
                        g[i][j] = 1
                        g[j][i] = 1
                        added += 1
            pos += 1
    return(g)

def rand_affinity_graph(n): # Returns a graph which edges are weighted with a random float between 0 and 1, it is symmetric
    g = np.zeros((n,n))
    for i in range(n):
        for j in range(i): # not i+1 because we don't want a connexion i-i
            r = random.random()
            g[i][j] = r
            g[j][i] = r
    return(g)

## Some functions

def symmetrize(g):
    (n,m) = np.shape(g)
    for i in range(n):
        for j in range(i):
            g[j][i] = g[i][j]

def diagonal(g):
    (n,m) = np.shape(g)
    if n != m:
        raise "Diagonal of a non square matrix."
    return([ g[i][i] for i in range(n)])

def rd_noise(max_noise):
    r = random.random() * max_noise
    i = random.randint(0,1)
    if i == 1:
        r = -r
    return(r)

def noise_graph(max_noise):
    res = np.zeros((number_of_people,number_of_people))
    for i in range(number_of_people):
        for j in range(i):
            r = rd_noise(max_noise)
            res[i][j] = r
            res[j][i] = r
    return(res)

def normalize_seniority(x): # x is a number but this will be applied to a matrix
    k_sat = top_seniority/2
    return(x/(k_sat + x))

def normalize_rep_flow(x): # x is a number but this will be applied to a matrix
    k_sat = max_flow / 2
    return(x/(k_sat + x))


### Number of paths per length when loops are counter

def nbr_of_path_per_length0(g,k_max): # with the paths with loops
    if not count_loops:
        raise "Wrong call of nbr_of_path_per_length0."
    m = np.copy(g)
    (n,n_) = np.shape(g)
    res=[m]
    for k in range(2,k_max+1):
        m2 = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                for h in range(n):
                    m2[i][j] += m[i][h] * g[h][j]
        copy_into(m2,m)
        res.append(np.copy(m))
    return(res)

### Number of paths per length without loops for max_length = 1


def nbr_of_path_per_length_k1(g,k_max): # only for k = 1, without loop
    if k_max != 1:
        raise "Wrong call of nbr_of_path_per_length_k1."
    (n,n_) = np.shape(g)
    res=np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            if i != j: # avoid easy loops
                for h in range(n):
                    if h != j and h != i:
                        if g[i][h] * g[h][j] == 1:
                            res[i][j] += 1
                            res[j][i] += 1
    return([g,res])

### Number of paths per length without loops (general case)

"""
# Failed try to delet loops
def nbr_of_path_per_length_k1(g,k_max): # only paths without loop
    m = np.copy(g)
    (n,n_) = np.shape(g)
    res=[g]
    cycles=[diagonal(g)]
    for k in range(1,k_max+1):
        m2 = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                for h in range(n):
                    #if h != j: (inutile)
                    m2[i][j] += m[i][h] * g[h][j]
                for t in range(k):
                    m2[i][j] -= res[t][i][j] * cycles[k-t-1][j]
        cycles.append(diagonal(m2))
        for i in range(n):
            m2[i][i] = 0 # delete loops
        m = np.copy(m2)
        res.append(np.copy(m))
    print("res:")
    print(res)
    print("cycles:")
    print(cycles)
    return(res)
"""

def first_subset(n,i,j,k):
    res = np.zeros(n) # first cell is useless
    if (k >= n) or (i!=j and k >= n-1):
        raise "Not enough vertices to build a path of length k"
    else:
        res[i] = -1
        res[j] = -1
        k_act = 1
        pos = 0
        while pos < n and k_act <= k:
            if pos == i or pos == j:
                pos += 1
            else:
                res[pos] = k_act
                k_act += 1
                pos += 1
        return(res)

def is_eq(t1,t2): # Compare two arrays of dimension 1
    n = len(t1)
    m = len(t2)
    if n != m:
        return(False)
    for i in range(n):
        if t1[i] != t2[i]:
            return(False)
    return(True)

def find_pos(l,x):
    n = len(l)
    pos = 0
    found = False
    while not found and pos < n:
        if l[pos] == x:
            found = True
        else:
            pos += 1
    if not found:
        raise "x is not in l"
    return(pos)

def extract_subs(subs_index,i,j,k,n):
    is_empty = is_eq(subs_index,np.zeros(n))
    res = [i]
    n = len(subs_index)
    for k_ in range(1,k+1):
        pos = find_pos(subs_index,k_)
        res.append(pos)
    res.append(j)
    return(is_empty,res)

def last_free_pos(x,l): # tells if x is in the last free position of l or not
    if x == 0:
        raise "x must be different from 0"
    pos = find_pos(l,x)
    is_last = True
    n = len(l)
    for t in range(pos+1,n):
        if l[t] == 0:
            is_last = False
    return(is_last)

def next_free_pos(pos,l):
    if pos < 0:
        raise("Negative index")
    res = pos + 1
    n = len(l)
    if res > n:
        raise "No next free position"
    found = False
    while res < n and not found:
        if l[res] == 0:
            found = True
        else:
            res += 1
    if not found:
        raise "No next free position"
    return(res)

"""
# Plan de reset index
si last_k = 1: return(finished=true,_)
k_act = last_k
res = copy(subs_index)
mettre tous les k > last_k (jusqu'à k_max) à 0
si c'est possible de décaler last_k de 1 : le faire
sinon :
    k_act -= 1
    finished, res = reset_index(res,k_act,k_max)
replacer tous les k > k_act à partir du départ dans res
return(finished,res)
"""
"""
def test_2_2(l):
    l2 = l.tolist()
    if 2 in l2:
        pos = find_pos(l2,2)
        l2.pop(pos)
    if 2 in l2:
        pos = find_pos(l2,2)
        raise "two two !!!"
"""

# last_k is the one which is incremented, all k > last_k must be put to 0 (as close as possible) and all k < last_k must stay still
def reset_index(subs_index,last_k,k_max):
    res = np.copy(subs_index)
    n = len(res)
    k_act = last_k
    if k_act <= 0:
        return(True,np.zeros(n)) # declare finished
    else:
        # Remove all k that cannot be incremented (in decreasing order)
        for t in range(n):
            if res[t] > k_act:
                res[t] = 0
        # in case it is not sufficient:
        while last_free_pos(k_act,res):
            k_act -= 1
            if k_act <= 0:
                return(True,np.zeros(n)) # declare finished
            # Remove all k > k_act :
            for t in range(n):
                if res[t] > k_act:
                    res[t] = 0
        for t in range(n):
            if res[t] > k_act:
                res[t] = 0

        # Increment k_act:
        p = find_pos(res,k_act)
        next_p = next_free_pos(p,res)
        res[p] = 0
        res[next_p] = k_act

        # Reinit all k > last_k :
        pos = 0
        if res[pos] != 0:
            pos = next_free_pos(pos,res)
        for k_ in range (k_act+1, k_max+1):
            res[pos] = k_
            if k_ < k_max:
                pos = next_free_pos(pos,res)
        return(False,res)

def next_subs(subs_index,i,j,k): # k is the number that must be incremented in the next cell
    n = len(subs_index) - 1
    res = np.copy(subs_index)
    finished = False
    #while not incremented and k_act > 0:
    if last_free_pos(k,res):
        finished, res = reset_index(res,k-1,k)
    else:
        pos = find_pos(res,k)
        next_pos = next_free_pos(pos,res)
        res[pos] = 0
        res[next_pos] = k
    return(finished,res)


def nbr_of_path_per_length_general(g,k_max): # only paths without loop
    talk = False # print or not
    (n,m) = np.shape(g)
    res = [g]
    for k in range(1,k_max+1):
        if talk:
            print("######################### K = ",k,"#######################")
        res_k = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if talk:
                    print(">>>>> i = ",i," j = ",j)
                finished = False
                #if (i == j and k < n) or k < n-1: # check that there are at leat k other vertices in the graph
                if i != j and k < n-1: # check that there are at leat k other vertices in the graph and do not calculate the diagonal (useless)
                    subs_index = first_subset(n,i,j,k)
                    while not finished:
                        is_empty, sequence = extract_subs(subs_index,i,j,k,n)
                        if talk:
                            print("subs_index:",subs_index,"sequence",sequence)
                        if is_empty:
                            raise "empty sequence"
                        else:
                            pbm = False
                            for t in range(k+1):
                                if not g[sequence[t]][sequence[t+1]]:
                                    pbm = True
                            if not pbm:
                                res_k[i][j]+=1
                            finished, subs_index = next_subs(subs_index,i,j,k)
        res.append(res_k)
    if talk:
        print("Result:")
        print(res)
    return(res)


### Attribution of nbr_of_path_per_length depending on the case

def nbr_of_path_per_length(g,k_max):
    if count_loops:
        return (nbr_of_path_per_length0(g,k_max))
    if max_length < 1:
        raise "The maximal length of paths is too small."
    if max_length == 1:
        return (nbr_of_path_per_length_k1(g,k_max))
    if max_length > 1:
        return(nbr_of_path_per_length_general(g,k_max))
    raise "How did I end up here?"

## Class

class Population:
    def __init__(self,kapa,beta,noise): # Random initialisation
        self.time = 0
        self.indiv_positions = positions()
        self.coop_graph = local_rand_graph(proba_init_connexion,number_of_people,self.indiv_positions)
        self.trust_graph = np.zeros((number_of_people,number_of_people))
        self.rep_flow_graph = np.zeros((number_of_people,number_of_people))
        self.affinity_graph = rand_affinity_graph(number_of_people)
        self.utility_graph = np.zeros((number_of_people,number_of_people))
        self.seniority_graph = np.zeros((number_of_people,number_of_people))
        self.kapa = kapa
        self.beta = beta
        self.noise = noise
    def actualise_seniority(self):
        self.seniority_graph *= self.coop_graph # if the cooperation no longer exists : senority <- 0
        self.seniority_graph += self.coop_graph # the other relation gain +1 of senioriy
        #print("time="+f'{self.time}')
        #print(self.seniority_graph)
    def actualise_rep_flow(self):
        l = nbr_of_path_per_length(self.coop_graph,max_length)
        for i in range(number_of_people):
            for j in range(i):
                self.rep_flow_graph[i][j] = 0
                self.rep_flow_graph[j][i] = 0
                for k in range(1,max_length+1): # doesn't count 1 for i-j
                    to_add = l[k][i][j] * alpha ** (k+1)
                    self.rep_flow_graph[i][j] += to_add
                    self.rep_flow_graph[j][i] += to_add
    def actualise_trust(self):
        self.actualise_rep_flow()
        # Do not actualise seniority !
        self.trust_graph = (1-self.beta) * normalize_rep_flow(self.rep_flow_graph) + self.beta * normalize_seniority(self.seniority_graph)
    def actualise_utility(self):
        self.actualise_trust()
        utility_noise = noise_graph(self.noise)
        self.utility_graph = self.kapa * self.trust_graph + (1-self.kapa) * self.affinity_graph + utility_noise

    def low_utility_friend(self,i): # returns the id of one of the friends with the lowest utility (randomly between the worst)
            found_a_friend = False
            k=-1
            (n,n_) = np.shape(self.coop_graph)
            while (not found_a_friend) and k < n:
                k += 1
                if self.coop_graph[i][k]:
                    found_a_friend = True
            if (not found_a_friend):
                raise "You asked for a low trusted friend of an individual with no friend at all."
            else:
                min_trust = self.utility_graph[i][k]
                l = [k] # list of equaly less "usefull" friends
                for j in range (n):
                    if self.coop_graph[i][j]:
                        if self.utility_graph[i][j] == min_trust:
                            l.append(j)
                        if self.utility_graph[i][j] < min_trust:
                            l = [j]
                            min_trust = self.utility_graph[i][j]
            return(random.choice(l))

    def friends_list(self,i):
        friends = []
        for j in range(number_of_people):
            if i != j:
                if self.coop_graph[i][j] == 1:
                    friends.append(j)
        return(friends)

    def nbr_friends(self,i):
        friends = self.friends_list(i)
        return(len(friends))

    def potential_friends(self,i):
        res = []
        for j in range(number_of_people):
            if i!=j and self.coop_graph[i][j] == 0: # not friends yet
                if self.nbr_friends(j) <= max_friends: # and j has some free time (needed ??)
                    res.append(j)
                elif random.random() <= proba_busy: # Try to befriend a busy guy
                    res.append(j)
        return(res)

    def best_potential_friend(self,i):
        potentials = self.potential_friends(i)
        if potentials == []:
            return(-1)
        if potentials == []: # prevent bug
            raise "I didn't stop after return :D"
        max_u = -1
        l = [] # list of friends with equaly best utility
        for j in potentials:
            if self.utility_graph[i][j] == max_u:
                l.append(j)
            if self.utility_graph[i][j] > max_u:
                l = [j]
                max_u = self.utility_graph[i][j]
        return(random.choice(l))

    def actualise_coop(self):
        for i in range(number_of_people):
            best_pot = self.best_potential_friend(i)
            if best_pot != -1 : # ie. There exists one
                if self.nbr_friends(i) < max_friends:
                    self.coop_graph[i][best_pot] = 1
                    self.coop_graph[best_pot][i] = 1
                else:
                    poor_friend = self.low_utility_friend(i)
                    poor_u = self.utility_graph[i][poor_friend]
                    pot_u = self.utility_graph[i][best_pot]
                    if pot_u > poor_u:
                        self.coop_graph[i][best_pot] = 1
                        self.coop_graph[best_pot][i] = 1
                        self.coop_graph[i][poor_friend] = 0
                        self.coop_graph[poor_friend][i] = 0

    def limit_friends(self):
    #suppress connexions if someone has more friends than the max authorised by deleting the ones with the lowest utility
        my_pop = [i for i in range(number_of_people)]
        random.shuffle(my_pop)
        for i in my_pop:
            friends = self.friends_list(i)
            l = len(friends)
            if l > max_friends:
                for k in range(l-max_friends):
                    #self.actualise_utility() # XXXXXXXXXXXX ça y est pas dans la V1 >< -> TROP COUTEUX !!!
                    supressed_friend = self.low_utility_friend(i)
                    self.coop_graph[i][supressed_friend] = 0
                    self.coop_graph[supressed_friend][i] = 0

    def actualise(self):
        self.time += 1
        # Prepare info
        self.actualise_seniority()
        self.actualise_utility() # activates self.actualise_rep_flow() and self.actualise_trust()
        # Add new friends (too many)
        self.actualise_coop()
        # Limit the number of friends
        self.limit_friends()

    def write_state(self,dir):
        file_name = dir + 'time' + str(self.time).rjust(nbr_size,"0") + '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def clustering_coef(self): # Bad complexity :/
        g = self.coop_graph
        (n,n_) = np.shape(g)
        nbr_of_triangles = 0
        nbr_of_neighboors_pairs = 0
        for i in range(n):
            for j in range(n):
                if (g[i][j] == 1) and (i != j):
                    for h in range(n):
                        if (g[j][h] == 1) and (h != j) and (h != i):
                            nbr_of_neighboors_pairs += 1
                            if g[h][i] == 1:
                                nbr_of_triangles += 1
        # nbr_of_triangles and nbr_of_neighboors_pairs should both be divided by 2 but no need thanks to the fraction.
        # triangles are already counted 3 times each.
        if nbr_of_triangles == 0:
            return(0)
        if nbr_of_neighboors_pairs == 0:
            raise "clustering coef is broken." # ?
        return(nbr_of_triangles / nbr_of_neighboors_pairs)

    def nbr_of_pairs(self): # Copy of clust coef...
        g = self.coop_graph
        (n,n_) = np.shape(g)
        nbr_of_neighboors_pairs = 0
        for i in range(n):
            for j in range(n):
                if (g[i][j] == 1) and (i != j):
                    for h in range(n):
                        if (g[j][h] == 1) and (h != j) and (h != i):
                            nbr_of_neighboors_pairs += 1
        return(nbr_of_neighboors_pairs)

    def nbr_connexions(self):
        g = self.coop_graph
        (n,n_) = np.shape(g)
        res = 0
        for i in range(n):
            for j in range(i):
                res += g[i][j]
        return(res)
