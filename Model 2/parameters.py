import numpy as np

## Simulation parameters
max_time = 30
nbr_size = 2 # nbr of caracters needed to write the time
number_of_people = 100
proba_init_connexion = 1
alpha = 0.5 # exponnential decrement for indirect connexions
max_length = 1 # longer indirect connexions are not taken into account to calculate trust
max_friends = 5
count_loops = False
proba_busy = 0.1 # probability of trying to befriend someone with maximal nbr of friends

max_flow = 1 #10.5 # CECI N'EST PAS UN PARAMETRE
top_seniority = 10 # arbitrary :') (~ max_time maybe)

# Generalized trust is given when running the code !

# Coordinates
width = 2000
height = 1400
indiv_radius = 8
max_width = 7 # for weighted connexions
margin = 50
top_margin = 140

max_dist = width/7

## Big simu parameters & plot average

nbr_simus = 1 #30 # for each trust threshold
nbr_simu_size = 3 # nbr of caracters needed to write the number of simulations
kapas = [0.0,0.5,1.0] #np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
nk = len(kapas)
nbr_kapas_size = 3
saved_steps = max_time+1 # the n last steps are saved in big simu

## Plot a simu parameters

times = np.arange(1,max_time+1,2) #[0,1,3,5,7,10] # times where we show the trust repartition
nbr_times = len(times)
graphics_width = 10
graphics_length = 60
