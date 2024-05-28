import os
from parameters import *

beta_list = [0,0.3,0.5,0.7,1]
noise_list = [0,0.1,0.2,0.5]
nbr_param = len(beta_list) * len(noise_list)
current_combi = 0

for beta in beta_list:
    for noise in noise_list:
        current_combi += 1
        print("########### Paramter combination number ",current_combi,"/",nbr_param,"###############")

        dir = "beta_" + f'{beta}' + "_noise_" + f'{noise}'
        
        print("Executing the simulations...")
        os.system("python big_simu.py "+dir+" "+f'{beta}'+ " " +f'{noise}')
        print("Simulations done.")
        
        print("Making the stats...")
        os.system("python big_stats.py "+dir)
        print("Stats gathered and ploted.")
        
        print("Making the follow_up...")
        os.system("python big_follow_up.py "+dir)
        print("Follow up ploted.")

        print("Making example videos...")
        for i in range(nk):
            dir_i = dir+"/"+"Kapa_"+f'{kapas[i]}'+"/Simu_000"
            os.system("python makeVideo.py big_simu/"+dir_i)
        print("Example videos done.")


print(" #### Finished :D ####")
