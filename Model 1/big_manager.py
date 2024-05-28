import os
from parameters import *

noise_list = [0.01]
nbr_param = len(noise_list)
current_combi = 0

for noise in noise_list:
    current_combi += 1
    print("########### Paramter combination number ",current_combi,"/",nbr_param,"###############")

    dir = "noise_" + f'{noise}'

    print("Executing the simulations...")
    os.system("python big_simu.py "+dir+" "+f'{noise}')
    print("Simulations done.")

    print("Making the stats...")
    os.system("python big_stats.py "+dir)
    print("Stats gathered and ploted.")

    print("Making the follow_up...")
    os.system("python big_follow_up.py "+dir)
    print("Follow up ploted.")

    print("Making example videos...")
    for i in range(nt):
        dir_i = dir+"/"+"Threshold_"+f'{thresholds[i]}'+"/Simu_000"
        os.system("python makeVideo.py big_simu/"+dir_i)
    print("Example videos done.")

print(" #### Finished :D ####")
