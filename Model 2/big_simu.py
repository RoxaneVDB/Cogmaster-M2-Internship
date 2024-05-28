from tqdm import tqdm
import sys
from parameters import *
from classes import *
import os

dir = "big_simu/"+sys.argv[1]+"/"
print("Direction: ",dir)
beta = float(sys.argv[2])
noise = float(sys.argv[3])
print("beta = ", beta, " noise = ", noise)

## Main

def simulate_lots():
    os.system("rm -r "+dir)
    os.system("mkdir "+dir)
    os.system("cp parameters.py "+dir)
    for ka in kapas:
        print(">>>>> Kapa: " + f'{ka}')
        os.system("mkdir "+dir+"Kapa_" + str(ka).rjust(nbr_kapas_size,"0"))
        for k in range(nbr_simus):
            print("Simulation " + f'{k}' + '/' + f'{nbr_simus}')
            os.system("mkdir "+dir+"Kapa_" + str(ka).rjust(nbr_kapas_size,"0") + "/Simu_" + str(k).rjust(nbr_simu_size,"0") )
            pop = Population(ka,beta,noise)
            for t in range(max_time+1):
                if max_time - t < saved_steps:
                    pop.write_state(dir+"Kapa_" + str(ka).rjust(nbr_kapas_size,"0") + "/Simu_" + str(k).rjust(nbr_simu_size,"0") + "/")
                pop.actualise()

# execute from shell
if __name__=="__main__":
    simulate_lots()
    """
    answer = input('Are you sure you want to delet the previous simulations? (Enter "y" to continue)')
    if answer.lower() in ["y","yes"]:
        simulate_lots()
    else:
        print("We didn't touch the simulations :)")
    """





