from PIL import Image, ImageDraw
from math import *
from image_tools import *
from classes import *

import os

dir = sys.argv[1]+"/"
os.system("mkdir "+dir+"steps")


### Draw all steps


# Nettoyage des images précédentes
print("Nettoyage des images...")
os.system("rm -r "+dir+"steps/")
os.system("mkdir "+dir+"steps/")

print("Génération des images...")
for t in tqdm(range(max_time+1)):

    # Create an image
    img  = Image.new( mode = "RGB", size = (width, height), color = (38, 1, 61) )

    pickleFile = open(dir + 'time' + str(t).rjust(nbr_size,"0") + '.pkl',"rb")
    pop_lue = pickle.load(open(dir + 'time' + str(t).rjust(nbr_size,"0") + '.pkl',"rb"))

    d1 = ImageDraw.Draw(img)
    d1.text((28, 36), "Time: "+ f'{t}' + "     Clusering: " + f'{pop_lue.clustering_coef()}', fill=(255, 255, 255), font_size=35) # + "     Alpha=" + f'{alpha}' + "      Noise=" + f'{noise}'+ "     Kapa=" + f'{kapa}' + "    Proba init connexion=" + f'{proba_init_connexion}' + "    Max friends=" + f'{max_friends}' + "    Init max dist="+f'{100*max_dist/width}'+"%", fill=(255, 255, 255), font_size=35)
    #d1.text((28, 86), "Max length=" + f'{max_length}', fill=(255, 255, 255), font_size=35)

    

    # Draw affinity
    #draw_weighted_graph(pop_lue.affinity_graph,pop_lue.indiv_positions,"green",img)
    # Draw trust
    #draw_weighted_graph(pop_lue.trust_graph,pop_lue.indiv_positions,"red",img)
    # Draw utility
    #draw_weighted_graph(pop_lue.utility_graph,pop_lue.indiv_positions,"magenta",img)
    # Draw reputational flow
    #draw_weighted_graph(pop_lue.rep_flow_graph/5,pop_lue.indiv_positions,"blue",img)
    # Draw coop
    draw_coop_graph(pop_lue.coop_graph,pop_lue.indiv_positions,img)


    #show max dist:
    #draw_line(10,10,10+max_dist,10,img)

    # Save the image
    img.save(dir+"steps/step_"+str(t).rjust(nbr_size,"0")+".jpg")

### Create a GIF
"""
# Take list of paths for images
image_path_list = []

for t in tqdm(range(max_time+1)):
    image_path_list.append("steps/step_"+str(t).rjust(nbr_size,"0")+".jpg")
"""

# For a .mp4 (works maybe only on linux) :
def save_video():
    # Compilation de la vidéo
    print("Compilation de la vidéo...")
    os.system("ffmpeg -framerate 1 -pattern_type glob -i "+dir+"'steps/*.jpg' "+dir+"simulation.mp4")

save_video()

