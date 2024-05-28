from PIL import Image, ImageDraw
from math import *
from parameters import *
import sys



### Settings

indiv_colors = [(252, 186, 3),(0, 81, 255),(57, 163, 0),(251, 0, 255),(166, 166, 166),(255, 0, 0),(255, 255, 0),(197, 107, 227),(85, 245, 32),(5, 231, 247),(247, 254, 255),(122, 81, 5),(150, 10, 3),(250, 190, 187),(203, 242, 189),(168, 7, 101),(255, 181, 112),(186, 156, 128),(9, 145, 129),(91, 86, 184),(124, 36, 130)]


### Shapes

def draw_rect(x_left,y_top,x_right,y_bot,color,img): # y_top < y_bot   and    x_left < x_right
    for x in range (x_left,x_right+1):
        img.putpixel((x,y_top), color)
        img.putpixel((x,y_bot), color)
    for y in range (y_top,y_bot+1):
        img.putpixel((x_left,y), color)
        img.putpixel((x_right,y), color)

def fill_circle(x_center,y_center,r,color,img):
    for x in range(x_center-r,x_center+r+1):
        for y in range(y_center-r,y_center+r+1):
            if (x-x_center)**2 + (y-y_center)**2 <= r**2:
                img.putpixel((x,y), color)

def draw_line(x1,y1,x2,y2,img):
    shape = [(x1,y1),(x2,y2)]
    d = ImageDraw.Draw(img)
    d.line(shape, fill = "white", width = 2)

def draw_weighted_line(x1,y1,x2,y2,w,color,img): # w entre 0 et 1
    shape = [(x1,y1),(x2,y2)]
    mywidth = round(max_width * w)
    d = ImageDraw.Draw(img)
    d.line(shape, fill = color, width = mywidth)

### Draw the graph (???)

def draw_indiv(x,y,img):
    #fill_circle(x,y,indiv_radius,indiv_colors[i]) #version with a different color for each individual
    fill_circle(x,y,indiv_radius,(100,100,255),img)

def draw_connexion(i,j,indiv_pos,img):
    (x1,y1) = indiv_pos[i]
    (x2,y2) = indiv_pos[j]
    draw_line(x1,y1,x2,y2,img)

def draw_weighted_connexion(i,j,indiv_pos,w,color,img):
    (x1,y1) = indiv_pos[i]
    (x2,y2) = indiv_pos[j]
    draw_weighted_line(x1,y1,x2,y2,w,color,img)

def draw_coop_graph(g,indiv_pos,img):
    for i in range(number_of_people):
        for j in range(i):
            if g[i][j] == 1 :
                draw_connexion(i,j,indiv_pos,img)
    for i in range(number_of_people):
        (x,y) = indiv_pos[i]
        draw_indiv(x,y,img)
        """
        if i == 0:
            fill_circle(x,y,8,(200,0,0),img)
        if i == 1:
            fill_circle(x,y,8,(0,200,0),img)
        if i == 4:
            fill_circle(x,y,8,(200,200,0),img)
        """

def draw_weighted_graph(g_w,indiv_pos,color,img):
    for i in range(number_of_people):
        for j in range(i):
            draw_weighted_connexion(i,j,indiv_pos,g_w[i][j],color,img)
