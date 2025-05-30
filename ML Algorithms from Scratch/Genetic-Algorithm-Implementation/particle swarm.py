from turtle import position
from numpy import nper
import random
import numpy as np 
import math 
from helperfuncs import *

functype    = "sphere"
lowerbound  = -100   
upperbound  = 100    
dimension   = 30     
size        = 50
iterations  = 500   
Velocitymax = 6     
wmax        = 0.9   
wmin        = 0.2   
c1          = 1
c2          = 2

lowerbound  = [lowerbound]*dimension
upperbound  = [upperbound]*dimension

velocity    = np.zeros((size,dimension))
bestscorep  = np.zeros(size)
#best score is the lowest score so we initialize scores with +inf and try to make it as low as we can
bestscorep.fill(float("inf"))
bestp       = np.zeros((size,dimension))
bestg       = np.zeros(dimension)
bestscoreg  = float("inf")


pos = np.zeros((size, dimension))

for i in range(dimension):
    pos[:,i] = np.random.uniform(0,1,size) * (upperbound[i]-lowerbound[i])+lowerbound[i]

convergence  = np.zeros(iterations)

for i in range(0,iterations):
    for j in range(0,size):
        for k in range(dimension):
            pos[j,k] = np.clip(pos[j,k], lowerbound[k],upperbound[k])
        fitness=func(functype,pos[j,:])
        if(bestscorep[j]>fitness):
            bestscorep[j] = fitness
            bestp[j,:]    = pos[j,:].copy()
        
        if(bestscoreg>fitness):
            bestscoreg    = fitness
            bestg         = pos[j,:].copy()
    
    w = wmax - i*((wmax-wmin)/iterations)

    for j in range(0,size):
        for k in range(0,dimension):
            r1            = random.random()
            r2            = random.random()
            
            velocity[j,k] = w*velocity[j,k]+c1*r1*(bestp[j,k]-pos[j,k])+c2*r2*(bestg[k]-pos[j,k])
            if(velocity[j,k] > Velocitymax):
                velocity[j,k] = Velocitymax
            if(velocity[j,k] < -Velocitymax):
                velocity[j,k] = -Velocitymax
            
            pos[j,k]=pos[j,k]+velocity[j,k]
    
    convergence[i]=bestscoreg

    if (i%1==0):
        print(['At iteration '+ str(i+1)+ ' the best fitness is '+ str(bestscoreg)])

plot_result(convergence)