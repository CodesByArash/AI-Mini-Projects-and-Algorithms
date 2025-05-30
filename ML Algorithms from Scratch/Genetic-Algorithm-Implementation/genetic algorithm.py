from copyreg import pickle
import random
import numpy as np
import matplotlib.pyplot as plt   
import math
from helperfuncs import *



def crossover(prototypes,fitness,size,threshold,n):
    nextgen        = np.empty_like(prototypes)
    nextgen[0:n]   = prototypes[0:n]


    for i in range(n,size,2):
        parent1        = prototypes[rouletteWheel(fitness, size)].copy()    
        parent2        = prototypes[rouletteWheel(fitness, size)].copy()
        length         = min(len(parent1), len(parent2))
        probability    = random.uniform(0.0, 1.0)
        if probability < threshold:
            breakpoint = random.randint(0, length - 1)
            child1     = np.concatenate([parent1[0:breakpoint],parent2[breakpoint:]])
            child2     = np.concatenate([parent2[0:breakpoint],parent1[breakpoint:]])
        else:
            child1     = parent1.copy()
            child2     = parent2.copy()

        nextgen[i]     = np.copy(child1)
        nextgen[i+1]   = np.copy(child2)
    
    return nextgen

def mutate(prototypes,size,threshold,n,lowerBound,upperBound):
    for i in range(n,size):
        probability          = random.uniform(0.0, 1.0)
        if probability       < threshold:
            prototype        = prototypes[i]
            Index            = random.randint(0, len(prototype) - 1)
            Value            = random.uniform(lowerBound[Index], upperBound[Index])
            prototype[Index] = Value

def rouletteWheel(scores,size):
    reverse        = max(scores) + min(scores)
    reversedScores = reverse - scores.copy()
    sumofscores            = sum(reversedScores)
    pick           = random.uniform(0,sumofscores)
    current        = 0
    for i in range(size):
        current   += reversedScores[i]
        if current > pick:
            return i

def fitness(prototypes,size,lowerbound,upperbound,functype):
    scores            = np.full(size , np.inf)

    for i in range(0,size):
        prototypes[i] = np.clip(prototypes[i],lowerbound,upperbound)
        scores[i]     = func(functype,prototypes[i,:])
    
    return scores



def geneticAlgorithm():
    iterations                  = 500
    dimension                   = 30
    size                        = 50
    crossoverProbability        = 1
    mutationProbability         = 0.01
    n                           = 0
    lowerbound                  = [-100]*dimension
    upperbound                  = [100]*dimension
    functype                    = "bentcigar"
    nokhbe                      = False

    best                        = np.zeros(dimension)
    scores                      = np.random.uniform(0.0,1.0,size)
    bestscore                   = float("inf")


    ga                          = np.zeros((size,dimension))
    for i in range(dimension):
        ga[:, i] = np.random.uniform(0,1,size) * (upperbound[i] - lowerbound[i]) + lowerbound[i]
    
    convergence  = np.zeros(iterations)

    for i in range(iterations):
        ga     = crossover(ga, scores, size, crossoverProbability, n)
        mutate(ga, size, mutationProbability, n, lowerbound, upperbound)

        new_ga = np.unique(ga, axis=0)
        oldLen = len(ga)
        newLen = len(new_ga)
        if newLen < oldLen:
            dupLen = oldLen - newLen
            new_ga = np.append(new_ga, np.random.uniform(0,1,(dupLen,len(ga[0]))) * (np.array(upperbound) - np.array(lowerbound)) +np.array(lowerbound), axis=0)
        
        ga     = new_ga
        
        scores = fitness(ga , size,lowerbound,upperbound,functype)

        if nokhbe == True:
            # maximum fitness score is the worst because we want to find global minimum of the func
            worst = np.where(scores == np.max(scores))
            worst = worst[0][0]
            if scores[worst]>bestscore:
                ga[worst]     = np.copy(best)
                scores[worst] = np.copy(bestscore)


        bestscore = min(scores)

        #sorting the scores and our prototypes
        sortedindex = scores.argsort()
        ga = ga[sortedindex]
        scores = scores[sortedindex]

        convergence[i] = bestscore

        if (i%1==0):
            print(['iteration '+ str(i+1)+ ' best fitness is '+ str(bestscore)])
    

    plot_result(convergence)




geneticAlgorithm()