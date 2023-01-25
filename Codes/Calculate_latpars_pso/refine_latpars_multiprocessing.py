
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import multiprocessing as mp


# In[ ]:

def pso(f='', distances = list(), lb = list(), ub = list()):
    
    pop = 500
    phi1 = 1.05
    phi2 = 1.10
    iterations=300
    verbose=False
    
    np.random.seed(10)
    
    weight = np.linspace(1,0.3, iterations)
    
    lb = np.asarray(lb) #Lower boundary
    ub = np.asarray(ub) #Upper boundary
    
    X = np.zeros((pop, lb.shape[0])) #Matrix of (particles, position_coordinates)
    V = np.zeros(X.shape) #Matrix of (particles, velocity_coordinates)
    
    px = np.zeros(X.shape)
    gx = np.zeros(X.shape)
    
    for ind in range(pop):
        xtemp = lb + (ub - lb)*np.random.uniform(size=len(ub))    
        X[ind] = xtemp        
    
    fitness = np.zeros((pop,1))
    #print(X[:10,:], distances)
    for ind in range(pop):
        fitness[ind] = globals() [f] (x = X[ind,:], d = distances)
    
    fixnan = 100*np.isnan(fitness)
    fitness = np.nan_to_num(fitness) + fixnan
    pbest = copy.deepcopy(fitness)
    px = copy.deepcopy(X)
    
    gbest = np.min(pbest)
    location = np.argmin(pbest)
    gx = X[location,:]

    minc = np.min(lb)
    maxc = np.max(ub)
    step = (maxc - minc)/50
   
    for i in range(iterations):
        
        rand = np.random.uniform(size = pop)[:,np.newaxis]

        V = weight[i]*V + phi1*np.multiply(rand,(px - X)) +             phi2*np.multiply(rand, (gx - X))
        
        X = X + V
        
        for j in range(pop):
            for k in range(lb.shape[0]):
                if X[j,k] < lb[k] or X[j,k] > ub[k]:
                    X[j,k] = lb[k] + (ub[k] - lb[k])*np.random.uniform()
        
        for ind in range(pop):
            fitness[ind] = globals() [f] (x = X[ind,:], d = distances)

            if fitness[ind] < pbest[ind]:
                px[ind] = X[ind]
                pbest[ind] = fitness[ind]
                
        if (px.std(axis=0) <= 1e-8).sum() == px.shape[-1]:
            print('converged in iteration ',i)
            print(px.mean(axis=0), pbest.mean(axis=0)[0])
            return px, pbest
                
        gbest = np.min(pbest)
        location = np.argmin(pbest)
        gx = X[location,:]
        
        if verbose:
            print(i, X[location], gbest)
    
    print('iterations ran out')
    print(px.mean(axis=0), pbest.mean(axis=0))        
    return px, pbest


# In[ ]:

def findlatpar(x = [10,10,10,1e-5,1e-5,1e-5], d = [10,10,10,10,10,10]):
    
    a, b, c = x[0], x[1], x[2]
    acos, bcos, gcos = x[3], x[4], x[5]
    
    asen, bsen, gsen = (1-acos**2), (1-bcos**2), (1-gcos**2)
   
    A = (a**2)*(b**2)*(c**2)*(1 - acos**2 - bcos**2 - gcos**2 + 2*acos*bcos*gcos)
    
    q11 = (b**2)*(c**2)*asen**2
    q22 = (a**2)*(c**2)*bsen**2
    q33 = (a**2)*(b**2)*gsen**2
    
    q12 = 2*a*b*(c**2)*(acos*bcos - gcos)
    q13 = 2*a*c*(b**2)*(acos*gcos - bcos)
    q23 = 2*b*c*(a**2)*(bcos*gcos - acos)
    
    Q11 = q11/A
    Q22 = q22/A
    Q33 = q33/A
    
    Q12 = q12/A
    Q13 = q13/A
    Q23 = q23/A    

    h, k, l = 1, 0, 0
    d100 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    h, k, l = 0, 1, 0
    d010 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    h, k, l = 0, 0, 1
    d001 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    h, k, l = 1, 1, 0
    d110 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    h, k, l = 0, 1, 1
    d011 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    h, k, l = 1, 0, 1
    d101 = (Q11*h**2 + Q22*k**2 + Q33*l**2 + Q12*h*k + Q13*h*l + Q23*k*l)**-0.5
    
    return 0.5*((d[0] - d100)**2 + (d[1] - d010)**2 +  (d[2] - d001)**2 + (d[3] - d110)**2 + (d[4] - d101)**2 + (d[5] - d011)**2)


# In[ ]:

df = pd.read_csv('assessed_dhkls.csv')


# In[ ]:

batch = df.shape[0]


# In[ ]:

reflats = np.zeros((batch, 6))
scores = list()

iterables = list()
for row in range(batch):

    print('OPTIMIZATION NUMBER', row)
    upper = list()
    lower = list()
    
    upper += [df['p0'][row]*1.20]
    upper += [df['p1'][row]*1.20]
    upper += [df['p2'][row]*1.20]
    
    lower += [df['p0'][row]*0.80]
    lower += [df['p1'][row]*0.80]
    lower += [df['p2'][row]*0.80]

    upper += [10e-1]
    upper += [10e-1]
    upper += [10e-1]
    
    lower += [-10e-1]
    lower += [-10e-1]
    lower += [-10e-1]

    d = list()

    d += [df['p0'][row]]
    d += [df['p1'][row]]
    d += [df['p2'][row]]
    d += [df['p3'][row]]
    d += [df['p4'][row]]
    d += [df['p5'][row]]
    
    iterables += [('findlatpar', d, lower, upper)]
    
if __name__ == "__main__":
        
    p = mp.Pool()
    lista = p.starmap(pso, iterables)
        
import pickle

file = open("optimized_parameters.obj","wb")
pickle.dump(lista,file)
file.close()

