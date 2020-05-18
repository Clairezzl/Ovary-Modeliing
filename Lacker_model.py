#!/usr/bin/env python
# coding: utf-8

# # Implement Laker model for follicular growth

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint


# In[54]:


#------------------------------------------------------------------------------------------------
# Generate list of names given lengh, n, starting with given label and a number from zero to n
#------------------------------------------------------------------------------------------------
def make_list_names(n):
    name_list = []
    for i in range(1, n+1):
        name_list.append(i)
    return(name_list)

#------------------------------------------------------------------------------------------------
# Generate a data frame of folicular sizes with days in rows and follicle number in columns
# Day zero is populate from given exponential distribution parameter (exp_par). 
#------------------------------------------------------------------------------------------------

def make_dataframe(nfollicles, ndays):
    # Make list of names for rows and columns
    column_names =  make_list_names(ndays) 
    row_names    =  make_list_names(nfollicles)
    # Make an empty dataframe of given size
    df = pd.DataFrame([], columns = column_names , index=row_names)
    # Generate initial set of follicles and save then as a "day0" set
    return(df)

#------------------------------------------------------------------------------------------------
# Function to update follicle size
#------------------------------------------------------------------------------------------------

M1=15.15
M2=3.85
M=10

def model(E,t):
    k=-(1-15.15/5)*(1-3.85/5)
    dEdt=E+k*E**3
    return dEdt
#------------------------------------------------------------------------------------------------
#  Simulation parameters
#------------------------------------------------------------------------------------------------
nfollicles = 10
time= 2
dt=0.1
nsteps = int(time/dt)

#------------------------------------------------------------------------------------------------
#  Initialize sizes of follicles
#------------------------------------------------------------------------------------------------  

initial=np.random.uniform(0.0,0.1,nfollicles)

#------------------------------------------------------------------------------------------------
#  Run simulation for given number of days
#------------------------------------------------------------------------------------------------  

t = np.linspace(0, time, nsteps)
Maturity=[]
for i in range(nfollicles):
    y=odeint(model,initial[i],t) 
    Maturity.append([])
    for j in range(1 ,int(nsteps*dt)+1): 
        Maturity[i].append(y[j])


# In[55]:


#Fill in Data Frame
df=make_dataframe(nfollicles, time)
for i in range(nfollicles):
    df.iloc[i]=Maturity[i]
df


# In[63]:


#Plot Maturity VS Time(Hours)
for i in range(nfollicles):
    t = np.linspace(0, time, nsteps)
    y=odeint(model,initial[i],t) 
    plt.plot(t, y, label=str(i+1))
plt.legend(loc="upper left")
plt.xlabel('time(hours)')
plt.ylabel('y(t)')
plt.show()


# In[18]:


#initial maturity, ie:Ei
np.random.uniform(0.0,0.05,5)

#maturity function 
M1=16.5
M2=6.5
def theta(Ei,E,M1,M2):
    return 1-(E-M1*Ei)*(E-M2*E2)
    
def f(Ei,E):
    return Ei*theta(Ei,E)


# In[ ]:




