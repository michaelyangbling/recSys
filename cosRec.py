
#%%
import pandas as pd
import numpy as np
from numpy import vectorize
import math
vfunc = vectorize( lambda x: 0 if x==0 else 1/( math.sqrt(x) ) )
R = pd.read_csv("user-shows.txt", sep = " ", header = None).values # userItem matrix R

P = np.diag( np.sum(R, axis=1) )   #row sum as diag

P2 = vfunc(P)

Rt = R.transpose()
res = P2 @ R @ Rt @ P2 @ R


#%%
import os
os.getcwd()


#%%
alexRate=list(res[499, :])[0:100] #predicted rating of alex, first 100 columns(items

i=0
while i<len(alexRate):
    alexRate[i] = [ alexRate[i], i ] #( rate, index ) tuple
    i+=1

alexRate.sort(key = lambda lst: (-lst[0], lst[1] ) )


#%%
with open('shows.txt', 'r') as myfile:
  data = myfile.read()


#%%
titleList = data.split('\n')[0:-1]

#%% [markdown]
# ⋆ SOLUTION: user user shows:
# • FOX 28 News at 10pm
# • Family Guy
# • 2009 NCAA Basketball Tournament • 
# NBC 4 at Eleven
# • Two and a Half Men

#%%
for tup in alexRate:
    tup.append( titleList [ tup[1] ] )
print(alexRate)


#%%

Q = np.diag( np.sum(R, axis=0) )   #column sum as diag

Q2 = vfunc(Q)

res2 = R @ Q2 @ Rt @ R @ Q2

#%% [markdown]
# item item shows:
# • FOX 28 News at 10pm
# • Family Guy
# • NBC 4 at Eleven
# • 2009 NCAA Basketball Tournament • 
# Access Hollywood

#%%
alexRate2=list(res2[499, :])[0:100] #predicted rating of alex, first 100 columns(items

i=0
while i<len(alexRate2):
    alexRate2[i] = [ alexRate2[i], i ] #( rate, index ) tuple
    i+=1

alexRate2.sort(key = lambda lst: (-lst[0], lst[1] ) )

for tup in alexRate2:
    tup.append( titleList [ tup[1] ] )
print(alexRate2)


#%%
u, s, vh = np.linalg.svd(R, full_matrices = False)
res3 =  u[:, range(0,320)] @ np.diag( s[ range(0,320) ] ) @ vh[range(0,320), :]
alexRate3=list(res3[499, :])[0:100] #predicted rating of alex, first 100 columns(items

i=0
while i<len(alexRate3):
    alexRate3[i] = [ alexRate3[i], i ] #( rate, index ) tuple
    i+=1

alexRate3.sort(key = lambda lst: (-lst[0], lst[1] ) )

for tup in alexRate3:
    tup.append( titleList [ tup[1] ] )
print(alexRate3)


#%%
isum = 0
for i in s[ range(0,320) ]:
    isum+= i**2
isum2=0
for i in s:
    isum2+= i**2
print(isum/isum2)


