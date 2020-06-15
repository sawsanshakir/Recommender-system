#!/usr/bin/env python
# coding: utf-8

# In[2]:


### 
### Reference: https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb

### ## compare the mean avarage precision at K=3,5,10,15,20,25 for the four algorithms


UBCF_mapk= [0.8686710434696111, 0.8437588282104844, 0.8023160917770736, 0.7954577496950977, 0.7918253350200463, 0.7912932375098817]
EUBCF_mapk= [0.8685218342783249, 0.8424199244006763, 0.8000742921924232, 0.7930240201167648, 0.7893333280082572, 0.7886659520365779]
IBCF_mapk = [0.5968616333432806, 0.46396896448821245, 0.3237631742242306, 0.28959979287834514, 0.27721383324703575, 0.2726211595887138]
SVD_mapk = [0.8507410723167215, 0.8451474684173877, 0.8772055488281721, 0.8922412916327676, 0.8968680227960569, 0.8994527818522856]
 


import matplotlib.pyplot as plt
import recmetrics


mapk_scores = [UBCF_mapk, EUBCF_mapk, IBCF_mapk ,SVD_mapk]
index = [3, 5,10, 15, 20, 25]
names = ['UBCF_mapk ', 'EUBCF_mapk', 'IBCF_mapk', 'SVD_mapk']

fig = plt.figure(figsize=(15, 7))
recmetrics.mapk_plot(mapk_scores, model_names=names, k_range=index)


# In[5]:


## ### Reference: https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
## compare the mean avarage recall at K=3,5,10,15,20,25 for the four algorithms


UBCF_mark= [0.29724866121573496, 0.4688902752071831, 0.6772801397333045, 0.7397214860889163, 0.7632806672681606, 0.773180385397193]
EUBCF_mark= [0.29700138412199406, 0.4682005170890454, 0.6751408672925868, 0.7372976969041258, 0.7608149396383603, 0.7705784456632339]
IBCF_mark= [0.18823187134776825, 0.23233473978506464, 0.255230224738292, 0.25996311016465007, 0.26135484710739587, 0.2619007616772527]
SVD_mark= [0.2891515187174481, 0.47363544061280966, 0.7531206222703732, 0.8369511131416096, 0.868625481761649, 0.8815853605313716]

import matplotlib.pyplot as plt
import recmetrics

mark_scores = [UBCF_mark, EUBCF_mark, IBCF_mark, SVD_mark]
index = [3, 5, 10, 15, 20, 25]
names = ['UBCF_mark ', 'EUBCF_mark', 'IBCF_mark','SVD_mark']

fig = plt.figure(figsize=(15, 7))
recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)


# In[9]:



### compare the precision-recall curves for the four algorithms

UBCF_pk= [0.9084380610412927,0.8847396768402154,0.7017953321364452,0.548294434470377,0.44106822262118495,0.3657091561938959]
EUBCF_pk= [0.9084380610412927,0.8833034111310593,0.7000897666068223,0.5469778575703174,0.4396768402154398,0.36466786355475767]
IBCF_pk = [0.737835875090777,0.5703703703703704,0.3354030501089325,0.23413217138707335,0.17859477124183004,0.14405228758169936]
SVD_pk = [0.908868501529052,0.9073394495412844,0.7801834862385322,0.6218960244648319,0.505091743119266,0.4201100917431193]

UBCF_rk= [0.31656734650469803, 0.5055415267828681, 0.7276046806832099, 0.7950746770425247,0.8196836156309496, 0.8301244987706939]
EUBCF_rk = [0.3171274058896105,0.5042853072548215, 0.7246846442911873,0.7919864734002526,0.8163976606009673,0.8268892761538986 ]
IBCF_rk= [0.1948414076785726, 0.24033651612861853, 0.2637691272846124, 0.26867982298740206, 0.27009149820848694,0.2706327814324085]
SVD_rk= [0.30834504426398784,0.5111749712755196,0.8184610230877879,0.9111029408279819,0.9456746119021202,0.9595861606783815]



import matplotlib.pyplot as plt
import numpy as np
 
x=np.arange(50)
 
fig=plt.figure()
ax=fig.add_subplot(111)

top_n = [3, 5,10,15,20, 25]

    
ax.plot(UBCF_rk, UBCF_pk, c='k',marker="o",ls='--', label='UBCF', fillstyle='none')
ax.plot(EUBCF_rk, EUBCF_pk, c='r', ls='-', label='EUBCF')
ax.plot(IBCF_rk, IBCF_pk, c='b', marker="s", ls='-', label='IBCF')
ax.plot(SVD_rk,SVD_pk, c='g', marker="v", ls='-', label='SVD')



plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc=4)

for n,x,y in zip(top_n,SVD_rk,SVD_pk):
    
    plt.annotate(n, xy=(x,y), textcoords="offset points", xytext=(0,10), ha='center')



plt.show()


# In[8]:


### compare the precision-recall curves for the four algorithms

import matplotlib.pyplot as plt
import numpy as np

 

UBCF_fpr = [0.2045550998049217, 0.35951058885601844, 0.6060395511963478, 0.683653683228117, 0.7141316208940747, 0.7289996979085744]
UBCF_tpr = [ 0.31656734650469803,0.5055415267828681, 0.7276046806832099,0.7950746770425247, 0.8196836156309496, 0.8301244987706939]


EUBCF_fpr =[0.20492947584485513,0.3560621023008161,0.6019547778983695,0.6737167107335137,0.7152002527495016,0.724000664761228]
EUBCF_tpr = [0.3171274058896105,0.5042853072548215,0.7246846442911873,0.7919864734002526,0.8163976606009673,0.8268892761538986]

IBCF_fpr = [0.11418693711538969,0.1502379631830547,0.16366796080606397,0.16681747352296436,0.16681747352296436,0.167649420278372]
IBCF_tpr = [0.1948414076785726, 0.24033651612861853, 0.2637691272846124, 0.26867982298740206, 0.27009149820848694, 0.2706327814324085]

SVD_fpr = [0.248416222343011,0.4283959763660263,0.7598191898691067,0.8608032775586852,0.8971667341218089,0.9202914204910876]
SVD_tpr = [0.30834504426398784,0.5111749712755196,0.8184610230877879,0.9111029408279819,0.9456746119021202,0.9595861606783815]
 
 
    

top_n = [3, 5,10,15,20, 25]


#x=np.arange(6)
 
fig=plt.figure()
ax=fig.add_subplot(111)


ax.plot( UBCF_fpr, UBCF_tpr, c='k',marker="o",ls='-', label='UBCF', fillstyle='none')
ax.plot(EUBCF_fpr , EUBCF_tpr , c='r',  ls='-', label='EUBCF')
ax.plot(IBCF_fpr, IBCF_tpr, c='b', marker="s", ls='-', label='IBCF')
ax.plot(SVD_fpr,SVD_tpr, c='g', marker="v", ls='--', label='SVD')
plt.xlim(0.1, 1)
plt.ylim(0.1, 1)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend(loc=4)

for n,x,y in zip(top_n,UBCF_fpr, UBCF_tpr):
    
    plt.annotate(n, xy=(x,y), textcoords="offset points", xytext=(0,10), ha='center')


plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np


IBCF_fpr = [0.11418693711538969,0.1502379631830547,0.16366796080606397,0.16681747352296436,0.16681747352296436,0.167649420278372]
IBCF_tpr = [0.1948414076785726, 0.24033651612861853, 0.2637691272846124, 0.26867982298740206, 0.27009149820848694, 0.2706327814324085]

top_n = [3, 5,10,15,20, 25]


#x=np.arange(6)
 
fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(IBCF_fpr, IBCF_tpr, c='b', marker="s", ls='-', label='IBCF')

plt.xlim(0.1, 0.2)
plt.ylim(0.1, 0.3)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend(loc=4)

for n,x,y in zip(top_n,IBCF_fpr,IBCF_tpr):
    
    plt.annotate(n, xy=(x,y), textcoords="offset points", xytext=(0,10), ha='center')


plt.show()


# In[ ]:




