
# coding: utf-8

# In[59]:


import codecs
import numpy as np
from collections import defaultdict


# In[2]:


eva_file = '../data/pooling_result3.txt'


# In[42]:


arrs = []
with codecs.open(eva_file) as f:
    for line in f:
        arr = line.split()
        if len(arr)!=4:
            continue
        arrs.append(arr)


# In[43]:


class Sample:
    def __init__(self,base,maker):
        self.base = base
        self.maker = maker
        self.compares = set()
        
    def add_compare(self,compare,seq):
        self.compares.add(compare)
        if seq=='1':
            self.best=compare


# In[44]:


sample_dict = {}
for arr in arrs:
    if (arr[0],arr[2]) in sample_dict:
        sample = sample_dict[(arr[0],arr[2])]
    else:
        sample = Sample(arr[0],arr[2])
    sample.add_compare(arr[1],arr[3])
    sample_dict[(arr[0],arr[2])] = sample
        
    


# In[50]:


makers = set()
for arr in arrs:
    makers.add(arr[2])
makers = list(makers)


# In[53]:


values = list(sample_dict.values())


# In[56]:


def check(v1,v2):
    if v1.base != v2.base:
        return -1
    if v1.maker == v2.maker:
        return -1
    if len(v1.compares)!=3 or len(v2.compares)!=3:
        return -1
    for k in v1.compares:
        if k not in v2.compares:
            return -1
    if v1.best != v2.best:
        return 0
    else:
        return 1
    


# In[66]:


corrects = defaultdict(int)
sums = defaultdict(int)
del_maker = ['王爽','李志豪']
for i in range(len(values)):
    for j in range(i,len(values)):
        
        v1,v2 = values[i],values[j]
        if v1.maker in del_maker or v2.maker in del_maker:
            continue
        if check(v1,v2)==-1:
            continue
        if check(v1,v2)==0:
            sums[v1.maker] +=1
            sums[v2.maker] +=1
            continue
        if check(v1,v2)==1:
            corrects[v1.maker] +=1
            corrects[v2.maker] +=1
            sums[v1.maker] +=1
            sums[v2.maker] +=1


# In[67]:


for k in sums:
    print(k,corrects[k]*1.0/sums[k])

