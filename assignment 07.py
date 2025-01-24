# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:49:11 2024

@author: hwang
"""

import numpy as np
from scipy.stats import ttest_1samp

data = np.array([15.3, 14.5, 13.2, 12.8, 13.6, 13.9, 15.5, 14.8])

ttest_1samp(data, popmean=15, alternative='less')

#ttest_1samp(sample, H0 value, destination)

#2

import numpy as np
from scipy.stats import ttest_rel
import pandas as pd

textbooks = pd.read_csv('C:/datafolder/textbooks.txt', sep=" ")
textbooks

ttest_rel(textbooks['uclaNew'], textbooks['amazNew'])

#3
import numpy as np
from scipy.stats import f, ttest_ind
import pandas as pd

def var_test(sample1,sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    S1 = sum((sample1-sample1.mean())**2)/(n1-1)
    S2 = sum((sample2-sample2.mean())**2)/(n2-1)
    dfn = n1-1
    dfd = n2-1
    F = S1/S2
    pval = 2*min(f.cdf(F,dfn=dfn,dfd=dfd), 1-f.cdf(F,dfn=dfn,dfd=dfd))
    
    print("F test to compare two variances")
    print("F = %s,num df=%s,denom df=%s, p-value=%s"%(round(F,5),dfn,dfd,round(pval,5)))

run = pd.read_csv('C:/datafolder/run10samp.txt', sep=" ")
group1 = run[run['gender']=='M']
group2 = run[run['gender']=='F']

print(var_test(group1['time'], group2['time']))
#pvalue=0.1833으로 유의수준 0.05에서 등분산 가정을 기각
print(ttest_ind(group1['time'], group2['time'], alternative='two-sided', equal_var = False))





















