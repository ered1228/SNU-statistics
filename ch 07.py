# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:02:41 2024

@author: hwang
"""

#t distribution

import numpy as np
from scipy.stats import ttest_1samp

score = np.array([22,25,34,35,41,41,46,
46,46,47,49,54,54,59,60])

ttest_1samp(score, popmean=40, alternative='greater')
#ttest_1samp(sample, H0 value, destination)

can = np.array([408,405,397,405,395,415,389,403,397,390])
ttest_1samp(can, popmean=400, alternative='two-sided')
#귀무가설 나올 확률이 88퍼니까 기각불가

# paired comparison
#cuz we compare D(=(E(X)-E(Y))), it is same as ttest_1samp

from scipy.stats import ttest_rel
import pandas as pd

paired = pd.read_csv('C:/datafolder/paired.txt')

ttest_rel(paired['purple'], paired['green'])


before = np.array([18, 21, 16, 22, 19, 24, 17, 21, 23, 18, 14, 16, 16, 19, 18, 20, 12, 22, 15, 17])
after = np.array([22, 25, 17, 24, 16, 29, 20, 23, 19, 20, 15, 15, 18, 26, 18, 24, 18, 25, 19, 16])

print(ttest_rel(after, before))
print(before.mean())
print(after.mean())

#var comparison test (F)

import pandas as pd
from scipy.stats import f

def var_test(sample1,sample2): #function that judges the equal_val hypothesis
    n1=len(sample1)
    n2=len(sample2)
    S1=sum((sample1-sample1.mean())**2)/(n1-1)
    S2=sum((sample2-sample2.mean())**2)/(n2-1)
    dfn = n1-1
    dfd = n2-1
    F = S1/S2
    pval = 2*min(f.cdf(F,dfn=dfn,dfd=dfd), 1-f.cdf(F,dfn=dfn,dfd=dfd))
    return F, dfn, dfd, pval

paint = pd.read_table("C:/datafolder/paint.txt", sep=" ")
group1 = paint[paint['group']==1]
group2 = paint[paint['group']==2]

F, dfn, dfd, pval = var_test(group1.time, group2.time)

print("F test to compare two variances")
print("F = %s,num df=%s,denom df=%s, p-value=%s"%(round(F,5),dfn,dfd,round(pval,5)))

#기각 불가 -> 독립이표본(ttest_ind) 등분산 가정
#note that equal_var = True ? False

from scipy.stats import ttest_ind

ttest_ind(group1.time, group2.time, alternative='greater', equal_var = True)

# ex
group1 = np.array([22, 23, 25, 26, 27, 19, 22, 28, 33, 24])
group2 = np.array([21, 25, 36, 24, 33, 28, 29, 31, 30, 32, 33, 35])

#집단별로 이 값이 차이가 있는지 확인하기 위해서 두 모평균 차이에 대한 추론을 해야 하는데,
#모분산을 모르고 n이 작아서 정규근사도 못한다. 그럼 등분산 가정이 성립하는지 확인

def var_test(sample1,sample2):
    n1=len(sample1)
    n2=len(sample2)
    S1=sum((sample1-sample1.mean())**2)/(n1-1)
    S2=sum((sample2-sample2.mean())**2)/(n2-1)
    dfn = n1-1
    dfd = n2-1
    F = S1/S2
    pval = 2*min(f.cdf(F,dfn=dfn,dfd=dfd), 1-f.cdf(F,dfn=dfn,dfd=dfd))
    
    print("F test to compare two variances")
    print("F = %s,num df=%s,denom df=%s, p-value=%s"%(round(F,5),dfn,dfd,round(pval,5)))

var_test(group1, group2)
#pvalue=0.64207로 유의수준 0.05에서 기각 불가 -> 등분산 가정
from scipy.stats import ttest_ind

ttest_ind(group1, group2, alternative='two-sided', equal_var = True)
#pval = 0.015578로 유의수준 0.05에서 null hypothesis 기각. -> 집단별로 값의 차이가 있디.
































