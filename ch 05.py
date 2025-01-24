# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:01:57 2024

@author: hwang
"""


#정규분포의 확률밀도함수(pdf)
from scipy.stats import norm
norm.pdf(x=40, loc=42, scale=5) #n, mean, std

#표준정규분포 그리기
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100, endpoint=True)
fx = norm.pdf(x, loc=0, scale=1)
plt.plot(x, fx)
plt.show()

#정규분포의 누적확률(cdf)
#X~N(2000, 200**2)
from scipy.stats import norm
#2500 이하일 확률
norm.cdf(x=2500, loc=2000, scale=200)
#1800 이상일 확률
1-norm.cdf(x=1800, loc=2000, scale=200)

#정규분위수 계산(ppf)
#q(분위수)가 주어져 있을 때 x 값
norm.ppf(q=0.98, loc=100, scale=15)
norm.ppf(q=0.9937903346742238, loc=2000, scale=200)

#이항분포의 확률질량함수(pmf)
from scipy.stats import binom
binom.pmf(k=3, n=10, p=0.5)

import math
a = math.comb(10, 3) * (0.5)**10
a

binom.pmf([1, 2, 3], n=10, p=0.5)

#이항분포의 확률질량함수 그리기
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 7)
px = binom.pmf(x, n=6, p=0.5)
plt.plot(x, px, 'C0*')
plt.vlines(x, 0, px, color='C0', linewidth=2.0)
#x값에 대해 y=0부터 px까지 선을 그려라
plt.show()

#이항분포의 누적확률계산
# 불량품 확률이 0.1일때 20개 중 5개 이하일 확률
binom.cdf(k=5, n=20, p=0.1)
#등호 주의. 5 미만이면 k=4

#이항분포의 정규근사 연습
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt

p, n = 0.2, 30
k = np.arange(0, n+1)
px = binom.pmf(k, n=n, p=p)
plt.plot(k, px, 'C1o')
plt.plot(k, px, color='C1', linewidth=2.0)

x = np.linspace(-5, 15, 100, endpoint=True)
mu = n*p
sd = np.sqrt(n*p*(1-p))
fx=norm.pdf(x, loc=mu, scale=sd)
plt.plot(x, fx, color='C0', linewidth=1.0)
plt.title("normal distribution, p=0.2, n= %i" % n)
plt.ylabel("probability")
plt.show()
#n이 커지면 np도 달라지겟죠/

#표본평균의 분포
#정규분포가 아닌 모집단으로부터 여러 번 표본을 뽑아 정규분포를 따르는지확인
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform #균일분포 U={0,1}

np.random.seed(0)
n = 10 #1회 시행에서 추출할 표본 개수
mean = []

for i in range(1000):
    x = uniform.rvs(loc=0, scale=1, size=n)
    mean.append(x.mean())
    
plt.hist(mean, bins=9, density=True, histtype='bar')

x = np.linspace(0, 1, 100, endpoint=True)
mu = 0.5
sd = np.sqrt(1/12) #공식 잇음. var = (b-a)^2/12
fx = norm.pdf(x, loc=mu, scale=sd/np.sqrt(n))
plt.plot(x, fx, color='C3', linewidth=3.0)

plt.title("Distribution of the Sample mean n = %i" % n)
plt.xlabel("mean")
plt.ylabel("Density")
plt.show()
































