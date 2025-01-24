# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:43:49 2024

@author: hwang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ames = pd.read_csv("C:/datafolder/ch05_data/ames.csv")

# 1
sale_price = ames['SalePrice']
plt.hist(sale_price, bins=30, color="C0", histtype='bar')
plt.title("Historgam of SalePrice")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

print(ames['SalePrice'].describe())
print(np.var(sale_price)/5000)

#2
sample = np.random.choice(ames['SalePrice'], size=50, replace=False)
print(sample.mean())

#3
sample_mean50 = []

for _ in range(5000):
    sample = np.random.choice(ames['SalePrice'], size=50, replace=False)
    sample_mean50.append(sample.mean())

sample_mean50 = pd.Series(sample_mean50)    
plt.hist(sample_mean50, bins=30, color="C0", histtype='bar')

#4
print(np.mean(sample_mean50))
print(np.var(sample_mean50))

print(6381883615.688427 / 50)

#5
sample_mean150 = []

for _ in range(5000):
    sample = np.random.choice(ames['SalePrice'], size=150, replace=False)
    sample_mean150.append(sample.mean())

sample_mean150 = pd.Series(sample_mean150)

plt.hist(sample_mean150, bins=30, color="Orange", histtype='bar')




# 데이터 로드
data_path = "C:\\datafolder\\ch05_data\\ames.csv"
df = pd.read_csv(data_path)

# SalePrice 변수에 대한 히스토그램 및 요약 통계
def example_1():
    sale_price = df['SalePrice']
    
    # 히스토그램
    plt.hist(sale_price, bins=30, edgecolor='k', alpha=0.7)
    plt.title("SalePrice Distribution")
    plt.xlabel("Sale Price")
    plt.ylabel("Frequency")
    plt.show()
    
    # 수치적 요약
    mean_price = sale_price.mean()
    median_price = sale_price.median()
    variance_price = sale_price.var()
    stddev_price = sale_price.std()
    
    print("평균:", mean_price)
    print("중앙값:", median_price)
    print("분산:", variance_price)
    print("표준편차:", stddev_price)

# 50개의 랜덤 표본을 선택하여 표본 평균 계산
def example_2():
    sample_50 = df['SalePrice'].sample(50)
    sample_mean_50 = sample_50.mean()
    print("표본 평균 (n=50):", sample_mean_50)

# 표본 크기 50, 반복 5000번의 표본 평균 분포 계산 및 히스토그램
def example_3():
    sample_means_50 = [df['SalePrice'].sample(50).mean() for _ in range(5000)]
    
    # 히스토그램
    plt.hist(sample_means_50, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Sample Mean Distribution (n=50)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()
    
    return sample_means_50

# sample_mean50의 평균과 분산 계산
def example_4(sample_means_50):
    mean_sample_means_50 = np.mean(sample_means_50)
    var_sample_means_50 = np.var(sample_means_50)
    
    print("표본 평균의 평균:", mean_sample_means_50)
    print("표본 평균의 분산:", var_sample_means_50)
    print("모집단 평균과의 관계:", mean_sample_means_50, "(모집단 평균에 근접)")
    print("표본 평균 분산과 모분산 관계:", var_sample_means_50, "(모분산 / 50에 근접)")

# 표본 크기 150, 반복 5000번의 표본 평균 분포 계산 및 히스토그램
def example_5():
    sample_means_150 = [df['SalePrice'].sample(150).mean() for _ in range(5000)]
    
    # 히스토그램
    plt.hist(sample_means_150, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Sample Mean Distribution (n=150)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()
    
    return sample_means_150

# 각 예제 실행
example_1()
example_2()
sample_means_50 = example_3()
example_4(sample_means_50)
sample_means_150 = example_5()
