# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:00:17 2024

@author: hwang
"""

import pandas as pd
cdc = pd.read_csv("C:/datafolder/cdc.txt", delimiter=" ", skipinitialspace=True)

#1-1
genhlth_data = cdc['genhlth'].value_counts()
print(genhlth_data)

#1-2
table = cdc.groupby("genhlth").size()
table.plot(kind='bar',x=table.index, y=table.values, title='barchart of genhlth')

#1-3
import matplotlib.pyplot as plt
plt.pie(genhlth_data,labels=genhlth_data.index,autopct='%1.1f%%')
plt.title('piechart of genhlth')
plt.show() 

#2
print(cdc['weight'].describe())

#3
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(cdc['weight'], cdc['wtdesire'], s=8)
plt.title("weight_wtdesire scatter")
plt.xlabel('Weight')
plt.ylabel('Desired Weight')
plt.show()
print(np.corrcoef(cdc['weight'], cdc['wtdesire']))

#4
import matplotlib.pyplot as plt
cdc['wdiff'] = cdc['weight'] - cdc['wtdesire']
print(cdc['wdiff'].describe())

plt.hist(cdc['wdiff'], bins = 100)
plt.title('Histogram of wdiff')
plt.xlabel('wdiff')
plt.ylabel('Frequency')
plt.show()

#5
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(cdc['age'])
plt.title('Histogram of Age')

# 구간 50
plt.subplot(1, 3, 2)
plt.hist(cdc['age'], bins=50)
plt.title('Histogram of Age (Bins=50)')
# 구간 100
plt.subplot(1, 3, 3)
plt.hist(cdc['age'], bins=100)
plt.title('Histogram of Age (Bins=100)')
    
plt.show()




#--------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 파일 불러오기
file_path = "C:/datafolder/cdc.txt"
cdc = pd.read_csv(file_path, delimiter=" ", skipinitialspace=True)

# 예제 1: genhlth 변수에 대해 범주형 자료 요약
def summarize_genhlth(cdc):
    genhlth_counts = cdc['genhlth'].value_counts()
    print("Genhlth Frequency:\n", genhlth_counts)
    
    # 막대그래프 그리기
    sns.countplot(x='genhlth', data=cdc, order=cdc['genhlth'].value_counts().index)
    plt.title('Distribution of General Health (genhlth)')
    plt.show()

summarize_genhlth(cdc)

# 예제 2: weight 변수에 대한 수치적 요약
def summarize_weight(cdc):
    weight_mean = cdc['weight'].mean()
    print("Average Weight:", weight_mean)
    
    # 수치적 요약 출력
    print(cdc['weight'].describe())

summarize_weight(cdc)

# 예제 3: weight 변수와 wtdesire 변수의 산점도 및 상관계수
def plot_weight_vs_wtdesire(cdc):
    sns.scatterplot(x='weight', y='wtdesire', data=cdc)
    plt.title('Scatter Plot: Weight vs Desired Weight')
    plt.xlabel('Weight (lbs)')
    plt.ylabel('Desired Weight (lbs)')
    plt.show()
    
    # 상관계수 계산
    correlation = cdc['weight'].corr(cdc['wtdesire'])
    print("Correlation between Weight and Desired Weight:", correlation)

plot_weight_vs_wtdesire(cdc)

# 예제 4: wdiff 변수 생성 및 분포 요약
def analyze_wdiff(cdc):
    cdc['wdiff'] = cdc['weight'] - cdc['wtdesire']
    
    # 수치적 요약
    print("wdiff Summary:\n", cdc['wdiff'].describe())
    
    # 히스토그램 그리기
    plt.hist(cdc['wdiff'], bins=20, edgecolor='black')
    plt.title('Distribution of wdiff (Weight - Desired Weight)')
    plt.xlabel('Difference between Weight and Desired Weight')
    plt.ylabel('Frequency')
    plt.show()

analyze_wdiff(cdc)

# 예제 5: age 변수의 히스토그램 그리기 (구간 수 50, 100 비교)
def plot_age_histogram(cdc):
    plt.figure(figsize=(10, 5))
    
    # 구간 50인 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(cdc['age'], bins=50, edgecolor='black')
    plt.title('Age Distribution (Bins=50)')
    
    # 구간 100인 히스토그램
    plt.subplot(1, 2, 2)
    plt.hist(cdc['age'], bins=100, edgecolor='black')
    plt.title('Age Distribution (Bins=100)')
    
    plt.show()

plot_age_histogram(cdc)
