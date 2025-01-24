# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:03:32 2024

@author: hwang
"""

#---------------ANOVA 1----------------
import numpy as np
import pandas as pd

col = np.repeat(range(1,5), np.repeat(6,4))
num = np.array([45,59,48,46,38,47,21,12,14,17,13,17,
              37,32,15,25,39,41,16,11,20,21,14,7])

bugs = pd.DataFrame({'col':col,'num':num})
import statsmodels.api as sm
from statsmodels.formula.api import ols

model1=ols('num~C(col)',data=bugs).fit() #col이 양적 자료(숫자)라 C 명령어로 질적 자료로. 이거 안하면 회귀분석함
table1=sm.stats.anova_lm(model1)
print(table1)


#------------ANOVA 2 with no repeat--------------
import numpy as np
import pandas as pd

y = np.array([64,53,47,51,49,51,45,43,50,48,50,52])
A = np.tile(range(1, 5), 3)
B = np.repeat(range(1, 4), 4)
cor = pd.DataFrame({'A':A, 'B':B, 'y':y})

import statsmodels.api as sm
from statsmodels.formula.api import ols

model2 = ols('y~C(A)+C(B)', data=cor).fit()
table2 = sm.stats.anova_lm(model2)
print(table2)


#-------------ANOVA 2 with repeat---------------
import seaborn as sns
import numpy as np
import pandas as pd

alz = pd.read_table("C:/datafolder/alzheimer.txt", sep=' ')
#alz.head()
sns.pointplot(x = "A", y = "y", hue = "B", data=alz) #avg reaction diagram

import statsmodels.api as sm
from statsmodels.formula.api import ols

model3 = ols('y~C(A)*C(B)', data=alz).fit() #C(A)*C(B) is a simple notation that makes ols operate the whole ANOVA     
table3 = sm.stats.anova_lm(model3)
print(table3)








print("I'll keep fighting until the very last moment of my life.")
print("I am a Captain Vladilena Milize, the commender of Formal Republic's Defense Forces.")
print("And I will never run from this war!")


