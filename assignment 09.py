# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:39:28 2024

@author: hwang
"""

#(1)
import pandas as pd
import seaborn as sns

handspan = pd.read_table("C:/datafolder/handspan.txt", sep= '\t')

print(handspan.loc[:,['Height','HandSpan']].corr(method='pearson'))
sns.lmplot(x='Height', y='HandSpan',data=handspan)

#(2)
from scipy.stats import pearsonr
pearsonr(handspan['Height'], handspan['HandSpan'])

#(3)
from statsmodels.formula.api import ols

model=ols("HandSpan ~Height", handspan).fit()
print(model.summary())

#(4)
import matplotlib.pyplot as plt

fitted= model.fittedvalues
residual =model.resid
rstandard= model.resid_pearson
pd.DataFrame({'Fitted':fitted,'Residual':residual, 'Rstandard':rstandard })

plt.scatter(x=fitted, y=rstandard)
plt.axhline(y=0)
plt.title("studentized residuals vs fitted values")
plt.ylabel("studentized residuals")
plt.xlabel("fitted values")
plt.show()


#2-(1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hospital = pd.read_table("C:/datafolder/hospital.txt", sep='\t')
sns.pairplot(data=hospital[['Stay', 'Age', 'Xray', 'InfctRsk']])
plt.show()
print(hospital[['Stay', 'Age', 'Xray', 'InfctRsk']].corr(method='pearson'))

#2-(2)
from statsmodels.formula.api import ols

model= ols("InfctRsk ~ Stay + Age + Xray", hospital).fit()
print(model.summary())

#2-(3)
fitted= model.fittedvalues
residual =model.resid
rstandard= model.resid_pearson
pd.DataFrame({'Fitted':fitted,'Residual':residual, 'Rstandard':rstandard })

plt.scatter(x=fitted, y=rstandard)
plt.axhline(y=0)
plt.title("studentized residuals vs fitted values")
plt.ylabel("studentized residuals")
plt.xlabel("fitted values")
plt.show()

