# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 01:47:30 2024

@author: hwang
"""

#1
#45 20 20 15
#92 69 32 42 235

#null hypo : job is same

import numpy as np
from scipy.stats import chisquare

job = np.array([92, 69, 32, 42])
P0 = np.array([0.45, 0.2, 0.2, 0.15])*np.array([sum(job)])
print(P0)
print(chisquare(job, P0))

#2
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

y=np.array([[288, 378],[10, 7], [61, 62]])
table =pd.DataFrame(y,['알고 있다', '잘 모른다', '전혀 모른다'], ['male', 'female'])
print(table)

chi2, pval, df, expected =chi2_contingency(table)
print("Chi2 =",chi2)
print("p-value =",pval)
print("df =",df)
print("Expected value =",expected)

#3
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

y=np.array([[11, 39], [14, 26]])
table =pd.DataFrame(y,['혈액 희석제 사용안함', '혈액 희석제 사용함'], ['alive', 'dead'])
print(table)

chi2, pval, df, expected =chi2_contingency(table)
print("Chi2 =",chi2)
print("p-value =",pval)
print("df =",df)
print("Expected value =",expected)