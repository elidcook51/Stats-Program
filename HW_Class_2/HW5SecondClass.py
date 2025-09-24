import Stats_Functions as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sheets = np.arange(1, 23, 1)
priorProb = (((1 / 20) - (1 / 146)) / (22-1)) * (sheets - 1) + (1 / 146)
# plt.scatter(x = sheets, y = priorProb, color = 'gray')
# plt.title('Prior Probabilty as a Function of Sheet Number')
# plt.xlabel('Sheet Number')
# plt.ylabel('Prior Probability')
# plt.savefig(stat.getDownloadsTab() + '/Prior Probability.png')
# plt.clf()

alpha0 = 5.0971
alpha1 = 16.9409
params0 = [alpha0, 0, 0]
params1 = [alpha1, 0, 0]

nums = np.arange(0.01, 50, 0.01)
df0 = stat.getdf('EX', nums, params0)
df1 = stat.getdf('EX', nums, params1)

plt.scatter(x = nums, y = df0, color = 'gray', label = 'f0(x)', s= 5)
plt.scatter(x = nums, y = df1, color = 'lightgray', label = 'f1(x)', s= 5)
plt.legend()
plt.title('Conditional density functions')
plt.xlabel('Departure')
plt.ylabel('Probability density')
plt.show()