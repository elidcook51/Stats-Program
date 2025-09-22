import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def width(cred, T):
    p = (-1* (cred - 1)) / 2
    z = stat.inverseStandardNormal(1 - p)
    return z * T

z1 = stat.inverseStandardNormal(0.75)
z2 = stat.inverseStandardNormal(0.99)

A = 0.6449
B = 193.9170
T = 71.5062
S = 120
M = 780

nums = np.arange(420,1140, 0.01)

'''nums = np.arange(336.1557, 990.6443, 0.01)
median = A * nums + B
top50 = A * nums + B + width(0.5, T)
bottom50 = A * nums + B - width(0.5, T)
top98 = A * nums + B + width(0.98, T)
bottom98 = A * nums + B - width(0.98,T)
plt.scatter(x = nums, y = median, color = 'gray', label = 'Posterior mean')
plt.scatter(x = nums, y = top50, color = 'darkgray')
plt.scatter(x = nums, y = bottom50, color = 'darkgray', label = '50% credible interval')
plt.scatter(x = nums, y = top98, color = 'lightgray')
plt.scatter(x = nums, y = bottom98, color = 'lightgray', label = '98% credible interval')
plt.title('Posterior mean of W as a function of x')
plt.xlabel('Realization of X')
plt.ylabel('Posterior mean')
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/Posterior means of W.png')'''

'''prior = (1 / (S * np.power(2 * math.pi, 1/2))) * np.exp((-1/2) * np.power((nums - M)/ S, 2))
post600 = (1 / (T * np.power(2 * math.pi, 1/2))) * np.exp((-1/2) * np.power((nums - A * 600 - B)/ T, 2))
post900 = (1 / (T * np.power(2 * math.pi, 1/2))) * np.exp((-1/2) * np.power((nums - A * 900 - B)/ T, 2))

plt.scatter(x = nums, y = prior, color = 'gray', label = 'Prior probability')
plt.scatter(x = nums, y = post600, color = 'lightgray', label = 'Posterior probability for X = 600')
plt.scatter(x = nums, y = post900, color = 'darkgray', label = 'Posterior probability for X = 900')
plt.legend()
plt.title('Posterior and prior density functions for W')
plt.ylabel('Probability density')
plt.xlabel('Realization of W')
plt.savefig(stat.getDownloadsTab() + '/Density functions for W.png')'''

'''priorDf = stat.standardNormalArray((nums - M) / S)
post600Df = stat.standardNormalArray((nums - A * 600 - B) / T)
post900Df = stat.standardNormalArray((nums - A * 900 - B) / T)

plt.scatter(x = nums, y = priorDf, color = 'gray', label = 'Prior')
plt.scatter(x = nums, y = post600Df, color = 'lightgray', label = 'Posterior with X = 600')
plt.scatter(x = nums, y = post900Df, color = 'darkgray', label = 'Posterior with X = 900')
plt.legend()
plt.title('Prior and Posterior distribution functions for W')
plt.xlabel('Realization of W')
plt.ylabel('Cumulative probability')
plt.savefig(stat.getDownloadsTab() + '/Distribution Function for W.png')'''
