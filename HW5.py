import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math

soybeans = [45.03, 47.30, 50.37, 50.92, 51.00, 48.07, 45.97, 48.42, 44.70, 50.32, 52.42, 51.33, 50.28, 53.72, 52.15, 49.32, 51.27, 51.83, 48.28, 55.03]
soybeans = sorted(soybeans)
pn = stat.metaGaussianPlottingPositions(len(soybeans))

#plt.scatter(x = soybeans, y = pn, color = 'grey')
#plt.title("Empirical distribution function for soybean yield")
#plt.xlabel("Soybean yield (bags/hectare)")
#plt.ylabel("Cumulative distribution")
#plt.show()

low = 44
up = 58
v = np.log(-1 * (up-low) / (np.array(soybeans) - up) - 1)
u = np.log(-1 * np.log(1 - np.array(pn)))
aHat, bHat = stat.LSregression(v, u, True, True)
alpha = bHat
beta = aHat
nums = np.arange(low+0.01, up, 0.01)
estimate = 1 - np.exp(-1 * np.exp((np.log((nums - low) / (up - nums)) - beta) / (alpha) ))
#plt.scatter(x = soybeans, y = pn, color = 'gray')
#plt.scatter(x = nums, y = estimate, color = 'lightgray')
#plt.title("LR-RG estimated curve versus empirical distribution function")
#plt.xlabel("Yield of soybeans (bags/hectare)")
#plt.ylabel("Cumulative distribution")
#plt.show()

#denestimate = ((up - low) / (alpha * (nums - low) * (up - nums))) * np.exp((np.log((nums - low) / (up - nums)) - beta) / (alpha)) * np.exp(-1 * np.exp((np.log((nums - low) / (up - nums)) - beta) / (alpha)))
#plt.scatter(x = nums, y = denestimate, color = 'gray')
#plt.title('Estimated LR-RG density function')
#plt.xlabel('Yield of soybeans (bags/hectare)')
#plt.ylabel('Probability denstiy')
#plt.show()
#subEstimate = 1 - np.exp(-1 * np.exp((np.log((np.array(soybeans) - low) / (up - np.array(soybeans))) - beta) / (alpha)))
#print(stat.calcKStest(subEstimate))
#print(len(soybeans))

percentiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
#up = 0.8616
#low = 0.6536
#quantiles = up - (up - low) / (np.exp(beta + alpha * np.log(-1 * np.log(1 - percentiles))) + 1)
quantiles = (up - (up - low) / (np.exp(beta + alpha * np.log(-1 * np.log(1 - percentiles)))+1)) / 59.9
print(quantiles)