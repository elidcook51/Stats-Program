import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math

N = 21
sp = stat.standardPlottingPositions(N)
sp[-1] -= 0.0001
sp[0] += 0.0001
wp = stat.WeibullPlottingPositions(N)
mgp = stat.metaGaussianPlottingPositions(N)
inverseStandard = []
for p in mgp:
    inverseStandard.append(stat.inverseStandardNormal(p))
#print(stat.sampleMean(inverseStandard))
#print(stat.sampleVariance(inverseStandard, False))

#A = 1.18
#B = 0.5
#E = 110
#E2 = 12100
#nums = np.array(np.arange(E, 150, 0.0005))
#nums2 = np.array(np.arange(E2, 14000, 0.01))
#estimateDFX = np.exp(-1 * np.power(A / (nums - E), B))
#plt.scatter(x = nums, y = estimateDFX)
#plt.title('Distribution Function for length of a plate side plotted on (110, 500)')
#plt.xlabel('Length of a side (mm)')
#plt.ylabel('Cumulative probability')
#estimateDFY = np.exp(-1 * np.power(A / (np.power(nums2,1/2) - E), B))
#plt.scatter(x = nums2, y = estimateDFY)
#plt.title('Distribution Function for area of a plate plotted on (12100, 50000)')
#plt.xlabel('Area of a plate (mm^2)')
#plt.ylabel('Cumulative probability')
#plt.ylim(0, 1)
#estimatedfX = (B/A) * np.power(A / (nums - E), B + 1) * np.exp(-1 * np.power(A / (nums - E),B))
#plt.scatter(x = nums, y = estimatedfX)
#plt.title('Density function for length of a plate side plotted on (110, 150)')
#plt.xlabel('Length of a side (mm)')
#plt.ylabel('Probability density')
#estimatedfY = (1 / (2 * np.power(nums2, 1/2))) * (B/A) * np.power(A / (np.power(nums2, 1/2) - E), B+1) * np.exp(-1 * np.power(A / (np.power(nums2, 1/2) - E), B))
#plt.scatter(x = nums2, y = estimatedfY)
#plt.title('Density function for area of a plate side plotted on (12100, 14000)')
#plt.xlabel('Area of a plate (mm^2)')
#plt.ylabel('Probability density')
#plt.show()

d = [95, 49, 78, 84, 55, 64, 57, 52, 61, 58, 71, 74, 53, 69, 60, 56, 67, 62, 54, 66, 47, 51, 59, 55, 63]
d = sorted(d)
pn = stat.metaGaussianPlottingPositions(len(d))
#plt.scatter(x = d, y = pn)
#plt.title("Empirical distribution function")
#plt.xlabel("Diameter of a log")
#plt.ylabel("Plotting positions for each diameter")
#plt.show()
E = 30
wbD = np.log(np.log(np.array(d) - E + 1))
wbPn = np.log(-1 * np.log(1 - np.array(pn)))
logD = np.log(np.array(d) - E)
logPn = np.log(np.array(pn) / (1 - np.array(pn)))
#aHat, bHat = stat.LSregression(wbD, wbPn, True, True)
aHat, bHat = stat.LSregression(logD, logPn, True, True)
beta = 1 / bHat
alpha = np.exp(aHat)
nums = np.arange(E, 100, 0.01)
logEstimate = np.power(1 + np.power((nums - E) / alpha , -1 * beta) ,-1)
wbEstimate = 1 - np.exp(-1 * np.power(np.log(nums - E + 1) / alpha , beta))
#logDensityEstimate = (beta / alpha) * np.power((nums - E) / alpha, -1 * beta - 1) * np.power(1 + np.power((nums - E) / alpha, -1 * beta), -2)
#plt.scatter(x = nums, y = logEstimate, color = 'lightgray')
#plt.scatter(x = nums, y = wbEstimate, color = 'lightgray')
#plt.scatter(x = d, y = pn, color = 'dimgray')
#plt.title('Log-Weibull estimated curve versus empirical distribution function')
#plt.xlabel('Diameter of a tree trunk')
#plt.ylabel("Cumulative probability")
#plt.show()
#plt.scatter(x = nums, y = logDensityEstimate, color = 'dimgray')
#plt.title("Estimate Log-Logistic density function")
#plt.xlabel("Diameter of tree trunk")
#plt.ylabel("Probability density")
#plt.show()
eta = 85458.24
l = np.power(94.9535,1/2)
#nums2 = np.arange(eta, 1000000, 10)
#logEstimateY = np.power(1 + np.power((np.power(nums2,0.5) / l - E) / alpha , -1 * beta) ,-1)
#plotNums = nums2 / 10000
#nums3 = np.power(nums2,0.5) / l
#logEstimateDensityY = (1 / (l * 2 * np.power(nums2, 1/2))) * (beta / alpha) * np.power((nums3 - E) / alpha, -1 * beta - 1) * np.power(1 + np.power((nums3 - E) / alpha, -1 * beta), -2)
#plt.scatter(x = plotNums, y = logEstimateDensityY, color = 'dimgray')
#plt.scatter(x = plotNums, y = logEstimateY, color = 'dimgray')
#plt.title('Distribution Function for weight of a truck on (8.5, 100)')
#plt.xlabel('Weight of a truck (t)')
#plt.ylabel('Cumulative Probability')
#plt.show()
#logEstimateSpec = np.power(1 + np.power((np.array(d) - E) / alpha , -1 * beta) ,-1)
#print(stat.calcKStest(logEstimateSpec))
#print(stat.calcMAD(pn, logEstimateSpec))
#print(len(d))
print(np.power(1 + np.power((np.power(950000,0.5) / l - E) / alpha , -1 * beta) ,-1))

