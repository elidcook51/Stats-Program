import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt
import math



hp = [305, 265, 300, 302, 260, 252, 285, 315, 268, 237, 306, 330, 303, 220, 240, 263, 272, 232, 268, 263, 258, 291, 237, 270, 290, 306, 260, 295, 280, 252, 260, 265, 305, 268, 250, 227, 311, 235]
hp = sorted(hp)
plottingPositions = stat.metaGaussianPlottingPositions(len(hp))


wbHp = np.log(np.array(hp) - 0)
wbPlotPos = np.log(-1 * np.log(1 - np.array(plottingPositions)))
logHp = np.log(np.array(hp) - 0)
logPlotPos = np.log(np.array(plottingPositions) / (1 - np.array(plottingPositions)))
#aHat, bHat = stat.LSregression(wbHp, wbPlotPos, True, True)
aHat, bHat = stat.LSregression(logHp, logPlotPos, True, True)
beta = 1 / bHat
alpha = np.exp(aHat)
'''
plt.scatter(x = sortedHp, y = plottingPositions)
plt.xlabel('Realizations x(n)')
plt.ylabel('Plotting positions pn')
plt.title('Realizations plotted against their plotting positions')
plt.show()
'''
eta = 0
nums = np.arange(0, 400, 0.01)

wbfrac = (np.array(nums) - eta) / alpha
wbpower = np.power(wbfrac, beta)
wbexp = np.exp(-1 * wbpower)
wbestimate = 1 - wbexp
#plt.scatter(x = nums, y = wbestimate)

llfrac = (np.array(nums) - eta) / alpha
llpower = np.power(llfrac, -1 * beta)
llestimate = np.power(1 + llpower, -1)

'''
plt.scatter(x = nums, y = llestimate)
plt.scatter(x = hp, y = plottingPositions)
plt.title('LS Estimated Log-Logistic function graphed against empirical H(x)')
plt.xlabel('Realization')
plt.ylabel('Cumulative density value')
plt.show()
'''

wbspecestimate = 1 - np.exp(-1 * np.power((np.array(hp) - eta) / alpha, beta))
llspecestimate = np.power(1 + np.power((np.array(hp) - eta) / alpha, -1 * beta), -1)
#print(stat.calcMAD(plottingPositions, wbspecestimate))

wbdensityestimate = 0.0655 * np.power(np.array(nums) / 270.8722,16.7323) * np.exp(-1 * np.power(np.array(nums) / 270.8722, 16.7323))
#plt.scatter(x = nums, y = wbdensityestimate)
#plt.title('LS Weibull density function')
#plt.show()

newA = 0.38
newB = 6
newN = 0.9
nums = np.array(np.arange(newN, 2, 0.001))
#estimateDFX = 1 - np.exp(-1 * np.power(np.log(nums - newN + 1) / newA,newB))
estimateDFY = 1 - np.exp(-1 * np.power(np.log(np.power(6*np.power(nums,3)/math.pi, 1/3) - newN + 1) / newA, newB))
#plt.scatter(x = nums, y = estimateDFX)
#plt.title('Distribution Function for diameter of a drop plotted on (0.9, 2)')
#plt.xlabel('Diameter of a drop (mm)')
#plt.ylabel('Cumulative probability of diameter')
plt.scatter(x = nums, y = estimateDFY)
plt.title('Distribution Function for volume of a drop plotted on (0.9, 2)')
plt.xlabel('Volume of a drop (mm^3)')
plt.ylabel('Cumulative probability of volume')
#estimatedfX = (newB / (newA * (nums - newN +1))) * np.power(np.log(nums - newN + 1) / newA, newB - 1) * np.exp(-1 * np.power(np.log(nums - newN + 1) / newA, newB))
#estimatedfY = ((1/3) * np.power(6 / math.pi, 1/3) * np.power(nums, -2/3)) * (newB / (newA * (np.power(6 * nums / math.pi, 1/3) - newN + 1))) * np.power(np.log(np.power(6 * nums / math.pi, 1/3)+ 0.1) / 0.38, newB - 1) * np.exp(-1 * np.power(np.log(np.power(6 * nums / math.pi, 1/3) + 0.1) / 0.38,newB))
#plt.scatter(x = nums, y = estimatedfX)
#plt.title('Density Function for diameter of a drop plotted on (0.9, 2)')
#plt.xlabel('Diameter of a drop (mm)')
#plt.ylabel('Density of given diameter')
#plt.scatter(x = nums, y = estimatedfY)
#plt.title('Density Function for volume of a drop plotted on (0.9, 2)')
#plt.xlabel('Volume of a drop (mm^3)')
#plt.ylabel('Density of given volume')
plt.show()