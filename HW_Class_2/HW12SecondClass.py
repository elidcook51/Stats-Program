import Stats_Functions as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# colNames = ['Optimal decision $a^*$', 'WB(1, 5.5, 0)']

# decisionTypes = ['0.2-quantile of W' ,'0.6321-quantile of W', '0.8-quantile of W', 'Median of W', 'Mean of W', 'Mode of W']


# params = [1, 5.5 ,0]
# nums = np.arange(0.00001, 1.9, 0.000001)
# wbDF = stat.getDF("WB", nums, params)
# wbdf = stat.getdf('WB', nums, params)

# maxIndex = np.where(wbdf == np.max(wbdf))
# print(nums[maxIndex])

# figName = 'Density function for W.png'
# plt.scatter(x = nums, y =wbdf, color = 'gray', s = 5)
# plt.title("Density function for W")
# plt.xlabel("Realization of W")
# plt.ylabel("Probability Density")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(12, figName)


# quantiles = [0.2, 0.6321, 0.8, 0.5]
# first4 = [stat.inverseDist('WB', np.array(p), params) for p in quantiles]
# first4.append(0.9232)
# first4.append(np.max(wbdf))

# stat.printInLatexTable([decisionTypes, first4], colNames)

params1 = [1.5, 0.6]
params2 = [3.6, 0.9]

nums = np.arange(-14, 10, 0.001)

p1 = 0.7

g1 = stat.getdf('GB',nums, params1)
g2 = stat.getdf('RG', nums, params2)
g = p1 * g1 + (1 - p1) * g2

pvals = [0.3, 0.7]

gvals = [p * g1 + (1 - p) * g2 for p in pvals]

maxgindex = [np.where(gs == np.max(gs)) for gs in gvals]

numindex = [nums[ind] for ind in maxgindex]

maxgvals = [np.max(gs) for gs in gvals]

figNames = ['p1 point 3.png', 'p1 point 7.png']

for i in range(len(gvals)):
    plt.scatter(x = nums, y = gvals[i], color = 'lightgray', s= 5)
    plt.scatter(x = numindex[i], y = maxgvals[i], color = 'gray', s =120, label = 'Optimal Aiming Point')
    plt.title("Density function with optimal aiming point indicated")
    plt.legend()
    plt.xlabel("Aiming points")
    plt.ylabel("Probability Density")
    plt.savefig(stat.getDownloadsTab() + figNames[i])
    stat.createFigureLatex(12, figNames[i])
    plt.clf()

# colNames = ['$p_1$ value', '$a^*$', '$U^*$']

# stat.printInLatexTable([pvals, numindex, maxgvals], colNames)


# figName = 'Second p1 density.png'
# plt.scatter(x = nums,y = g1, color = 'lightgray', s = 5, label = 'g1')
# plt.scatter(x = nums, y = g2, color = 'darkgray', s = 5, label = 'g2')
# plt.scatter(x = nums, y = g, color = 'gray', s =5, label = 'g')
# plt.legend()
# plt.title("Density functions")
# plt.xlabel("Radar Location")
# plt.ylabel("Probability Density")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(12, figName)


# params = [196, 2.8, 280]

# a = 570

# aDF = stat.getDF('IW', np.array(a), params)

# print(aDF)
# print(1 / aDF  - 1)

# params = [0, 2, 30, 90]

# lambdao = 0.71
# lambdau = 0.7875

# nums = np.arange(30.001, 89.999, 0.001)
# p1df = stat.getDF('P1', nums, params)

# astar = stat.inverseDist("P1", 1 / (1 + lambdao /lambdau) , params)
# print(1 - 1 / (1 + lambdao / lambdau))
# dw = 0.0000001
# nums1 = np.arange(30 + dw, astar, dw)
# nums2 = np.arange(astar, 90 - dw, dw)

# df1 = stat.getdf('P1', nums1, params)
# df2 = stat.getdf('P1', nums2, params)

# int1 = np.sum(df1 * (1.21 * nums1 - 0.5 * astar) * dw)
# int2 = np.sum(df2 * (0.71 * astar) * dw)

# print(int1 + int2)

# totNums = np.arange(30 + dw, 90 - dw, dw)

# totdf = stat.getdf('P1', totNums, params)

# totint = np.sum(totdf * (0.71 * totNums) * dw)

# print(totint)

# print(astar)

# figName = 'astar plot.png'
# plt.scatter(x = nums, y = p1df, color = 'lightgray', s  =5)
# plt.scatter(x = astar, y = stat.getDF("P1", np.array(astar), params), color = 'gray', s = 120, label  = 'Optimal Order')
# plt.legend()
# plt.title("Distribution function of demand with optimal baking amount")
# plt.xlabel("Demand")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(12, figName)