import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Creating the new data
# data = pd.read_csv("C:/Users/ucg8nb/Downloads/Table 3.11.csv")
# lowerBoundList = data['Lower Bound'].tolist()
# upperBoundList = data['Upper Bound'].tolist()
# counts = data['Count'].tolist()
#
# newData = []
# for i in range(len(counts)):
#     L = lowerBoundList[i]
#     C = counts[i]
#     for n in range(C):
#         newData.append(L + 50/C * (n+1))
# tempDf = pd.DataFrame(newData)
# tempDf.to_csv(stat.getDownloadsTab() + '/New Data.csv')

newData = pd.read_csv("C:/Users/ucg8nb/Downloads/New Data.csv", header = None)
newData = newData[0].tolist()

# outputDf = stat.findUDFit(newData, -500, 2100, numSteps = 100)
# outputDf.to_csv(stat.getDownloadsTab() + '/OutputDf Updated MAD.csv')


plottingPositions = stat.metaGaussianPlottingPositions(len(newData))
distDf = pd.read_csv("C:/Users/ucg8nb/Downloads/OutputDf Updated MAD.csv")
nums = np.arange(-500, 2100, 0.05)
# stat.plotDistributions(distDf, nums, newData)

# stat.plotDistributionFromDf(distDf, 'LP', nums, data = newData, title = 'Laplace Distribution Function', xlabel = 'Stock Return', ylabel = 'Cumulative Probability Density', savefig = stat.getDownloadsTab() + '/Laplace Disitrubtion Function.png')
# stat.plotDistributionFromDf(distDf, 'LL', nums, data = newData, title = 'Log-Logistic Distribution Function', xlabel = 'Stock Return', ylabel = 'Cumulative Probability Density', savefig = stat.getDownloadsTab() + '/Log-Logistic Disitrubtion Function.png')
# stat.plotDistributionFromDf(distDf, 'LRLP', nums, data = newData, title = 'Log-ratio Laplace Distribution Function', xlabel = 'Stock Return', ylabel = 'Cumulative Probability Density', savefig = stat.getDownloadsTab() + '/Log-Ratio Laplace Disitrubtion Function.png')
# stat.plotDistributionFromDf(distDf, 'LL', nums, DF = False, title = 'Log-ratio Laplace Density Function', xlabel = 'Stock Return', ylabel = 'Probability Density', savefig = stat.getDownloadsTab() + '/Log-Ratio Laplace Density Function.png')
# params = [454.1344, 5.0710, -500]
# quantilePoints = [0.25, 0.5, 0.75]
# quantilePoints = np.array(quantilePoints)
# ranges = [-250, -50, 0, 50, 250, 500]
# ranges = np.array(ranges)
# quantiles = stat.inverseDist('LL', quantilePoints, params)
# print(quantiles)
# events = stat.getDF('LL', ranges, params)
# print(events)
# print(1 - events)


#Plotting the Empirical Distribution function
# plt.scatter(newData, plottingPositions, color = 'gray', s = 3)
# plt.title('Empirical Distribution of Stock Returns')
# plt.xlabel("Stock Return")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + '/Empirical Distribution of Stock Returns.png')

