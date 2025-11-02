import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt

quantileProbs = [0.125, 0.25, 0.5, 0.75, 0.875]
quantileVals = [1200, 2100, 3100, 4200, 5400]

# plt.scatter(quantileVals, quantileProbs, color = 'gray', s = 25)
# plt.xlabel('Scooter Demand')
# plt.ylabel("Cumulative Probability")
# plt.title("Pointwise Representation of Judgemental Distribution Function")
# plt.ylim(0,1)
# plt.savefig(stat.getDownloadsTab() + '/Pointwise DF.png')
# stat.printInLatexTable([quantileProbs, quantileVals], ['Quantile Probabilities', 'Judgemental Assessment'])
# stat.createFigureLatex(9, '/Pointwise DF.png', 0.65)

# outputDf = stat.findUDFit(quantileVals, 0, 7000, plottingPositions = quantileProbs)
# outputDf.to_csv(stat.getDownloadsTab() + '/Fit Curves.csv')
eta = 0
mu, sigma = stat.quantileMethod5(quantileProbs, quantileVals)
# nums = np.arange(0.1, 7000, .1)
nums = np.array(quantileVals)
realOutputs = np.array(quantileProbs)
estimates = stat.getDF('LN', nums, [sigma, mu, eta])
print(estimates)
print(realOutputs)
print(np.abs(estimates, realOutputs))
