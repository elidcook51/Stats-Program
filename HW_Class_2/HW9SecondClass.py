import Stats_Functions as stat
import numpy as np
import matplotlib.pyplot as plt

# quantileProbs = [0.125, 0.25, 0.5, 0.75, 0.875]
# quantileVals = [1200, 2100, 3100, 4200, 5400]

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
# eta = 0

# stat.printInLatexTable([quantileProbs, quantileVals, np.log(quantileVals)], ['$p$', '$y_p$', '$x_p = \ln(y_p-\eta)$'])

# mu, sigma = stat.quantileMethod5(quantileProbs, np.log(quantileVals))
# print(mu, sigma)
# nums = np.arange(1, 10000, 1)
# nums = np.array(quantileVals)
# realOutputs = np.array(quantileProbs)
# quantileEstimates = stat.getDF('LN', np.array(quantileVals), [sigma, mu, 0])
# distances = np.abs(realOutputs - quantileEstimates)
# stat.printInLatexTable([quantileVals, quantileProbs, quantileEstimates, distances], ['Judgemental Value', 'Judgemental Quantile', 'Distribution Quantile', 'Distance'])
# estimates = stat.getDF('LN', nums, [sigma, mu, 0])
# figName = '/Parametric Vs Pointwise.png'
# plt.scatter(x = nums, y = estimates, color = 'lightgray', s = 10)
# plt.scatter(x = quantileVals, y = quantileProbs, color = 'gray', s = 25)
# plt.title("Parametic Model versus Pointwise Distribution Function")
# plt.xlabel("Scooter Demand")
# plt.ylabel("Cumulative Probability")
# plt.ylim(0,1)
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(9, figName, 0.65)
# density = stat.getdf('LN', nums, [sigma, mu, 0])
# figName = '/Density demand parametric.png'
# plt.scatter(x = nums, y = density, color = 'gray', s = 10)
# plt.title("Density Function of Parametric Model")
# plt.xlabel("Scooter Demand")
# plt.ylabel("Probability Density")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(9, figName, 0.65)

# print(stat.getMAD(quantileVals, quantileProbs, "LN", [sigma, mu, 0]))

# params = [22, 0.81, 61]
# quantiles = np.array([0.2, 0.5 ,0.8])
# wp = stat.inverseWB(quantiles, params[0], params[1], params[2])
# yp = wp + np.array([45, 65, 95])

# wbfit = stat.findUDFitDist('WB' ,yp, lowerBound = 90, numSteps = 10000, plottingPositions = quantiles)
# # print(wbfit)
# newParams = [69.2852, 1.1621, 90]
# revised = stat.inverseDist('WB', quantiles, newParams)
# print(revised)
# stat.printInLatexTable([quantiles,wp, yp, revised], ['Probability Values', 'Origional Quantiles', 'Adjusted Quantiles', 'Revised Quantiles'])
# stat.printInLatexTable([quantiles, wp, yp], ['Probability Values', 'Origional Quantiles', 'Adjusted Quantiles'])

# nums = np.arange(50, 300, 0.01)
# oldDist = stat.getDF('WB', nums, params)
# newDist = stat.getDF('WB', nums, newParams)
# figName = '/Original vs Revised.png'
# plt.scatter(x = nums, y = oldDist, color = 'darkgray', s = 10, label = 'Original Distribution')
# plt.scatter(x = nums, y = newDist, color = 'lightgray', s = 10, label = 'Revised Distribution')
# plt.scatter(x = yp, y = quantiles, color = 'gray', s = 45, label = 'Adjusted points')
# plt.title("Original and Revised Distributions")
# plt.legend()
# plt.xlabel("Scooter Demand")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(9, figName, 0.5)

# print(stat.inverseDist('WB', np.array([0.9]), params))
# print(stat.inverseDist('WB', np.array([0.9]), newParams))

probs = [0.05, 0.25, 0.5, 0.75, 0.95]
accountantA = [7,15,19,22,26]
accountantB = [-19,-3,13,36,59]
accountantC = [np.nan, 12,23,28,43]
accountantD = [-8,4,11,20,np.nan]

medianVal = [np.nanmedian([accountantA[i], accountantB[i], accountantC[i], accountantD[i]]) for i in range(len(accountantA))]

# colNames = ['Probabilities', 'Accountant A', 'Accountant B', 'Accountant C', 'Accountant D', 'Median']
# stat.printInLatexTable([probs, accountantA, accountantB, accountantC, accountantD, medianVal], colNames)

# figName = '/Pointwise distributions all accountants.png'

# plt.plot(accountantA,probs, color = 'lightgray')
# plt.scatter(accountantA,probs, label = 'Accountant A', color = 'lightgray')

# plt.plot(accountantB,probs, color = 'darkgray')
# plt.scatter(accountantB,probs, label = 'Accountant B', color = 'darkgray')

# plt.plot(accountantC,probs, color = 'gray')
# plt.scatter(accountantC,probs, label = 'Accountant C', color = 'gray')

# plt.plot(accountantD,probs, color = 'dimgray')
# plt.scatter(accountantD,probs, label = 'Accountant D', color = 'dimgray')

# plt.plot(medianVal,probs, color = 'black')
# plt.scatter(medianVal,probs, label = 'Group Value', color = 'black')
# plt.legend()
# plt.title('Pointwise judgemental distribution functions')
# plt.xlabel("Value of the company")
# plt.ylabel("Cumulative probability")
# plt.ylim(0,1)
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(9, figName, 0.75)

l = -25
u = 65

yp = np.array(medianVal)
zp = stat.inverseStandardNormalArray(np.array(probs))
xp = np.log((yp - l) / (u - yp))


anm, bnm = stat.LSregression(yp, zp, True, True)
nmparams = [bnm, anm]

alr, blr = stat.LSregression(xp, zp, True, True)
lrparams = [blr, alr, l, u]

nums = np.arange(-24.9, 64.9, 0.1)
normalDF = stat.getDF('NM', nums, nmparams)
normaldf = stat.getdf('NM', nums, nmparams)

# figName = '/Parametric normal.png'
# plt.scatter(x = nums, y = normalDF, color = 'lightgray', s= 10)
# plt.scatter(x = yp, y = probs, color = 'gray', s= 45)
# plt.title("Parametric normal distribution")
# plt.xlabel("Company Value")
# plt.ylabel("Cumulative Probability")
# plt.savefig(stat.getDownloadsTab() + figName)
# stat.createFigureLatex(9, figName)

print(stat.getDF('NM', np.array([32]), nmparams))
print(stat.inverseDist('NM', np.array([0.4]), nmparams))



# colNames = ['Probabilities' , '$y_p$', 'Normal Hypothesized', 'Normal Distance', 'Log-Ratio Normal Hypothesized', 'Log-Ratio Normal Distance']

# normalEstimates = np.flip(stat.inverseDist('NM', probs, nmparams))
# logNormalEstimates = np.flip(stat.inverseDist('LRNM', probs, lrparams))
# normalEstimates = stat.getDF('NM', yp, nmparams)
# logNormalEstimates = stat.getDF('LRNM', yp, lrparams)

# stat.printInLatexTable([probs, yp, normalEstimates, np.abs(normalEstimates - probs), logNormalEstimates, np.abs(logNormalEstimates - probs)], colNames)

# colNames = ['Probabilities', '$y_p$', '$x_p$', '$z_p$']
# stat.printInLatexTable([probs, yp, xp, zp], colNames)
# print(nmparams, lrparams)