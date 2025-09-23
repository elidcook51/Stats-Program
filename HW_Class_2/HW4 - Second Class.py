import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ucg8nb/Downloads/New Data.csv", header = None)
newData = np.array(data[0].to_list())
plottingPositions = stat.metaGaussianPlottingPositions(len(newData))

lowBounds = []
for i in range(-500, -1050, -50):
    lowBounds.append(i)

outputDf = pd.DataFrame()
firstIteration = True

# print(stat.findUDFit(newData, -1500, 2100, numSteps=150))

# for l in lowBounds:
#     tempDf = stat.findUDFit(newData, l, 2100, numSteps = 150)
#     if firstIteration:
#         outputDf = tempDf
#         firstIteration = False
#     else:
#         outputDf = pd.concat([outputDf, tempDf], ignore_index = True)
#     print(f"Finished UDFit for lower bound:{l}")

# outputDf.to_csv(stat.getDownloadsTab() + '/All Lower Bounds.csv')

readDf = pd.read_csv(stat.getDownloadsTab() + '/All Lower Bounds.csv')
# keptDistributions = ['IW', 'LL']
# keptDf = readDf[readDf['Dist'].isin(keptDistributions)]
# keptDf.to_csv(stat.getDownloadsTab() + '/Kept Lower Bounds.csv')

# IWDf = readDf[readDf['Dist'] == 'IW']
# LLDf = readDf[readDf['Dist'] == 'LL']
# lowBound = IWDf['etaL'].tolist()
# IWMAD = IWDf['MAD'].tolist()
# LLMAD = LLDf['MAD'].tolist()
# plt.scatter(x = lowBound, y = IWMAD, s = 35, color = 'grey', label = 'IW')
# plt.scatter(x = lowBound, y = LLMAD, s = 35, color = 'lightgrey', label = 'LL')
# plt.title('Lower bound vs. MAD for IW and LL distributions')
# plt.xlabel("Lower Bound")
# plt.ylabel("MAD found from UD method")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Lower bound vs MAD.png')
# plt.show()
# plt.clf()

# alphaIW = 885.1790
# betaIW = 6.2957

# alphaLL = 978.5899
# betaLL = 8.7731

# eta = -1000

# nums = np.arange(-1000, 2100, 1)
# IW = stat.getDF('IW', nums, [alphaIW, betaIW, eta])
# LL = stat.getDF('LL', nums, [alphaLL, betaLL, eta])

# plt.scatter(x = newData, y = plottingPositions, s = 10, color = 'gray', label = 'Plotting Positions')
# plt.scatter(x = nums, y = IW, label = 'IW', color = 'lightgray', s = 1)
# plt.scatter(x = nums, y = LL, label = "LL", color = 'darkgray', s=1)
# plt.title('LL and IW Distributiosn with eta = -1000')
# plt.xlabel("Stock Return")
# plt.ylabel("Cumulative Probabilty")
# plt.legend()
# plt.savefig(stat.getDownloadsTab() + '/Eta -1000.png')


a0 = 2.1
a1 = 2.6
b = 3.0
e = 4

nums = np.arange(4.01, 15, 0.001)
df0 = stat.getdf('IW', nums, [a0, b, e])
df1 = stat.getdf('IW', nums, [a1, b, e])

# plt.scatter(x = nums, y = df0, color = 'gray', label = 'f0(x)', s= 10)
# plt.scatter(x = nums, y = df1, color = 'lightgray', label = 'f1(x)', s =10)
# plt.legend()
# plt.title('Conditional Density Function')
# plt.savefig(stat.getDownloadsTab() + '/Conditional Density Functions.png')
# plt.clf()

# likelihoodRatio = df0 / df1
# plt.scatter(x = nums, y = likelihoodRatio, color = 'gray', s = 10)
# plt.title("Likelihood Ratio Function")
# plt.savefig(stat.getDownloadsTab() + '/Likelihood Ratio Function.png')

DF0 = stat.getDF('IW', nums, [a0, b, e])
DF1 = stat.getDF("IW", nums, [a1, b, e])

detection = 1 - DF1
falseAlarm = 1 - DF0

# plt.scatter(x = falseAlarm, y = detection, color = 'gray', s = 5)
# plt.title('ROC for given conditions')
# plt.xlabel("Probability of False Alarm")
# plt.ylabel("Probability of Detection")
# plt.savefig(stat.getDownloadsTab() + '/ROC.png')

gList = [0.3, 0.5, 0.7]
colors = ['gray','darkgray', 'lightgray']
for i in range(len(gList)):
    g = gList[i]
    pix = stat.pix(g, 1, df0, df1)
    plt.scatter(x = nums, y = pix, color = colors[i], label = f"Prior Probabilty {g}", s = 5)

plt.title('Posterior Probability')
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/PIX for prior probability.png')