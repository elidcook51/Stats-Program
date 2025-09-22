import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

newData = pd.read_csv("C:/Users/ucg8nb/Downloads/New Data.csv", header = None)
newData = newData[0].tolist()

plottingPositions = stat.metaGaussianPlottingPositions(len(newData))
distDf = pd.read_csv("C:/Users/ucg8nb/Downloads/OutputDf Updated MAD.csv")
nums = np.arange(-500, 2100, 0.05)

# wbParams = [551.1068, 2.5396, -500]
# wbTitle = 'Weibull Distributuion Function vs. Empirical Distribution'
# wbXlabel = 'Stock Return'
# wbYlabel = 'Cumulative Probability'
# wbSaveFig = stat.getDownloadsTab() + '/Weibull Distribution Function.png'
# stat.plotDistributionFromDf(distDf, 'WB', nums, data = newData, title = wbTitle, xlabel = wbXlabel, ylabel = wbYlabel, savefig = wbSaveFig)
# iwTitle = 'Inverse Weibull Distribution Function vs. Empirical Distribution'
# iwXlabel = 'Stock Return'
# iwYlabel = 'Cumulative Probability'
# iwSaveFig = stat.getDownloadsTab() + '/Inverse Weibull Distribution Function.png'
# stat.plotDistributionFromDf(distDf, 'IW', nums, data = newData, title = iwTitle, xlabel = iwXlabel, ylabel = iwYlabel, savefig = iwSaveFig)

wbParams = stat.getParamsFromDf(distDf, 'WB')
iwParams = stat.getParamsFromDf(distDf, 'IW')
llParams = stat.getParamsFromDf(distDf, 'LL')
wbDF = stat.getDF('WB', nums, wbParams)
iwDF = stat.getDF('IW', nums, iwParams)
llDF = stat.getDF('LL', nums, llParams)

plt.scatter(x = nums, y = wbDF, color = 'darkgray', label = 'WB Distribution Function', s =3 )
plt.scatter(x = nums, y = iwDF, color = 'gray', label = 'IW Distribution Function', s=3 )
plt.scatter(x = nums, y = llDF, color = 'dimgray', label = 'LL Distribution Function',s=3 )
plt.scatter(x = newData, y = stat.metaGaussianPlottingPositions(len(newData)), color = 'lightgray',s = 3, label = 'Empirical Distribution')
plt.title("Distribution Functions vs. Empirical Distribution")
plt.xlabel("Stock Return")
plt.ylabel('Cumulative Distribution')
plt.legend()
plt.savefig(stat.getDownloadsTab() + '/All Three Compare.png')