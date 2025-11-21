import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import os
import shutil

# picklepath = "C:/Users/ucg8nb/Downloads/GMM_results.pkl"

# with open(picklepath, 'rb') as f:
#     results = pickle.load(f)

# bigProbList = []
# bigTruthList = []

# for r in results:
#     truths = r['truth']
#     probs = r['probs']
#     for i in range(len(truths)):
#         bigProbList.append(probs[i])
#         bigTruthList.append(truths[i])

# outputDf = pd.DataFrame()
# outputDf['Truth'] = truths
# outputDf['Probs'] = probs
# outputDf.to_csv("C:/Users/ucg8nb/Downloads/GMMTruthProbsData.csv")

# truthProbsDf = pd.read_csv("C:/Users/ucg8nb/Downloads/GMMTruthProbsData.csv")

# poisoned = truthProbsDf[truthProbsDf['Truth'] == 1]
# notPoisoned = truthProbsDf[truthProbsDf['Truth'] == 0]

# poisonedList = poisoned['Probs'].tolist()
# notPoisonedList = notPoisoned['Probs'].tolist()

# plotPosPoisoned = stat.metaGaussianPlottingPositions(len(poisonedList))
# plotPosNotPoisoned = stat.metaGaussianPlottingPositions(len(notPoisonedList))

# notPoisonedFitParams = [2.425549, 1.189291, -0.0001, 1.0001]
# poisonedFitParams = [-0.01451, 0.856153, -0.0001]

# nums = np.arange(0.0001, 0.9999, 0.0001)
# poisonedDF = stat.getDF('IW', nums, poisonedFitParams)
# notPoisonedDF = stat.getDF('LRRG', nums, notPoisonedFitParams)

# plt.scatter(x = sorted(poisonedList), y = plotPosPoisoned, color = 'gray', label = 'poisoned')
# plt.scatter(x = sorted(notPoisonedList), y = plotPosNotPoisoned, color = 'lightgray', label = 'not poisoned')
# plt.scatter(x = nums, y = poisonedDF, color = 'blue', s = 5, label = 'Poisoned DF')
# plt.scatter(x = nums, y = notPoisonedDF, color = 'red', s = 5, label = 'Not Poisoned DF')
# plt.legend()
# plt.title("Distribution of poisoned and unpoisoned GMM probabilities")
# plt.ylabel('Cumulative Probability')
# plt.xlabel("GMM Poisoning output")
# plt.show()

# stat.findUDFit(poisonedList, -0.0001, 1.0001, 5000).to_csv("C:/Users/ucg8nb/Downloads/Poisoned UD Fitter.csv")
# stat.findUDFit(notPoisonedList, -0.0001, 1.0001, 5000).to_csv("C:/Users/ucg8nb/Downloads/Poisoned UD Fitter.csv")

def folderReader(folderPath, outputFolderPath):
    if os.path.exists(outputFolderPath):
        shutil.rmtree(outputFolderPath)
    os.makedirs(outputFolderPath)
    for filename in os.listdir(folderPath):
        filepath = os.path.join(folderPath, filename)
        outputFilepath = os.path.join(outputFolderPath, filename)
        readToOutputCSV(filepath, outputFilepath)
        print(f'Completed {filepath}')


def readToOutputCSV(csvPath,outputPath):


    newDf = pd.read_csv(csvPath)


    cleanedLabels = []
    for l in newDf['Cleaned Labels'].tolist():
        cleanedLabels.extend(l)
    cleanedLabels = [int(x) for x in cleanedLabels if x.isdigit()]
    poisonedLabels = []
    for l in newDf['Poisoned Labels'].tolist():
        poisonedLabels.extend(l)
    poisonedLabels = [int(x) for x in poisonedLabels if x.isdigit()]
    probPoisoned = []
    for l in newDf['Probability Vector'].tolist():
        probPoisoned.extend(l)
    joined = ''.join(probPoisoned)
    probPoisoned = [float(x) for x in re.findall(r'\d+\.\d+', joined)]

    wvector = [1 if clean != poisoned else 0 for (clean, poisoned) in zip(cleanedLabels, poisonedLabels)]
    outputDf = pd.DataFrame()
    outputDf['W'] = wvector
    outputDf['X'] = probPoisoned
    outputDf.to_csv(outputPath)

GMMresultFolder = "C:/Users/ucg8nb/Downloads/GMM Results"
GMMOutputFolder = "C:/Users/ucg8nb/Downloads/GMM Results Output"
folderReader(GMMresultFolder, GMMOutputFolder)

# firstTest = pd.read_csv("C:/Users/ucg8nb/Downloads/First Test GMM.csv")
# poisSet = firstTest[firstTest['W'] == 1]
# controlSet = firstTest[firstTest['W'] == 0]

# X1 = poisSet['X'].tolist()
# X0 = controlSet['X'].tolist()

# plotPosX1 = stat.metaGaussianPlottingPositions(len(X1))
# plotPosX0 = stat.metaGaussianPlottingPositions(len(X0))

# plt.scatter(x = sorted(X1), y = plotPosX1, s= 5, label = 'Poisoned')
# plt.scatter(x = sorted(X0), y = plotPosX0, s= 5, label = 'Not Poisoned')
# plt.legend()
# plt.savefig("C:/Users/ucg8nb/Downloads/Quick Display.png")




