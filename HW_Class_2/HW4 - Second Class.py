import Stats_Functions as stat
import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/ucg8nb/Downloads/New Data.csv", header = None)
newData = np.array(data[0].to_list())

lowBounds = []
for i in range(-500, -1050, -50):
    lowBounds.append(i)

outputDf = pd.DataFrame()
firstIteration = True

for l in lowBounds:
    tempDf = stat.findUDFit(newData, l, 2100, numSteps = 150)
    if firstIteration:
        outputDf = tempDf
        firstIteration = False
    else:
        outputDf = pd.concat([outputDf, tempDf], ignore_index = True)
    print(f"Finished UDFit for lower bound:{l}")

outputDf.to_csv(stat.getDownloadsTab() + '/All Lower Bounds.csv')