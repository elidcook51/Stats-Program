import Stats_Functions as stat
import numpy as np
import pandas as pd
import os
import shutil

crosswalk = pd.read_csv("C:/Users/ucg8nb/Downloads/dateToCostCrosswalk.csv")
crosswalk['Date'] = pd.to_datetime(crosswalk['Date'])
crosswalk['Month'] = crosswalk['Date'].dt.month
totCosts = crosswalk['Cost'].tolist()

months = list(range(1,13))
outputFolderName = 'All Months Dists'
outputFolder = stat.getDownloadsTab() + outputFolderName

if os.path.exists(outputFolder):
    shutil.rmtree(outputFolder)
os.makedirs(outputFolder)

steps = 20000

for m in months:
    tempDf = crosswalk[crosswalk['Month'] == m]
    costs = tempDf['Cost'].tolist()
    fileName = f"Fitted Distributions for {m}.csv"
    filepath = os.path.join(outputFolder, fileName)
    stat.findUDFit(costs, 10, 17, numSteps = steps).to_csv(filepath)
    print(f'Fitted Distribution for month {m}')

totFilename = f"Total Fitted Distribution.csv"
totFilepath = os.path.join(outputFolder, totFilename)
stat.findUDFit(totCosts, 10, 17, numSteps = steps).to_csv(totFilepath)