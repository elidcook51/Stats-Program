import Stats_Functions as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import ast
import os
import shutil
from sklearn.metrics import roc_curve

df = pd.read_json("C:/Users/ucg8nb/Downloads/results.jsonl", lines = True)
seconddf = pd.read_json("C:/Users/ucg8nb/Downloads/results (1).jsonl", lines = True)
df = pd.concat([df, seconddf.apply(pd.Series)], ignore_index = True)

def parse_value(val_str):
    val_str = val_str.strip()

    try:
        return int(val_str)
    except ValueError:
        pass

    try:
        return float(val_str)
    except ValueError:
        pass

    try:
        return ast.literal_eval(val_str)
    except (ValueError, SyntaxError):
        return val_str

def parse_poison_string(s):
    s = s.strip()

    m = re.match(r"\s*(?P<method>[A-Za-z_][A-Za-z_0-9]*)\s*\(\s*(?P<body>.*)\s*\)\s*$", s)

    if not m:
        return {'method': None}
    
    method = m.group('method')
    body = m.group('body')

    parts = []
    buf = []
    depth = 0

    for ch in body:
        if ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        if ch == ',' and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))

    kv = {}
    for p in parts:
        if "=" in p:
            k, v= p.split('=', 1)
            k = k.strip()
            v = v.strip()

            if k == 'propr':
                k = 'prop'
            kv[k] = parse_value(v)

    record = {'poison method': method}
    record.update(kv)

    # record['seed'] = record.get('seed')
    # record['prop'] = record.get('prop')
    # record['old_label'] = record.get('old')
    # record['new_label'] = record.get('new')
    # record['labels'] = record.get('label')
    # record['labels_to_flip'] = record.get('labelsToFlip')

    return record  

def detectPoison(cleanLabels, poisonLabels):
    return (np.array(cleanLabels) != np.array(poisonLabels)).astype(int).tolist()

def flattenMatrix(probabilityVectorMatrix):
    return np.array(probabilityVectorMatrix).flatten().tolist()

parse = df['Poison Method'].apply(parse_poison_string)
df = pd.concat([df, parse.apply(pd.Series)], axis = 1)
# df = df.drop(columns = [c for c in df.columns if c.startswith('self.')])


extracted = df['Detection Strategy'].str.extract(
    r"^(?P<method>[A-Za-z_][A-Za-z_0-9]*)"           # method name (e.g., GMM_GPU, WassersteinDistance1D)
    r"(?:\s*\(\s*.*?threshold\s*=\s*(?P<threshold>[-+]?\d*\.?\d+)\s*.*\))?$",  # optional threshold
    expand=True
)

df[['detection method', 'threshold']] = extracted

df['isPoisoned'] = df.apply(lambda row: detectPoison(row['Cleaned Labels'], row['Poisoned Labels']), axis = 1)

detectionStrategies = ['GMM_GPU', 'WassersteinDistance1D']

poisonMethods = ['RandomPoison', 'SwitchLabel', 'OneToMany', 'ManyToOne', 'FlipNPoints']

proportionPosioned = [0, 0.02, 0.05, 0.15, 0.3]

gmm = df[df['detection method'] == 'GMM_GPU'] 

wasserstein = df[df['detection method'] == 'WassersteinDistance1D']
wasserstein['Probability Vector'] = wasserstein['Probability Vector'].apply(flattenMatrix)
wasserstein['threshold'] = pd.to_numeric(wasserstein['threshold'])

def roc_curve_save(isPoisoned, probVector, figPath, figName):
    fpr, tpr, thresh = roc_curve(isPoisoned, probVector)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], 'k--', label = 'No Skill')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(f'ROC Curve for {figName}')
    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    plt.savefig(figPath)
    plt.close()

# outputFolder = stat.getDownloadsTab() + 'GMM ROC Curves'
# if os.path.exists(outputFolder):
#     shutil.rmtree(outputFolder)

# os.makedirs(outputFolder)
# for pMeth in poisonMethods:
#     for prop in proportionPosioned:
#         figName = f"{pMeth} at {prop * 100}%"
#         figPath = os.path.join(outputFolder, f"{figName}.png")
#         tempDf = gmm[gmm['poison method'] == pMeth]
#         tempDf = tempDf[tempDf['self.prop'] == prop]
#         probVectorList = [item for sublist in tempDf['Probability Vector'].tolist() for item in sublist]
#         isPoisonedList = [item for sublist in tempDf['isPoisoned'].tolist() for item in sublist]
#         roc_curve_save(isPoisonedList, probVectorList, figPath, figName)

thresholds = [3,4,5]

wassersteinOutputFolder = stat.getDownloadsTab() + 'Wasserstein'
if os.path.exists(wassersteinOutputFolder):
    shutil.rmtree(wassersteinOutputFolder)
os.makedirs(wassersteinOutputFolder)
for t in thresholds:
    newDirs = os.path.join(wassersteinOutputFolder, f'Threshold of {t}')
    os.makedirs(newDirs)
    for pMeth in poisonMethods:
        for prop in proportionPosioned:
            figName = f"{pMeth} at {prop * 100}%"
            figPath = os.path.join(newDirs, f"{figName}.png")
            tempDf = wasserstein[wasserstein['poison method'] == pMeth]
            tempDf = tempDf[tempDf['self.prop'] == prop]
            tempDf = tempDf[tempDf['threshold'] == t]
            probVectorList = [item for sublist in tempDf['Probability Vector'].tolist() for item in sublist]
            isPoisonedList = [item for sublist in tempDf['isPoisoned'].tolist() for item in sublist]
            print(len(probVectorList))
            print(len(isPoisonedList))
            roc_curve_save(isPoisonedList, probVectorList, figPath, figName)