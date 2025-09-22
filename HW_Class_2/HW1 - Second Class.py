import Stats_Functions as stat
import numpy as np

list = [2, 3, 1, 3, 2, 1, 1, 1, 2]
print(stat.sampleMean(list))
print(stat.sampleVariance(list, False))
print(stat.sampleVariance(list, True))
