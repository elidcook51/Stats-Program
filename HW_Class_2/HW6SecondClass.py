import Stats_Functions as stat
import numpy as np

nums = [0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.6, 0.5, 0.8]
nums = np.array(nums)
N = len(nums)

varx = np.sum((nums) * (1 - nums))
print(varx)
