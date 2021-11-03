from scipy.stats import ttest_1samp
import numpy as np

stdid = ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
hieghts = ([5.1,5.2,5.4,5.5,5.5,5.6,5.6,5.7,5.7,5.8,5.8,5.9,6,6.1,6])
print(hieghts)

hiegt_mean = np.mean(hieghts)
print(hiegt_mean)

tset, pval = ttest_1samp(hieghts, 5.5)
print('p-values' ,pval)

if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")