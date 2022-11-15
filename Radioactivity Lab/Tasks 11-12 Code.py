# Author: Hiroki Kozuki

#%%
# Task 11:
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson



Count_per_cycle_11 = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\session 3\task 11 200 cycle 1s sample time poisson.txt", unpack = True, delimiter = ',')

# the histogram of the data
n11, bin_edges11, patches11 = plt.hist(Count_per_cycle_11, bins=10, color='#0504aa', alpha=0.7)
# n = number of occurances in each cycle.
# calculate bin centres
bin_middles11 = 0.5 * (bin_edges11[1:] + bin_edges11[:-1])

plt.errorbar(bin_middles11, n11, yerr=np.sqrt(n11), mew=1, ms=3, capsize=3, color='black', ls='none') # Calculate error in histogram data, yerr is calculated by taking the square root, xerr=widthBin/2

#plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle) # Plot the mean values

a = 220
mu = 3.30
def poisson11_fit(x):
    return a*mu**x*np.exp(-mu)/sp.special.factorial(x)

# k = number of occurence. mu = average number of occurence. 

# fit with curve_fit
#params, cov_params = curve_fit(poisson_fit, bin_middles, n)
# plot poisson-deviation with fitted parameter
x = np.linspace(0, 10, 10000)

plt.plot(x, poisson11_fit(x), linestyle='-', color = "orange", label='Poisson')
plt.legend()
plt.xlabel("Counts per Cycle")                                            # Label axes, add titles and error bars
plt.ylabel("Number of Cycles")
plt.title("Task 11, 1s sample time, 200 cycles")
plt.show()
plt.savefig('Task 11 Histogram with a Poisson Fit.jpeg', dpi=1000)

#%%
# Task 12

Count_per_cycle_12 = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\session 3\task 12 200 cycle 1s sample time poisson-gaussian trial 2.txt", unpack = True, delimiter = ',')

# the histogram of the data
n12, bin_edges12, patches12 = plt.hist(Count_per_cycle_12, bins=20, color='#0504aa', alpha=0.7)
# n = number of occurances in each cycle.
# calculate bin centres
bin_middles12 = 0.5 * (bin_edges12[1:] + bin_edges12[:-1])

plt.errorbar(bin_middles12, n12, yerr=np.sqrt(n12), mew=1, ms=3, capsize=3, color='black', ls='none') # Calculate error in histogram data, yerr is calculated by taking the square root, xerr=widthBin/2

#plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle) # Plot the mean values

x2 = np.linspace(170, 240, 10000, endpoint = True)

b = 750
mu2 = 206.5
sigma = 13.45
def gaussian_fit(x):
    return b/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu2)/sigma)**2)

c = 430
mu3 = 36.5
def poisson12_fit(x):
    return c*mu3**(x-170)*np.exp(-mu3)/sp.special.factorial(x-170)
# k = number of occurence. mu = average number of occurence. 

# fit with curve_fit
#params, cov_params = curve_fit(poisson_fit, bin_middles, n)
# plot poisson-deviation with fitted parameter

plt.plot(x2, poisson12_fit(x2), linestyle='-', color = "magenta", label='Poisson')
plt.plot(x2, gaussian_fit(x2), linestyle='-', color = "red", label='Gaussian')
plt.legend()
plt.xlabel("Counts per Cycle")                                            # Label axes, add titles and error bars
plt.ylabel("Number of Cycles")
plt.title("Task 12, 1s sample time, 200 cycles")
plt.show()
plt.savefig('Task 12 Histogram with a Gaussian Fit.jpeg', dpi=1000)

#%%