#%%

# Task 17
# INVERSE SQUARE FIT for true count rate vs. distance 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import umath
from uncertainties import ufloat
from uncertainties import unumpy

font = {'fontname':'Times New Roman'}
fontAxesTicks = {'size':7}

d, n, t, unc_n = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=1)
# d_restricted, n_restricted, t_restricted, unc_n_restricted = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=9)

deadtime = 1.1e-06

true_count_rate = (n/t)/(1-(n/t)*deadtime)
unc_true_count_rate = np.sqrt(n)/t

# count_rate_restr = n_restricted/t_restricted
# unc_count_rate_restr = np.sqrt(n_restricted)/t_restricted

area,unc_area = 1.97e-4, 1.4e-6
def inverse_square(d,X,Y,Z):                                                                                 
     return X*area/(4*np.pi*(d+Y)**2) + Z

params_rate, cov_params_rate = curve_fit(inverse_square,d,true_count_rate,p0=[1400000,0.007,-100], maxfev=100000) # -0.575
x = np.linspace(0,0.85,10000)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("True Count Rate ($s^{-1}$)", **font)
plt.grid()
#plt.ylim(5,25)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
plt.ylim(-0.2e04, 0.05e06)
plt.title("True Count rate ($s^{-1}$) vs. Distance (m)", **font)
plt.plot(x, (params_rate[0]*area)/(4*np.pi*(x+params_rate[1])**2) + params_rate[2], color = 'orange', label='Inverse square fit') # 11.8 , ls='-'
#plt.plot(x, exponential_decay(x, *params),'r')    
plt.errorbar(d, true_count_rate, yerr=unc_true_count_rate, xerr=0.003, ls='', mew=1, ms=0.5, capsize=3, color = 'blue', label='Measured Data') # Plots uncertainties in points          
plt.legend()
plt.savefig('Task 17 Estimating Activity.jpeg', dpi=1000)
plt.show()      


print(params_rate)

print(np.sqrt(cov_params_rate[0][0]), np.sqrt(cov_params_rate[1][1]), np.sqrt(cov_params_rate[2][2]))

u_area = ufloat(1.97e-4, 1.4e-5)
X = ufloat(params_rate[0],np.sqrt(cov_params_rate[0,0])) # change vairable name
Y = ufloat(params_rate[1],np.sqrt(cov_params_rate[1,1]))
Z = ufloat(params_rate[2],np.sqrt(cov_params_rate[2,2]))

print("Activity X", X)
print("Y", Y)
print("Z", Z)





#%%

# Task 17
# EXPONENTIAL FIT for true u vs. distance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import umath
from uncertainties import ufloat
from uncertainties import unumpy

font = {'fontname':'Times New Roman'}
fontAxesTicks = {'size':9}

d, n, t, unc_n = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=1)
restr_d, restr_n, restr_t, unc_restr_n = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=4) 
# new_d, new_n, new_t, unc_new_n = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Radioactivity\Radioactivity Data\NEW count rate vs. distance.csv", delimiter=",", unpack=True, skiprows=4)

deadtime = 1.1e-06

# raw data
dist = unumpy.uarray(d, 0.001)
count_rate = unumpy.uarray(n/t, np.sqrt(n)/t)
true_count_rate_both = count_rate/(1-count_rate*deadtime)
true_count_rate = unumpy.nominal_values(true_count_rate_both)
unc_true_count_rate = unumpy.std_devs(true_count_rate_both)

u_both = true_count_rate*(dist**2)
u = unumpy.nominal_values(u_both)
unc_u = unumpy.std_devs(u_both)

# restricted data
restr_dist = unumpy.uarray(restr_d, 0.001)
restr_count_rate = unumpy.uarray(restr_n/restr_t, np.sqrt(restr_n)/restr_t)
restr_true_count_rate_both = restr_count_rate/(1-restr_count_rate*deadtime)
restr_true_count_rate = unumpy.nominal_values(restr_true_count_rate_both)
unc_restr_true_count_rate = unumpy.std_devs(restr_true_count_rate_both)

restr_u_both = restr_true_count_rate*(restr_dist**2)
restr_u = unumpy.nominal_values(restr_u_both)
unc_restr_u = unumpy.std_devs(restr_u_both)

# NEW data
# new_dist = unumpy.uarray(new_d, 0.001)
# new_count_rate = unumpy.uarray(new_n/new_t, np.sqrt(new_n)/new_t)
# new_true_count_rate_both = new_count_rate/(1-new_count_rate*deadtime)
# new_true_count_rate = unumpy.nominal_values(new_true_count_rate_both)
# unc_new_true_count_rate = unumpy.std_devs(new_true_count_rate_both)

# new_u_both = new_true_count_rate*(new_dist**2)
# new_u = unumpy.nominal_values(new_u_both)
# unc_new_u = unumpy.std_devs(new_u_both)



area,unc_area = 1.97e-4, 1.4e-6
def exponential_decay(d,A,mu,k,B):                                                                                 
     return A*area*np.exp(-mu*d+k)/(4*np.pi) + B


params, cov_params = curve_fit(exponential_decay,restr_d, restr_u,p0=[5550000,2.8,-2.45,11.2])
params_new, cov_params_new = curve_fit(exponential_decay,restr_d[:12], restr_u[:12],p0=[5550000,2.8,-2.45,11.2])
x = np.linspace(0,0.85,10000)
plt.title("Task 17: u ($m^{2}$ $s^{-1}$) vs. Distance (m)", **font)
plt.xlabel("Source-detector Separation d (m)", **font)                                            # Label axes, add titles and error bars
plt.ylabel("True u ($m^{2}$ $s^{-1}$)", **font)
plt.grid()
plt.ylim(11,20)
plt.xticks(**font, **fontAxesTicks)
plt.yticks(**font, **fontAxesTicks)
#plt.axhline(87, color='green', xmin=0, xmax=0.85, linestyle='--')
plt.plot(x, params[0]*area*np.exp(-params[1]*x+params[2])/(4*np.pi) + params[3], ls='-', color = 'red', label='With fluctuation') # 11.8
plt.plot(x, params_new[0]*area*np.exp(-params_new[1]*x+params_new[2])/(4*np.pi) + params_new[3], ls='-', color = 'green', label='Without fluctuation') 
#plt.plot(x, exponential_decay(x, *params),'r')    
plt.errorbar(d, u, yerr=unc_u, xerr=0.003, ls='', elinewidth=0.6, mew=0.6, ms=0.6, capsize=3, color = 'blue', label='Measured data') # Plots uncertainties in points
plt.legend()
plt.savefig('Final Task 17 Estimating Activity.png', format='png', dpi=1000)
plt.show()      

restr_u_fit = exponential_decay(restr_d, *params)
r = restr_u - restr_u_fit
chisq = np.sum((r/unc_restr_u)**2)
print('chi square = ', chisq)

new_u_fit = exponential_decay(restr_d[:12], *params_new)
r_new = restr_u[:12] - restr_u_fit[:12]
chisq_new = np.sum((r_new/unc_restr_u[:12])**2)
print('chi square_new = ', chisq_new)



print(params)
print(np.sqrt(cov_params[0][0]), np.sqrt(cov_params[1][1]), np.sqrt(cov_params[2][2]), np.sqrt(cov_params[3][3]))

u_area = ufloat(1.97e-4, 1.4e-5)
A = ufloat(params[0],np.sqrt(cov_params[0][0])) # change vairable name
mu = ufloat(params[1],np.sqrt(cov_params[1][1]))
k = ufloat(params[2],np.sqrt(cov_params[2][2]))
B = ufloat(params[3],np.sqrt(cov_params[3][3]))

#print("Activity A", A)
# print("mu", mu)
# print("k", k)
# print("B", B)

print(params_new)
print(np.sqrt(cov_params_new[0][0]), np.sqrt(cov_params_new[1][1]), np.sqrt(cov_params_new[2][2]), np.sqrt(cov_params_new[3][3]))

A_new = ufloat(params_new[0],np.sqrt(cov_params_new[0][0])) # change vairable name
mu_new = ufloat(params_new[1],np.sqrt(cov_params_new[1][1]))
k_new = ufloat(params_new[2],np.sqrt(cov_params_new[2][2]))
B_new = ufloat(params_new[3],np.sqrt(cov_params_new[3][3]))

#print("NEW Activity A", A_new)








# %%

# Uncertainties for tasks 19 & 20

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import umath
from uncertainties import ufloat
from uncertainties import unumpy

density_al = ufloat(2.70,0.03)
density_cu = ufloat(8.96,0.09)

d_al_low = ufloat(0.16,0.02)
d_al_high = ufloat(0.37,0.07)
d_cu_low = ufloat(0.038,0.010)
d_cu_high = ufloat(0.11,0.04)

R_al_low = density_al*d_al_low
R_al_high = density_al*d_al_high
R_cu_low = density_cu*d_cu_low
R_cu_high = density_cu*d_cu_high

# print(R_al_low)
# print(R_al_high)
# print(unumpy.nominal_values(R_cu_low), "+/-", unumpy.std_devs(R_cu_low))
# print(unumpy.nominal_values(R_cu_high), "+/-", unumpy.std_devs(R_cu_high))


E_al_low = unumpy.sqrt(((R_al_low/0.11+1)**2-1)/22.4)
E_al_high = unumpy.sqrt(((R_al_high/0.11+1)**2-1)/22.4)
E_cu_low = unumpy.sqrt(((R_cu_low/0.11+1)**2-1)/22.4)
E_cu_high = unumpy.sqrt(((R_cu_high/0.11+1)**2-1)/22.4) 

print(E_al_low)
print(E_al_high)
print(unumpy.nominal_values(E_cu_low), "+/-", unumpy.std_devs(E_cu_low))
print(unumpy.nominal_values(E_cu_high), "+/-", unumpy.std_devs(E_cu_high))

# %%
