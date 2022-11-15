#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:13:52 2022

@author: yaarsafra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Al_thick,Al_nrate,Al_ln_nrate,Al_sig_nrate,Cu_thickness,CountRate2,LogCount2,UncCount2 = np.loadtxt("/Users/yaarsafra/Desktop/Aluminium_data_desk.csv",unpack=True,delimiter=",",skiprows=1)

Cu_thick = Cu_thickness[0:18]
Cu_nrate = CountRate2[0:18]
Cu_ln_nrate = LogCount2[0:18]
Cu_sig_nrate = UncCount2[0:18]
X_values = np.linspace(0,1,num=1000)

def Al_exp(x,α,x1,c):# (x,α,x1,A,c):
    return 2.718281828459045**(α*x-x1)+c
def Cu_exp(x,α,x1,c):
    return 2.718281828459045**(α*x-x1)+c

def Al_ln(x,α,x1,c):
    return α*(x-x1)+c
def Cu_ln(x,α,x1,c):
    return α*(x-x1)+c

def linear(x,m,c):
    return m*x+c

curvefitAl, cov_curvefitAl = curve_fit(Al_exp, Al_thick, Al_nrate, sigma=np.sqrt(Al_nrate*30)/30, absolute_sigma=True)#p0=[105000,50,200])

plt.xlabel("Thickness (cm)")
plt.ylabel("Count rate (s^-1)")
plt.xticks()
plt.yticks()
plt.title('Aluminium Exponential Distribution')
plt.plot(X_values, Al_exp(X_values, *curvefitAl))
plt.grid()
plt.errorbar(Al_thick, Al_nrate, ls='',yerr = Al_sig_nrate,  mew=1, ms=3, capsize=2, color = 'red')
plt.savefig('Aluminium Exponential Distribution', dpi = 200)
plt.show()

curvefitCu, cov_curvefitCu = curve_fit(Cu_exp, Cu_thick, Cu_nrate, sigma=np.sqrt(Cu_nrate*30)/30, absolute_sigma=True)#p0=[105000,50,200])

plt.xlabel("Thickness (cm)")
plt.ylabel("Count rate (s^-1)")
plt.xticks()
plt.yticks()
plt.title('Copper Exponential Distribution')
plt.plot(X_values/2, Cu_exp(X_values/2, *curvefitCu))
plt.grid()
plt.errorbar(Cu_thick, Cu_nrate, ls='',yerr = np.sqrt(Cu_nrate*30)/30,  mew=1, ms=3, capsize=2, color = 'red')
plt.savefig('Copper Exponential Distribution', dpi = 200)
plt.show()


#%%

curvefitAl_ln, cov_curvefitAl_ln = curve_fit(Al_ln, Al_thick, Al_ln_nrate, sigma = 1/np.sqrt(Al_nrate), absolute_sigma=True)#p0=[105000,50,200])

α,x1,c = curvefitAl[0],curvefitAl[1],curvefitAl[2] # set values of curvefit  α,x1,A,c = curvefitAl[0],curvefitAl[1],curvefitAl[2],curvefitAl[3]
sig_α,sig_x1,sig_c = np.sqrt(cov_curvefitAl[0,0]),np.sqrt(cov_curvefitAl[1,1]),np.sqrt(cov_curvefitAl[2,2]) #set values of covarience matrix  sig_α,sig_x1,sig_A,sig_c = cov_curvefitAl[0,0],cov_curvefitAl[1,1],cov_curvefitAl[2,2],cov_curvefitAl[3,3]

Al_sig_ln_nrate = np.sqrt(((Al_thick-x1)*sig_α)**2+(α*0)**2+(α*sig_x1)**2) #proppagation formula for error in the logarith of the count rate, made error in thickness = 0

plt.xlabel("Thickness (cm)")
plt.ylabel("Natural Logarithm of Count rate ")
plt.xticks()
plt.yticks()
plt.title('Aluminium Logarithmic Distribution')
#plt.plot(X_values, Al_ln(X_values/2, *curvefitAl_ln))
plt.plot(X_values,np.zeros(1000), '--', color = 'black') #x axis
plt.plot(np.zeros(1000),(X_values-0.1)*12, '--', color = 'black') #y axis

#linear fits of different sections of data

linearfitAl_1, cov_linearfitAl_1 = curve_fit(linear, Al_thick[0:7], Al_ln_nrate[0:7])
linearfitAl_2, cov_linearfitAl_2 = curve_fit(linear, Al_thick[8:15], Al_ln_nrate[8:15])
linearfitAl_3, cov_linearfitAl_3 = curve_fit(linear, Al_thick[16:21], Al_ln_nrate[16:21])


plt.plot(X_values/1.5, linear(X_values/1.5, *linearfitAl_1)) #low energy particles
plt.plot(X_values/2, linear(X_values/2, *linearfitAl_2)) #higher energy particles
plt.plot(X_values, linear(X_values, *linearfitAl_3)) # background radiation

#solve for intersection of lines
    #first intersection
Al_x1 = (linearfitAl_2[1]-linearfitAl_1[1])/(linearfitAl_1[0]-linearfitAl_2[0])
y1 = Al_x1*linearfitAl_1[0]+linearfitAl_1[1]

Al_x2 = (linearfitAl_3[1]-linearfitAl_2[1])/(linearfitAl_2[0]-linearfitAl_3[0])
y2 = Al_x2*linearfitAl_2[0]+linearfitAl_2[1]

x3 = (linearfitAl_1[1]-linearfitAl_3[1])/(linearfitAl_3[0]-linearfitAl_1[0])
y3 = x3*linearfitAl_1[0]+linearfitAl_1[1]

m1,m2,m3,c1,c2,c3 = linearfitAl_1[0],linearfitAl_2[0],linearfitAl_3[0],linearfitAl_1[1],linearfitAl_2[1],linearfitAl_3[1]
sig_m1,sig_m2,sig_m3,sig_c1,sig_c2,sig_c3 = np.sqrt(cov_linearfitAl_1[0,0]),np.sqrt(cov_linearfitAl_2[0,0]),np.sqrt(cov_linearfitAl_3[0,0]),np.sqrt(cov_linearfitAl_1[1,1]),np.sqrt(cov_linearfitAl_2[1,1]),np.sqrt(cov_linearfitAl_3[1,1])

Al_sig_x1 = np.sqrt((sig_m1/(c2-c1))**2+(sig_m2/(c2-c1))**2+(2*(m1-m2)*sig_c2/(c2-c1))**2+(2*(m1-m2)*sig_c1/(c2-c1))**2)


Al_sig_x2 = np.sqrt((sig_m3/(c2-c3))**2+(sig_m2/(c2-c3))**2+(2*(m3-m2)*sig_c2/(c2-c3))**2+(2*(m3-m2)*sig_c3/(c2-c3))**2)



plt.grid()
plt.errorbar(Al_thick, Al_ln_nrate, ls='',yerr = Al_sig_ln_nrate/2,  mew=1, ms=3, capsize=2) #CHANGE ERROR FOR LOGARITHMIC Y VALUES
plt.savefig('Aluminium Logarithmic Distribution', dpi = 200)
plt.show()

print('--------ALUMINIUM LOGARITHMIC DATA ANALYSIS--------')
print('first intersection: (%.5f, %.5f)' % (Al_x1, y1))
print('second intersection: (%.5f, %.5f)' % (Al_x2, y2))
print('third intersection: (%.5f, %.5f)' % (x3, y3))
print('low energy radiation LoBF: %.5fx + %.5f' % (linearfitAl_1[0],linearfitAl_1[1]))
print('higher energy radiation LoBF: %.5fx + %.5f' % (linearfitAl_2[0],linearfitAl_2[1]))
print('background radiation LoBF: %.5fx + %.5f' % (linearfitAl_3[0],linearfitAl_3[1]))
print('---------')
print('Range in of low energy beta- particles = %.5f +/- %.5f cm' % (Al_x1, Al_sig_x1))
print('Range in of high energy beta- particles = %.5f +/- %.5f cm' % (Al_x2, Al_sig_x2))
print('---------')
print('')




#-----------------------------------------------------------------------


curvefitCu_ln, cov_curvefitCu_ln = curve_fit(Cu_ln, Cu_thick, Cu_ln_nrate, sigma = 1/np.sqrt(Cu_nrate), absolute_sigma=True)#p0=[105000,50,200])

α,x1,c = curvefitCu[0],curvefitCu[1],curvefitCu[2] # set values of curvefit to make propogation formula easier

sig_α,sig_x1,sig_c = np.sqrt(cov_curvefitCu[0,0]),np.sqrt(cov_curvefitCu[1,1]),np.sqrt(cov_curvefitCu[2,2]) #set values of covarience matrix to make propogation formula easier
Cu_sig_ln_nrate = np.sqrt(((Cu_thick-x1)*sig_α)**2+(α*0)**2+(α*sig_x1)**2) #proppagation formula for error in the logarithm of the count rate, made error in thickness = 0

plt.xlabel("Thickness (cm)")
plt.ylabel("Natural Logarithm of Count rate")
plt.xticks()
plt.yticks()
plt.title('Copper Logarithmic Distribution')
plt.plot(X_values/2,np.zeros(1000), '--', color = 'black') #x axis
plt.plot(np.zeros(1000),(X_values-0.1)*12, '--', color = 'black') #y axis


#linear fits of different sections of data !!NEED TO DETERMINE DIFFERENT SECTIONS!!

linearfitCu_1, cov_linearfitCu_1 = curve_fit(linear, Cu_thick[0:6], Cu_ln_nrate[0:6])
linearfitCu_2, cov_linearfitCu_2 = curve_fit(linear, Cu_thick[6:11], Cu_ln_nrate[6:11])
linearfitCu_3, cov_linearfitCu_3 = curve_fit(linear, Cu_thick[11:18], Cu_ln_nrate[11:18])


plt.plot(X_values/6, linear(X_values/6, *linearfitCu_1)) #low energy particles
plt.plot(X_values/7, linear(X_values/7, *linearfitCu_2)) #higher energy particles
plt.plot(X_values/2, linear(X_values/2, *linearfitCu_3)) # background radiation

#solve for intersection of lines
    #first intersection
Cu_x1 = (linearfitCu_2[1]-linearfitCu_1[1])/(linearfitCu_1[0]-linearfitCu_2[0])
y1 = Cu_x1*linearfitCu_1[0]+linearfitCu_1[1]
    #second
Cu_x2 = (linearfitCu_2[1]-linearfitCu_3[1])/(linearfitCu_3[0]-linearfitCu_2[0])
y2 = Cu_x2*linearfitCu_2[0]+linearfitCu_2[1]
    #third
x3 = (linearfitCu_3[1]-linearfitCu_1[1])/(linearfitCu_1[0]-linearfitCu_3[0])
y3 = x3*linearfitCu_3[0]+linearfitCu_3[1]

m1,m2,m3,c1,c2,c3 = linearfitCu_1[0],linearfitCu_2[0],linearfitCu_3[0],linearfitCu_1[1],linearfitCu_2[1],linearfitCu_3[1]
sig_m1,sig_m2,sig_m3,sig_c1,sig_c2,sig_c3 = np.sqrt(cov_linearfitCu_1[0,0]),np.sqrt(cov_linearfitCu_2[0,0]),np.sqrt(cov_linearfitCu_3[0,0]),np.sqrt(cov_linearfitCu_1[1,1]),np.sqrt(cov_linearfitCu_2[1,1]),np.sqrt(cov_linearfitCu_3[1,1])

Cu_sig_x1 = np.sqrt((sig_m1/(c2-c1))**2+(sig_m2/(c2-c1))**2+(2*(m1-m2)*sig_c2/(c2-c1))**2+(2*(m1-m2)*sig_c1/(c2-c1))**2)


Cu_sig_x2 = np.sqrt((sig_m3/(c2-c3))**2+(sig_m2/(c2-c3))**2+(2*(m3-m2)*sig_c2/(c2-c3))**2+(2*(m3-m2)*sig_c3/(c2-c3))**2)




plt.grid()
plt.errorbar(Cu_thick, Cu_ln_nrate, ls='',yerr = Cu_sig_ln_nrate/10,  mew=1, ms=3, capsize=2) #CHANGE ERROR FOR LOGARITHMIC Y VALUES
plt.savefig('Copper Logarithmic Distribution', dpi = 200)
plt.show()

print('--------COPPER LOGARITHMIC DATA ANALYSIS--------')
print('first intersection: (%.5f, %.5f)' % (Cu_x1, y1))
print('second intersection: (%.5f, %.5f)' % (Cu_x2, y2))
print('third intersection: (%.5f, %.5f)' % (x3, y3))
print('low energy radiation LoBF: %.5fx + %.5f' % (linearfitCu_1[0],linearfitCu_1[1]))
print('higher energy radiation LoBF: %.5fx + %.5f' % (linearfitCu_2[0],linearfitCu_2[1]))
print('background radiation LoBF: %.5fx + %.5f' % (linearfitCu_3[0],linearfitCu_3[1]))
print('---------')
print('Range in of low energy beta- particles = %.5f +/- %.5f cm' % (Cu_x1, Cu_sig_x1))
print('Range in of high energy beta- particles = %.5f +/- %.5f cm' % (Cu_x2, Cu_sig_x2))
print('---------')
