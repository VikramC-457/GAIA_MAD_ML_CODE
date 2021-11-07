"""
Version 3 Gaiadr2 survey from Gaia ESA SQL code to get HR diagram data Code looks at MAD(Mean Absolute Deviation) over age
polyfit with a degree of 3 calculates 65% accurate model allowing for Astronomers to calculate Age using MAD and see GMC-
Giant Molecular Cloud to Open Star cluster over time.
SQL CODE:
SELECT bp_rp, parallax, pmra, pmdec, phot_g_mean_mag AS gp
FROM gaiadr2.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', RA, DEC, FOV))
AND phot_g_mean_flux_over_error > 50
AND phot_rp_mean_flux_over_error > 20
AND phot_bp_mean_flux_over_error > 20
AND phot_bp_rp_excess_factor < 1.3 + 0.06 * POW(bp_rp, 2)
AND phot_bp_rp_excess_factor > 1.0 + 0.015 * POW(bp_rp, 2)
AND visibility_periods_used > 8
Sources: https://gaia.aip.de/query/
Improvments: More data needed we were limited with the data and information that the Interent could provide
Copyright 2021, Vikramaditya Chandra, All rights reserved.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats
import sklearn.metrics as sm
csv_file=['M42.csv', 'BeeHive.csv', 'M45.csv', 'NGC2516.csv', 'Horsehead.csv', 'M23.csv', 'M25.csv', 'M26.csv', 'M36.csv', 'M93.csv', 'NGC2547.csv', 'M67.csv', 'IrisTrain.csv', 'NGC752.csv','M103.csv','NGC__6791.csv', 'ArpMadore.csv']
age=[3,600,175,110,6,250,67.6,85.3,25.1,387.3,25,4000, 0.055, 2000,16,1700,3000]
#1,
arr2=[]
datasetY=[]
datasetX=[]
Y=[]
av=0
count=[]
count2=[]
MAD=[]
"""
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1)
    return results
original accuracy calculation
"""
def isNaN(num):#isNaN Calculates if the CSV cell is not a number leading to a error, removes the possibility of the error.
    return num!= num
"""
def objective(x, a, b, c):
	return a * x + b
needed for scipy modeling, polyfit was more accurate
"""
"""
Line 59-68 checks if CSV data is NAN if it is it will ignore the value and only take the data that can be used
"""
for i in range(len(csv_file)):
    data=pd.read_csv(csv_file[i])
    arr=data['gp']
    arr2=data['bp_rp']
    for i in range(len(arr2)):
        if(isNaN(arr2[i])):
            continue
        else:
            datasetX.append(arr2[i])
            datasetY.append(arr[i])
    mad=stats.median_absolute_deviation(datasetY)#Calculate MAD for Magnitude
    mad2=stats.median_absolute_deviation(datasetX)#Calculate MAD for Color
    madav=(mad+mad2)/2#Total MAD
    MAD.append(madav)#Appending to an Array for training and plotting
    datasetX.clear()#Clearing for next Iteration
    datasetY.clear()#Clearing for next Iteration
"""
Plotting data and Traning
"""
fig = plt.figure()
ax1 = fig.add_subplot('111')
ax1.scatter(age, MAD)
plt.ylim(max(MAD),min(MAD))
polyline = np.linspace(-5, 4000, 20)
mod1 = np.poly1d(np.polyfit(age, MAD, 3))#Train for a function of degree 3
ax1.plot(polyline,mod1(polyline), color='green')
predict = np.poly1d(mod1)
print("R2 score =", round(sm.r2_score(MAD, predict(age)), 2))#Calculate Accuracy using r2score function
print(mod1)
plt.show()
"""
Ways to improve:
Access database and read to import into csv
Then take necessary inputs and use gaia API key to export data into different CSV
This will allow for more data to be gathered at a faster and more efficient rate.
"""
