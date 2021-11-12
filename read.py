"""
Imports needed for the code.
"""
import numpy as np
import pandas as pd
from itertools import chain
from astroquery.gaia import Gaia
from astropy.io import ascii
import wget
import requests
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats
import sklearn.metrics as sm
defaults = [0] * 3
data = []
def reject_outliers(data):
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered
def isNaN(num):
    return num != num
def HMS2deg(ra='', dec=''):
  RA, DEC, rs, ds = '', '', 1, 1
  if ra:
    H, M, S, *_ = [float(i) for i in chain(ra.split(), defaults)]
    if str(H)[0] == '-':
      rs, H = -1, abs(H)
    deg = (H*15) + (M/4)
    RA = '{0}'.format(deg*rs)

  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC
def HMS2degDEC(dec='', ra=''):
     RA, DEC, rs, ds = '', '', 1, 1
     if dec:
       D, M, S, *_ = [float(i) for i in chain(dec.split(), defaults)]
       S = S[0] if S else 0
       if str(D)[0] == '-':
         ds, D = -1, abs(D)
       deg = D + (M/60) + (S/3600)
       DEC = '{0}'.format(deg*ds)
     if ra and dec:
       return (RA, DEC)
     else:
       return RA or DEC
count=1
csv_file='test1.csv'
data = pd.read_csv(csv_file, error_bad_lines=False)
radata=data['R.A.']
decdata=data['Dec.']
agedata=data['Age(Myr)']
diamaterdata=data['Diameter']
ra=[]
dec=[]
age=[]
csv_files=['M42.csv', 'Horsehead.csv', 'M93.csv', 'IrisTrain.csv']
ages=[3, 6, 25, 0.055]
diameter=[]
gooddata=[]
for i in range(len(radata)):
    if(isNaN(radata[i])):
        ra.append(0)
    else:
        ra.append(HMS2deg(radata[i]))
print(ra)
for i in range(len(decdata)):
    if(isNaN(decdata[i])):
        dec.append(0)
    else:
        dec.append(HMS2degDEC(decdata[i]))
print(dec)
for i in range(len(diamaterdata)):
    if(isNaN(diamaterdata[i])):
        diameter.append(0)
    else:
        diameter.append(((diamaterdata[i])/3600)*100)
print(diameter)
for i in range(len(ra)):
    query1="""    SELECT bp_rp, parallax, pmra, pmdec, phot_g_mean_mag AS gp
    FROM gaiadr2.gaia_source
    WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
    """
    query1=query1+"                   CIRCLE('ICRS'," +str(ra[i])+","+ str(dec[i])+","+str(diameter[i])+")"+")"
    string2="""
    AND phot_g_mean_flux_over_error > 50
    AND phot_rp_mean_flux_over_error > 20
    AND phot_bp_mean_flux_over_error > 20
    AND visibility_periods_used > 8
    """
    print(query1)
    query1=query1+string2
    try:
        job = Gaia.launch_job(query1)
        print(job)
        results = job.get_results()
        ascii.write(results, 'values'+str(count)+'.csv', format='csv', fast_writer=False)
        csv_files.append('values'+str(count)+'.csv')
        ages.append(agedata[i])
        print(ages)
        count+=1
    except:
        continue
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

"""
def objective(x, a, b, c):
	return a * x + b
needed for scipy modeling, polyfit was more accurate
"""
"""
Line 59-68 checks if CSV data is NAN if it is it will ignore the value and only take the data that can be used
"""
for i in range(len(csv_files)):
    data=pd.read_csv(csv_files[i])
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
ages3=[]
MAD2=[]
ages2 = [4000 if math.isnan(i) else i for i in ages]
for i in range(len(ages2)):
    ages3.append(float(ages2[i]))
print(ages3)
print(MAD)
MAD=[1.5 if math.isnan(i) else i for i in MAD]
for i in range(len(MAD)):
    MAD2.append(float(MAD[i]))
fig = plt.figure()
ax1 = fig.add_subplot('111')
ax1.scatter(ages3, MAD2, color='blue')
plt.ylim(-5,5)
polyline = np.linspace(-5, 9000, 20)
mod1 = np.poly1d(np.polyfit(ages3, MAD2, 2))#Train for a function of degree 3
ax1.plot(polyline,mod1(polyline), color='red')
predict = np.poly1d(mod1)
print(mod1)
plt.show()
