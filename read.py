"""
Imports needed for the code.
"""
"""
Code Written by: Vikramaditya Chandra 2021
Contributers: Demetrios Dresios
"""
"""
Script to get and clean data
"""
import numpy as np
import pandas as pd
from itertools import chain
from astroquery.gaia import Gaia
from pynverse import inversefunc
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
defaults = [0] * 3#needed for ignoring values that don't exsist
data = []#array for storing data
def reject_outliers(data):#Outlier Rejection Function
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered
def isNaN(num):#Checking if it is NaN(Not a Number)
    return num != num
def HMS2deg(ra='', dec=''):#Convert from form RA to Degree RA(Gaia Form)
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
def HMS2degDEC(dec='', ra=''):#Convert from form Dec to Degree Dec(Gaia Form)
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
csv_file='test1.csv'#Data Storing File for Gaia
data = pd.read_csv(csv_file, error_bad_lines=False)#Ignore the bad lines
radata=data['R.A.']#get RA
decdata=data['Dec.']#get dec
agedata=data['Age(Myr)']#get Age
diamaterdata=data['Diameter']#get Diameter later converted to FOV
ra=[]#cleaned RA
dec=[]#cleaned Dec
age=[]#Cleaned age
csv_files=['M42.csv', 'Horsehead.csv', 'M93.csv', 'IrisTrain.csv']#Pre exsisting data
ages=[3, 6, 25, 0.055]#pre exsisting data's age
diameter=[]#Diameter cleaned data
gooddata=[]#Overall data storage for cleaned data
for i in range(len(radata)):#cleaning RA data and converting
    if(isNaN(radata[i])):
        ra.append(0)
    else:
        ra.append(HMS2deg(radata[i]))
print(ra)
for i in range(len(decdata)):#Cleaning Dec Data and converting
    if(isNaN(decdata[i])):
        dec.append(0)
    else:
        dec.append(HMS2degDEC(decdata[i]))
print(dec)
for i in range(len(diamaterdata)):#cleaning diameter data and converting to FOV
    if(isNaN(diamaterdata[i])):
        diameter.append(0)
    else:
        diameter.append(((diamaterdata[i])/3600)*100)
print(diameter)
for i in range(len(ra)):#Modified Query for each object
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
    try:#Try the following code
        job = Gaia.launch_job(query1)#Launch query to gaia webpage
        print(job)
        results = job.get_results()#get results
        ascii.write(results, 'values'+str(count)+'.csv', format='csv', fast_writer=False)
        csv_files.append('values'+str(count)+'.csv')#store in CSV
        ages.append(agedata[i])#append data
        print(ages)
        count+=1#avoid re-writing CSV file by creating different ones
    except:#If the code throws any error, usually 'can't query' it will ignore the file, another filter to clean out any useless or bad data
        continue
"""
End of Cleaning and Gathering Data
"""
"""
Training and Creating Model with the data
"""
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
count=0
for i in range(len(csv_files)):
    data=pd.read_csv(csv_files[i])
    arr=data['gp']
    arr2=data['bp_rp']
    for i in range(len(arr2)):
        if(isNaN(arr2[i])):
            continue
        elif(13<=arr[i]<=19 and 0<arr2[i]<0.7):
            datasetX.append(arr2[i])
            datasetY.append(arr[i])
            count+=1
    mad=stats.median_absolute_deviation(datasetY)#Calculate MAD for Magnitude
    mad2=stats.median_absolute_deviation(datasetX)#Calculate MAD for Color
    madav=(mad+mad2)/2#Total MAD
    MAD.append(count)#Appending to an Array for training and plotting
    datasetX.clear()#Clearing for next Iteration
    datasetY.clear()#Clearing for next Iteration
    count=0
"""
Plotting data and Traning
"""
ages3=[]
MAD2=[]
ages2 = [0 if math.isnan(i) else i for i in ages]#ignore any age nan values
print(len(ages3))
print(len(MAD))
MAD=[1.5 if math.isnan(i) else i for i in MAD]#ignore any MAD computation error values
for i in range(len(MAD)):
    if(-500<=MAD[i]<=1500 and -25<=ages2[i]<170 or (100<=MAD[i]<=1262) and (278<=ages2[i]<=5067) or (-20<=MAD[i]<=20) and (3900<=ages2[i]<=4100) or (2642<=MAD[i]<=4750) and (0<=ages2[i]<=200) or (7800<=MAD[i]<=315800) and (0<=ages2[i]<=20)):#Manual Rejection
        continue
    else:
        ages3.append(float(ages2[i]))
        MAD2.append(float(MAD[i]))
fig = plt.figure()
ax1 = fig.add_subplot('111')
ax1.scatter(ages3, MAD2, color='blue')
plt.ylim(-7800,315800)
polyline = np.linspace(-5, 9000, 20)
mod1 = np.poly1d(np.polyfit(ages3, MAD2, 2))#Train for a function of degree 2(Loss of energy, expected degree polynomial) 
predict = np.poly1d(mod1)
ax1.plot(polyline,mod1(polyline), color='red')
print(np.interp(0.795, mod1(polyline),polyline))
print(mod1)#print model
plt.show()
"""
End of Training and Creating model/End of Script
"""
