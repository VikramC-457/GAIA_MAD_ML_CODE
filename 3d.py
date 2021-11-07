import pandas as pd
import matplotlib.pyplot as plt
import math
data=pd.read_csv('Flame_updated.csv')
parallax=data['parallax']
pmra=data['pmra']
pmdec=data['pmdec']
plotpar=[]
plotra=[]
plotdec=[]
for i in range(len(pmra)):
    plotra.append(pmra[i])
for i in range(len(pmdec)):
    plotdec.append(pmdec[i])
for i in range(len(parallax)):
    try:
        d=1.01541148669/(math.tan(((parallax[i])*(1/1000))/2))
    except ZeroDivisionError:
        d=1
    plotpar.append(d)
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
print(plotra)
ax1.scatter(plotra,plotdec,plotpar)
plt.show()
