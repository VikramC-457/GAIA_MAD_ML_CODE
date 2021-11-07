import pandas as pd
import matplotlib.pyplot as plt
csv_file='Iris.csv'
data=pd.read_csv(csv_file)
arr=data['bp_rp']
arr2=data['gp']
X=[]
Y=[]
for i in range(len(arr2)):
    if(13<=arr2[i]<=19):
        Y.append(arr2[i])
        X.append(arr[i])
fig = plt.figure()
plt.ylim(max(Y), min(Y))
ax=fig.add_subplot(111)
plt.scatter(x=X, y=Y, c=X, cmap='RdYlBu_r')
plt.colorbar(label="Star Color(B-V)", orientation="horizontal")
plt.ylabel('Magnitude')
lhs,rhs=csv_file.split(".",1)
plt.title(lhs)
plt.show()
