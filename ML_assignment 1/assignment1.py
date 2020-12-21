import math
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import statistics as stat
from tabulate import tabulate
from numpy import linalg as la
from sklearn.neighbors import NearestNeighbors as kn
from sklearn.neighbors import KNeighborsClassifier as kNC


ns = pd.read_csv('NormalSample.csv')
print(ns)
saved_column = ns.x.values.copy() #you can also use df['column_name']
saved_column.sort()
#Q1-a
Q1 = np.percentile(saved_column,25)
Q2 = np.median(saved_column)
Q3 = np.percentile(saved_column,75)
IQR = Q3 - Q1
N = len(saved_column)
rec_h = 2*IQR*pow(N,(-1/3))
print("Recommended bandwidth",rec_h)
#Q1-b
print("Minimum value",saved_column[0])
print("Maximum value",saved_column[len(saved_column)-1])
#Q1-c
a=math.ceil(saved_column[0])-1
print("Largest value less than minimum",a)
b=math.floor(saved_column[len(saved_column)-1])+1
print("Smallest value greater than maximum",b)
#Q1-d
def calc_MidPoints(a, b, h1):
    midpoints = []
    i = round(a + (h1/2),2)
    midpoints.append(i)
    while i+h1<b:
        i = round(i + h1,2)
        midpoints.append(i)
    return midpoints
def w(u):
    if u<=0.5 and u>-0.5:
        return 1
    return 0
def generate_coordinates(a,b,saved_column,h):
    midpoints =calc_MidPoints(a,b,h)
    coordinate = []
    for i in midpoints:
        wu=0
        for x in saved_column:
            u = (x-i)/h
            wu = wu+w(u)
            density_estimation= wu/(saved_column*h)
            coordinate.append([i,density_estimation])
    header = ["mi", "p(mi)"]
    print("Coordinates of Density Estimator")
    print(tabulate(coordinate, headers=header))
    bin_interval = [round(x - (h / 2),2) for x in midpoints]
    bin_interval.append(b)
    y, edges = np.histogram(saved_column, bin_interval)
    plt.hist(saved_column, bin_interval)
    plt.plot(midpoints, y, '-*')
    plt.title("Plot")
    plt.ylabel("Density estimator")
    plt.xlabel("Mid points")
    plt.show()
l = len(saved_column)
h1=0.25
print(generate_coordinates(a,b,saved_column,h1))
#Q1 -e
h2=0.5
print(generate_coordinates(a,b,saved_column,h2))
#Q1 -f
h3=1
print(generate_coordinates(a,b,saved_column,h3))
#Q1 -g
h4=2
print(generate_coordinates(a,b,saved_column,h4))
print("------------------------------------------------------------------------------------------------------------------------")
#Q2
def five_number_summary(xval):
    xval.sort()
    Quartile1=np.percentile(xval,25)
    Quartile2=np.percentile(xval,50)
    Quartile3=np.percentile(xval,75)
    print("The minimum value is: ",xval[0])
    print("First quartile Q1 is ",Quartile1)
    print("The median (Quartile 2) is ",Quartile2)
    print("The third quartile is ",Quartile3)
    InterQuartileRange = Quartile3 - Quartile1
    print("Interquartile range is ", InterQuartileRange)
    Lower_whisker = max(xval[0],(Quartile1-1.5*InterQuartileRange))
    Upper_whisker = min(xval[len(xval)-1],(Quartile3+1.5*InterQuartileRange))
    return[Lower_whisker,Upper_whisker]
#Q2 - a
print("Five number summary of X")
w = five_number_summary(saved_column)
#Q2 - b
group_0=[]
group_1=[]
print("Five number summary of group 0")
#group_0 = ns[ns['group']==0]['x'].values.copy()
val = ns.group.values.copy()
for i in range(len(val)):
    if (val[i]==0):
        group_0.append(ns.x[i])
    else:
        group_1.append(ns.x[i])
w0 = five_number_summary(group_0)
print("Five number summary of group 1")
#group_1 = ns[ns['group']==1]['x'].values.copy()
w1 = five_number_summary(group_1)
#Q2-c
B= plt.boxplot(saved_column)
plt.title("Box plot")
plt.show()
#Q2-d
values = (saved_column,group_0,group_1)
B1=plt.boxplot(values)
plt.title("Box plot for X, Box plot for X(group 0),Box plot for X(group 1)")
plt.xticks([1, 2, 3], ["X", "X (group 0)", "X (group 1)"])
plt.show()
#printing outliers
xout=[]
xout_0= []
xout_1 =[]
for x in saved_column:
    if((x < w[0]) or (x > w[1])):
        xout.append(x)
print("outlier for X",str(xout))
for x in group_0:
    if((x < w0[0]) or (x > w0[1])):
        xout_0.append(x)
print("outlier for group 0",xout_0)
for x in group_1:
    if((x < w1[0]) or (x > w1[1])):
        xout_1.append(x)
print("outlier for group 1",xout_1)
#Q3 - a
Fraud =pd.read_csv("Fraud.csv")
no_of_frauds = Fraud.FRAUD.values.copy()
sum = 0
for i in range(len(no_of_frauds)):
    sum =sum+no_of_frauds[i]
percent = (sum*100)/len(no_of_frauds)
print(round(percent,2))
#Q3 - b
true_fraud = []
false_fraud = []
ts = Fraud.TOTAL_SPEND.values.copy()
dv = Fraud.DOCTOR_VISITS.values.copy()
nc = Fraud.NUM_CLAIMS.values.copy()
md = Fraud.MEMBER_DURATION.values.copy()
op = Fraud.OPTOM_PRESC.values.copy()
nm = Fraud.NUM_MEMBERS.values.copy()
value = Fraud.FRAUD.values.copy()
def box_plot(var):
    value = Fraud.FRAUD.values.copy()
    for i in range(len(value)):
        if(value[i]==1):
            true_fraud.append(var[i])
        else:
            false_fraud.append(var[i])
    alpha =[false_fraud,true_fraud]
    plt.boxplot(alpha,vert=False)
    plt.yticks([1, 2], ["Non Fraudulent", "Fraudulent"])
    plt.show()
box_plot(ts)
box_plot(dv)
box_plot(nc)
box_plot(md)
box_plot(op)
box_plot(nm)
#Q3 -c
matrix = np.matrix([ts, dv, nc, md, op, nm]).transpose()
matrix_transformed = matrix.transpose() * matrix
print("t(x) * x = \n", matrix_transformed)
eigenvalues, eigenvectors =la.eigh(matrix_transformed)
print("Eigenvalues of x = \n", eigenvalues)
print("Eigenvectors of x = \n",eigenvectors)
file = []
for i in range(0,len(eigenvalues)):
    if eigenvalues[i] >= 1:
        file.append(i)
evalsfilt = eigenvalues[file]
print("Number of Dimesions used is ", (len(evalsfilt)))
transf = eigenvectors *la.inv(np.sqrt(np.diagflat(eigenvalues)))
print("Transformation Matrix = ", transf)
transf_matrix = matrix * transf
print("The Transformed x = ", transf_matrix)
xtx1 = transf_matrix.transpose() * transf_matrix
print("Expect an Identity Matrix = ", xtx1)
#Q3 - d
neigh = kn(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(transf_matrix)
kNNSpec = kNC(n_neighbors = 5)
nbrsC = kNNSpec.fit(transf_matrix,np.array(no_of_frauds))
scor = nbrsC.score(transf_matrix, np.array(no_of_frauds))
print("The result of score function is " + str(round(scor,4)))
#Q3 - e
focal = [7500, 15, 3, 127, 2, 2]
print("The focal observation is ",str(focal))
transfFocal = focal*transf
print("The Transformed focal observation is " + str(transfFocal))
myNeighbors_t = nbrs.kneighbors(transfFocal, return_distance = False)
print("The indices of the five neighbors of the focal are " + str(myNeighbors_t))
myNeighbors_t_values = matrix[myNeighbors_t]
print("The input and target values of the nearest neighbors are \n")
heads = ["ID", "TOTAL_SPEND", "DOCTOR_VISITS", "NUM_CLAIMS", "MEMBER_DURATION", "OPTOM_PRESC", "NUM_MEMBERS", "Target Value"]
resolution = np.hstack((myNeighbors_t.transpose(),myNeighbors_t_values[0,:,:]))
yac = np.array([no_of_frauds]).transpose()[myNeighbors_t]
res = np.hstack((resolution, yac[0,:,:]))
print(pd.DataFrame(res, columns=heads))
