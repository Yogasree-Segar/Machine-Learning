The code for the questions 1 and 3 were refered to the code provided by the Professor.

#Q1
a

import pandas as pd
import matplotlib.pyplot as plt
G = pd.read_csv('Groceries.csv',delimiter=',')
nItemPurchase = G.groupby('Customer').size()
unique_item = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
plt.bar(unique_item.index.values,unique_item)
plt.show()
import numpy as np
#print(unique_item)
sum = np.sum(unique_item)
freq= unique_item.cumsum()
median = freq[freq>=sum/2].index[0]
q1 = freq[freq>=sum/4].index[0]
q3 = freq[freq>=3*(sum/4)].index[0]
print("Median",median)
print("25th percentile",q1)
print("75th percentile",q3)

b

import pandas
ListItem = G.groupby(['Customer'])['Item'].apply(list).values.tolist()
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(ItemIndicator, min_support = (75/sum), max_len = 4, use_colnames = True)
print(frequent_itemsets.count())

c

from mlxtend.frequent_patterns import association_rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print(assoc_rules.count()["antecedents"])

d

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

e

from mlxtend.frequent_patterns import association_rules
from tabulate import tabulate
assoc_rules_greater_than_60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print(assoc_rules_greater_than_60)
print(" ")
print(str(assoc_rules_greater_than_60['antecedents'][0]) + " -> " + str(assoc_rules_greater_than_60['consequents'][0]))
print("Support: " + str(assoc_rules_greater_than_60['support'][0]))
print("Lift: " + str(assoc_rules_greater_than_60['lift'][0]) +"\n")
print(str(assoc_rules_greater_than_60['antecedents'][1]) + " -> " + str(assoc_rules_greater_than_60['consequents'][1]))
print("Support: " + str(assoc_rules_greater_than_60['support'][1]))
print("Lift: " + str(assoc_rules_greater_than_60['lift'][1]) +"\n")
print(str(assoc_rules_greater_than_60['antecedents'][2]) + " -> " + str(assoc_rules_greater_than_60['consequents'][2]))
print("Support: " + str(assoc_rules_greater_than_60['support'][2]))
print("Lift: " + str(assoc_rules_greater_than_60['lift'][2]) +"\n")
print(str(assoc_rules_greater_than_60['antecedents'][3]) + " -> " + str(assoc_rules_greater_than_60['consequents'][3]))
print("Support: " + str(assoc_rules_greater_than_60['support'][3]))
print("Lift: " + str(assoc_rules_greater_than_60['lift'][3]) +"\n")

#Q2
a
car = pd.read_csv('cars.csv')
Type = car['Type'].values
Origin = car['Origin'].values
DriveTrain = car['DriveTrain'].values
Cylinders = car['Cylinders'].values
print(car['Type'].unique())
print(car['DriveTrain'].unique())
Suv =0
Sedan = 0
Sports = 0
Wagon = 0
Truck = 0
Hybrid = 0
for i in Type:
    if (i=="SUV"):
        Suv=Suv+1
    elif(i=="Sedan"):
        Sedan = Sedan + 1
    elif(i=="Sports"):
        Sports = Sports + 1
    elif(i=="Wagon"):
        Wagon = Wagon + 1
    elif(i=="Truck"):
        Truck = Truck + 1
    elif(i=="Hybrid"):
        Hybrid = Hybrid + 1
print("Frequency of Suv",Suv)
print("Frequency of Sedan",Sedan)
print("Frequency of Sports",Sports)
print("Frequency of Wagon",Wagon)
print("Frequency of Truck",Truck)
print("Frequency of Hybrid",Hybrid)

b

AWD = 0
FWD = 0
RWD = 0
for i in DriveTrain:
    if (i=="AWD"):
        AWD=AWD+1
    elif(i=="FWD"):
        FWD = FWD + 1
    elif(i=="RWD"):
        RWD = RWD + 1
print("Frequency of AWD",AWD)
print("Frequency of FWD",FWD)
print("Frequency of RWD ",RWD)

c

Asia = 0
Europe = 0
print(car['Origin'].unique())
for i in Origin:
    if (i=="Asia"):
        Asia=Asia+1
    elif(i=="Europe"):
        Europe = Europe + 1
print("Frequency of Asia",Asia)
print("Frequency of Europe",Europe)
Distance_metric = ((1/Asia) + (1/Europe))
print("Distance metric",Distance_metric)

d

Five = 0
Missing = 0
print(car['Cylinders'].unique())
for i in Cylinders:
    if (i==5):
        Five=Five+1
    elif(np.isnan(i)):
        Missing = Missing + 1
print("Frequency of Five",Five)
print("Frequency of Missing numbers",Missing)
Distance_metric_cylinder = ((1/Five) + (1/Missing))
print("Distance metric",Distance_metric_cylinder)

#Q3
a
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.neighbors
from matplotlib import style
Four_Circle_Data = pd.read_csv('FourCircle.csv', delimiter=',')
plt.scatter(np.array(Four_Circle_Data['x']), np.array(Four_Circle_Data['y']))
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

b

trainData = Four_Circle_Data[['x','y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616).fit(trainData)
Four_Circle_Data['KMClusterLabel'] = kmeans.labels_
plt.scatter(Four_Circle_Data['x'], Four_Circle_Data['y'], c = Four_Circle_Data['KMClusterLabel'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

c

kNNSpec = neighbors.NearestNeighbors(n_neighbors = 1, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
Adjacency = numpy.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = numpy.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = numpy.arange(1,2,1) 
plt.plot(sequence, evals[0:1,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid("both")
plt.show()

d

kNNSpec = neighbors.NearestNeighbors(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
Adjacency = numpy.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())
print("Adjacency matrix: ",Adjacency)
Degree = numpy.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
print("Degree matrix: ",Degree)
Lmatrix = Degree - Adjacency
print("Laplace matrix: ",Lmatrix)
evals, evecs = np.linalg.eigh(Lmatrix)
print("Eigen values: ",evals)
print(evals[0:5]

e

zero_eigen = evecs[:,[0,1,2,3,4]]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616).fit(zero_eigen)
Four_Circle_Data['KMClusterLabel'] = kmeans.labels_
plt.scatter(Four_Circle_Data['x'], Four_Circle_Data['y'], c = Four_Circle_Data['KMClusterLabel'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
