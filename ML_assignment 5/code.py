Q1 - a

import numpy as np
import pandas as pd
data = pd.read_csv("SpiralWithCluster.csv", delimiter = ",")
one = len(data[data["SpectralCluster"] == 1]) / len(data)
print(one * 100)
x = data[['x','y']]
y = data['SpectralCluster']

Q1 - b

import sklearn.neural_network as nn
import sklearn.metrics as metrics
res = []
columns = []
function = ["identity","logistic","tanh","relu"]
layers = 5
neurons = 10
for f in ["identity","logistic","tanh","relu"]:
    for l in range(1, layers+1):
        for n in range(1, neurons+1):
            #loss, misclass, niteration_output, y_pred = Neural(f,l,n)
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (n,)*l, activation=f,solver = 'lbfgs',learning_rate_init = 0.1,max_iter = 5000,random_state = 20200408)
            thisFit = nnObj.fit(x,y)
            y_predicted_probability = nnObj.predict_proba(x)
            y_predicted = np.where(y_predicted_probability[:,1] >= one, 1,0)
            mlp_loss = nnObj.loss_
            mlp_missclassification_rate = 1 - metrics.accuracy_score(y, y_predicted)
            Activation_function = nnObj.out_activation_
            no_of_iteration = nnObj.n_iter_
    #RSquare=metrics.r2_score(y,y_pred)
            res.append([f,l,n,no_of_iteration,mlp_loss,mlp_missclassification_rate])
columns = pd.DataFrame(res)
#columns.append([f,l,n,no_of_iteration,loss, misclass])    

Q1 - c
print(Activation_function)

Q1 - d

dfcolumns=pd.DataFrame(res, columns=['ActivationFunc','NoOfLayers','Neurons/layers','NoOfIters','Loss','Misclassification%'])
print(dfcolumns)
bestModels = []
for actfn in ["identity", "logistic", "relu", "tanh"]:
    modelPerf = dfcolumns[dfcolumns['ActivationFunc'] == actfn]
    bestModel = modelPerf[modelPerf['Loss'] == min(modelPerf['Loss'])]
    bestModels.append(bestModel.values[0].tolist())
bestRes = pd.DataFrame(bestModels)
bestRes.columns = ["Activation function", "Number of layers", "Number of neurons per layer", "Number of iterations performed", "Loss value", "Misclassification rate"]
print(bestRes)
dfcolumns.sort_values(by = ['Loss']).iloc[0]
relu = dfcolumns.loc[dfcolumns['ActivationFunc'] == 'relu'].sort_values(by=['Loss']).iloc[0]
tanh = dfcolumns.loc[dfcolumns['ActivationFunc'] == 'tanh'].sort_values(by=['Loss']).iloc[0]
identity = dfcolumns.loc[dfcolumns['ActivationFunc'] == 'identity'].sort_values(by=['Loss']).iloc[0]
logistic = dfcolumns.loc[dfcolumns['ActivationFunc'] == 'logistic'].sort_values(by=['Loss']).iloc[0]
Result = (relu, tanh, identity, logistic)
result = pd.DataFrame(Result)
print(result)
leastLoss = result[result['Loss'] == min(result['Loss'])]
print(leastLoss.iloc[0])

Q1 - e

import matplotlib.pyplot as plt

nnObj = nn.MLPClassifier(hidden_layer_sizes = (int(leastLoss['Neurons/layers']),) * int(leastLoss['NoOfLayers']), activation = leastLoss['ActivationFunc'].values[0],verbose = False,solver = 'lbfgs', learning_rate_init = 0.1, max_iter =5000, random_state = 20200408)
thisFit = nnObj.fit(x,y)
y_predicted_probability = nnObj.predict_proba(x)
y_predicted = np.where(y_predicted_probability[:,1] >= one, 1,0)
x['Predicted_class'] = y_predicted
mlp_mean = x.groupby('Predicted_class').mean()
color = ['red','blue']
plt.figure(figsize=(10,10))
for i in range(2):
    Data = x[x['Predicted_class'] ==i]
    plt.scatter(x = Data['x'],y=Data['y'],c=color[i],label =i,s=25)
plt.grid(True)
plt.xlabel('X') 
plt.ylabel('Y') 
plt.title("Graph for lowest loss")
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 25)
plt.show()

Q1 - f

print(data)
data["Pred Prob(SpectralCluster = 1)"] = y_predicted_probability[:,1]
g = data.groupby("SpectralCluster")
print(g.get_group(1).describe())


Q2 - a

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sklearn.svm as svm
data = pd.read_csv("SpiralWithCluster.csv", delimiter = ",")
xTrain = data[['x','y']]
yTrain = data[['SpectralCluster']]
svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',random_state = 20200408,max_iter = -1)
thisFit = svm_Model.fit(xTrain,yTrain)
Class = thisFit.predict(xTrain)
y_predictClass = thisFit.predict(xTrain)
print('Accuracy = ',metrics.accuracy_score(yTrain, y_predictClass))
xTrain['PredictedClass_'] = y_predictClass
print('Intercept = ',thisFit.intercept_)
print('Coefficients = ',thisFit.coef_)

Q2 - b
acc = metrics.accuracy_score(yTrain, y_predictClass)
Mis_class_rate = 1-acc
print(Mis_class_rate)

Q2 - c

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-6,6)
yy = a * xx - (thisFit.intercept_[0]) / w[1]
carray = ['red','blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = xTrain[xTrain['PredictedClass_'] ==(i)]
    plt.scatter(x = subData['x'], y = subData['y'], c = carray[i], label = i,s=25)
plt.plot(xx,yy,color = 'black', linestyle = ':')
#plt.plot(xx,yy[:,1],color = 'black', linestyle = '-')
#plt.plot(xx,yy[:,2],color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machine on input data')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-7,7)
plt.ylim(-7,7)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1,1), fontsize = 14)
plt.show()

Q2 - d

xTrain['radius'] = np.sqrt(xTrain['x'] **2 + xTrain['y']**2)
xTrain['theta'] = np.arctan2(xTrain['y'],xTrain['x'])
def customArcTan (z):
    theta = np.where(z<0.0,2.0*np.pi+z, z)
    return(theta)
xTrain['theta'] = xTrain['theta'].apply(customArcTan)
xTrain_2 = xTrain[['radius','theta']]
xTrain['SpectralCluster'] = yTrain
yTrain_2 = yTrain
carray = ['red','blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = xTrain[xTrain['SpectralCluster'] ==i]
    plt.scatter(x = subData['radius'], y = subData['theta'], c = carray[i], label = i, s =25)
plt.grid(True)
plt.title('Polar Coordinates Plot')
plt.xlabel('Radius')
plt.ylabel('Angle in radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1,1), fontsize = 25)
plt.show()

Q2 - e

print(xTrain_2)
lone_point = xTrain.loc[(xTrain['radius'] <= 1.5) & (xTrain['theta'] >= 6) & (xTrain['SpectralCluster'] == 0)]
strip_1= xTrain.loc[(xTrain['radius'] <= 3) & (xTrain['theta'] >= 3) & (xTrain['SpectralCluster'] == 1)]
strip_3 = xTrain.loc[(xTrain['radius'] >= 2.5) & (xTrain['theta'] <= 3.2) & (xTrain['SpectralCluster'] == 1)]
strip_2 = pd.concat([xTrain,lone_point,strip_1,strip_3])
strip_2 = strip_2.drop_duplicates(keep = False)
lone_point['group'] = 0
strip_1['group'] = 1
strip_2['group'] = 2
strip_3['group'] = 3
new_data = pd.concat([lone_point,strip_1,strip_2,strip_3])
carray = ['red','blue','green','black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = new_data[new_data['group'] ==i]
    plt.scatter(x = subData['radius'], y = subData['theta'], c = carray[i], label = i, s =25)
plt.grid(True)
plt.title('Polar Coordinates Plot')
plt.xlabel('Radius')
plt.ylabel('Angle in radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1,1), fontsize = 25)
plt.show()

Q2 - f

print(new_data)
new_data = new_data[['radius','theta', 'group']]
group = [0,1]
svm0 = new_data[(new_data['group'] ==group[0]) | (new_data['group'] == group[1])]
xTrain_new = svm0[['radius', 'theta']]
yTrain_new = svm0[['group']]
svm_Model_0 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', random_state = 20200408, max_iter = -1)
thisFit_new = svm_Model_0.fit(xTrain_new,yTrain_new)
y_predict_class_new = thisFit_new.predict(xTrain_new)
xTrain_new['PredictedClass_'] = y_predict_class_new
w0 = thisFit_new.coef_[0]
a0 = -w0[0] / w0[1]
xx0 = np.linspace(1,6)
yy_0 = a0 * xx0 - (thisFit_new.intercept_[0]) / w0[1]
print('Intercept', thisFit_new.intercept_)
print('Coefficients = ',thisFit_new.coef_)
group1 = [1,2]
svm1 = new_data[(new_data['group'] ==group1[0]) | (new_data['group'] == group1[1])]
xTrain_new_1 = svm1[['radius', 'theta']]
yTrain_new_1 = svm1[['group']]
svm_Model_1 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', random_state = 20200408, max_iter = -1)
thisFit_new_1 = svm_Model_1.fit(xTrain_new_1,yTrain_new_1)
y_predict_class_new_1 = thisFit_new_1.predict(xTrain_new_1)
xTrain_new_1['PredictedClass_'] = y_predict_class_new_1
w1 = thisFit_new_1.coef_[0]
a1 = -w1[0] / w1[1]
xx1 = np.linspace(1,6)
yy_1 = a1 * xx1 - (thisFit_new_1.intercept_[0]) / w1[1]
print('Intercept', thisFit_new_1.intercept_)
print('Coefficients = ',thisFit_new_1.coef_)
group2 = [2,3]
svm2 = new_data[(new_data['group'] ==group2[0]) | (new_data['group'] == group2[1])]
xTrain_new_2 = svm2[['radius', 'theta']]
yTrain_new_2 = svm2[['group']]
svm_Model_2 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', random_state = 20200408, max_iter = -1)
thisFit_new_2 = svm_Model_2.fit(xTrain_new_2,yTrain_new_2)
y_predict_class_new_2 = thisFit_new_2.predict(xTrain_new_2)
xTrain_new_2['PredictedClass_'] = y_predict_class_new_2
w2 = thisFit_new_2.coef_[0]
a2 = -w2[0] / w2[1]
xx2 = np.linspace(1,4.5)
yy_2 = a2 * xx2 - (thisFit_new_2.intercept_[0]) / w2[1]
print('Intercept', thisFit_new_2.intercept_)
print('Coefficients = ',thisFit_new_2.coef_)

Q2 - g

carray = ['red','blue','green','black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = new_data[new_data['group'] == (i)]
    plt.scatter(x = subData['radius'], y = subData['theta'], c = carray[i], label = (i), s = 25)
plt.plot(xx0, yy_0, color = 'black', linestyle = 'dotted')
plt.plot(xx1, yy_1, color = 'black', linestyle = 'dotted')
plt.plot(xx2, yy_2, color = 'black', linestyle = 'dotted')
plt.grid(True)
plt.title('SVM on data for New Target Variable')
plt.xlabel('Radius')
plt.ylabel('Angle Theta (in Radians)')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 25)
plt.xlim(1, 5)
plt.show()

Q2 - h

h0_xx = xx0 * np.cos(yy_0)
h0_yy = xx0 * np.sin(yy_0)

h1_xx = xx1 * np.cos(yy_1)
h1_yy = xx1 * np.sin(yy_1)

h2_xx = xx2 * np.cos(yy_2)
h2_yy = xx2 * np.sin(yy_2)
carray = ['red','blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = data[data['SpectralCluster'] == (i)]
    plt.scatter(x = subData['x'], y = subData['y'], c = carray[i], label = (i), s = 25)
plt.plot(h0_xx, h0_yy, color = 'black', linestyle = 'dotted')
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = 'dotted')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = 'dotted')
plt.grid(True)
plt.title('SVM Cartesian System')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 25)
plt.show()
