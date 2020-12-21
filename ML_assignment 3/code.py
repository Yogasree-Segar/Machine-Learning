from sklearn.model_selection import train_test_split
import pandas as pd
cars = pd.read_csv("claim_history.csv")
train, test = train_test_split(cars, test_size=0.25, random_state=60616)
print(len(train))
print(len(test))
print(len(train[train['CAR_USE']=='Commercial']))
print(len(train[train['CAR_USE']=='Commercial'])/len(train))
print(len(train[train['CAR_USE']=='Private']))
print(len(train[train['CAR_USE']=='Private'])/len(train))
print(len(test[test['CAR_USE']=='Commercial']))
print(len(test[test['CAR_USE']=='Commercial'])/len(test))
print(len(test[test['CAR_USE']=='Private']))
print(len(test[test['CAR_USE']=='Private'])/len(test))
print(len(train[train['CAR_USE']=='Commercial']))
print("Probability that the Car_use in train is commercial:")
tot_commercial = len(train[train['CAR_USE']=='Commercial']) + len(test[test['CAR_USE']=='Commercial'])
commercial_train_prob = len(train[train['CAR_USE']=='Commercial'])/tot_commercial
print(round(commercial_train_prob,2))
print("Probability that the Car_use in test is Private:")
tot_private = len(test[test['CAR_USE']=='Private']) + len(train[train['CAR_USE']=='Private'])
private_test_prob = len(test[test['CAR_USE']=='Private'])/tot_private
print(round(private_test_prob,2))
from math import log 
private_train_prob = len(train[train['CAR_USE']=='Private'])/tot_commercial
print(round(private_train_prob,2))
root_entropy = -1 * ((commercial_train_prob)*log(commercial_train_prob,2) + (private_train_prob)*log(private_train_prob,2))
print("\n\nEntropy of root node is " + str(root_entropy))
def calcEntropy(total, commercial, private):
    p_commercial = commercial/total
    p_private = private/total
    log_p_commercial = log(p_commercial,2) if p_commercial != 0 else 0
    log_p_private = log(p_private,2) if p_private != 0 else 0
    entropy = -1*(p_commercial * log_p_commercial + p_private * log_p_private)
    return entropy
import itertools
import pandas as pd
import numpy as np
import math
from math import log
import sklearn.metrics as metrics

def allPossibleSets(S, varType):
    if varType == "Nominal":
            relS=set()
            n = len(S)
            k=int(n/2)
            for i in range(1,k):
                relS.update(set(itertools.combinations(S, i)))
            kth_subset = set(itertools.combinations(S, k))
            kth_subset = set(itertools.islice(kth_subset, int(len(kth_subset)/2)))
            relS.update(kth_subset)
            return relS
    elif varType == "Ordinal":
        relL = []
        n=len(S)
        for i in range(1,n):
            relL.append(set(itertools.islice(S, i)))
        relS = set(frozenset(i) for i in relL)
        return [list(x) for x in relS]
def EntropyPredictor(data, pred, varType):
    possibleSets = allPossibleSets(set(pred), varType)
    commercialTotal = len(data[data['CAR_USE'] == 'Commercial'])
    privateTotal = len(data[data['CAR_USE'] == 'Private'])
    nTotal = commercialTotal + privateTotal
    entropyList=[]
    for pset in possibleSets:
        filtData = data[pred.isin(pset)]
        total = len(filtData)
        commercial = len(filtData[filtData['CAR_USE'] == 'Commercial'])
        private = len(filtData[filtData['CAR_USE'] == 'Private'])
        entropy = calcEntropy(total, commercial, private)
        entropy2 = calcEntropy((nTotal - total), (commercialTotal - commercial), (privateTotal - private))
        tt = nTotal - total
        splitEntropy = (total/nTotal)*entropy + (tt/nTotal)*entropy2
        entropyList.append([pset, splitEntropy])
    splitEntropys = np.array(entropyList)[:,1]
    minSplitEntropy = min(splitEntropys)
    return entropyList[np.where(splitEntropys == minSplitEntropy)[0][0]]
def SplitCondition(node):
    car_type_entropy = EntropyPredictor(node, node['CAR_TYPE'], "Nominal")
    occupation_entropy = EntropyPredictor(node, node['OCCUPATION'], "Nominal")
    education_entropy = EntropyPredictor(node, node['EDUCATION'], "Ordinal")
    min_Ent_Pred = [car_type_entropy, occupation_entropy, education_entropy]
    all_entropy = np.array(min_Ent_Pred)[:,1]
    splitCondition = min_Ent_Pred[np.where(all_entropy == min(all_entropy))[0][0]]
    return splitCondition
splitCondition = SplitCondition(train)
print("\n\nSplit condition is " + str(splitCondition[0]))
True_node = train[(train['OCCUPATION'] == 'Blue Collar') | (train['OCCUPATION'] == 'Unknown') | (train['OCCUPATION'] == 'Student') ]

False_node = train[~train.isin(True_node)].dropna()

print("True predictor name and values:")
print("Car Type: " + str(set(True_node['CAR_TYPE'])) + "\nOccupation: " + str(set(True_node['OCCUPATION'])) + "\nEducation: " + str(set(True_node['EDUCATION'])))

print("False predictor name and values: ")
print("Car Type: " + str(set(False_node['CAR_TYPE'])) + "\nOccupation: " + str(set(False_node['OCCUPATION'])) + "\nEducation: " + str(set(False_node['EDUCATION'])))
print("\n\nSplit condition of First layer is " + str(splitCondition[1]))
split_Condition_True_Node =  SplitCondition(True_node)
print(split_Condition_True_Node)
split_Condition_False_Node =  SplitCondition(False_node)
print(split_Condition_False_Node)
print("The total no. of leaves = 4")
nodeTT = True_node[ (True_node['EDUCATION'] == 'Below High School') ]
nodeTF = True_node[~True_node.isin(nodeTT)].dropna()
nodeFT = False_node[ (False_node['CAR_TYPE'] == 'Minivan') | (False_node['CAR_TYPE'] == 'SUV') | (False_node['CAR_TYPE'] == 'Sports Car')]
nodeFF = False_node[~False_node.isin(nodeFT)].dropna()

def countTargetVals(node):
    print("Number of values where Car Use is Commercial " + str(len(node[node['CAR_USE'] == 'Commercial'])))
    print("Number of values where Car Use is Private " + str(len(node[node['CAR_USE'] == 'Private'])))
    com = len(node[node['CAR_USE'] == 'Commercial'])
    pri = len(node[node['CAR_USE'] == 'Private'])
    total_value = com + pri
    return com / total_value

print("Condition True - True")
ptt = countTargetVals(nodeTT)

print("\nCondition True - False")
ptf = countTargetVals(nodeTF)

print("\nCondition False - True")
pft = countTargetVals(nodeFT)

print("\nCondition False - False")
pff = countTargetVals(nodeFF)
#Kolmogorov Smirnov cutoff
import numpy
threshold = float((cars.groupby('CAR_USE').size() / cars.shape[0])['Commercial'])
cutoff = numpy.where(threshold > 1.0, numpy.nan,threshold)
print(cutoff)

testData = test[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()
nodeTrueTest = test[ (test['OCCUPATION'] == 'Blue Collar') | (testData['OCCUPATION'] == 'Student') | 
        (test['OCCUPATION'] == 'Unknown') ]
nodeFalseTest = test[~test.isin(nodeTrueTest)].dropna()
nodeTTtest = nodeTrueTest[ (nodeTrueTest['EDUCATION'] == 'Below High School') ]
nodeTFtest = nodeTrueTest[~nodeTrueTest.isin(nodeTTtest)].dropna() 
nodeFTtest = nodeFalseTest[ (nodeFalseTest['CAR_TYPE'] == 'Minivan') | (nodeFalseTest['CAR_TYPE'] == 'SUV') | (nodeFalseTest['CAR_TYPE'] == 'Sports Car') ]
nodeFFtest = nodeFalseTest[~nodeFalseTest.isin(nodeFTtest)].dropna()

threshold = float((train.groupby('CAR_USE').size() / train.shape[0])['Commercial'])
print("Threshold is", threshold)
testData['Predicted_Probability'] = 0
nodeTTtest['Predicted_Probability'] = ptt
nodeTFtest['Predicted_Probability'] = ptf
nodeFTtest['Predicted_Probability'] = pft
nodeFFtest['Predicted_Probability'] = pff
leafNodes = pd.concat([nodeTTtest, nodeTFtest, nodeFTtest, nodeFFtest])
leafNodes.loc[leafNodes['Predicted_Probability'] >= threshold, 'Predicted_Class'] = "Commercial"
leafNodes.loc[leafNodes['Predicted_Probability'] < threshold, 'Predicted_Class'] = "Private"
# Determine the predicted class of Y
Y = np.array(leafNodes['CAR_USE'].tolist())
nY = Y.shape[0]
predProbY = np.array(leafNodes['Predicted_Probability'].tolist())
predY = np.empty_like(Y)
for i in range(nY):
    if (predProbY[i] > 0.5):
        predY[i] = 'Commercial'
    else:
        predY[i] = 'Private'

# Calculate the Root Average Squared Error
RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Commercial'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = np.sqrt(RASE/nY)

# Calculate the Root Mean Squared Error
Y_true = 1.0 * np.isin(Y, ['Commercial'])
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = np.sqrt(RMSE)

                              
AUC = metrics.roc_auc_score(Y_true, predProbY)
accuracy = metrics.accuracy_score(Y, predY)

print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))
gini_coeff = 2 * AUC -1
print("The gini coefficient is:", gini_coeff)
import numpy
threshold = float((train.groupby('CAR_USE').size() / train.shape[0])['Commercial'])
cutoff = numpy.where(threshold > 1.0, numpy.nan,threshold)
print(cutoff)
print(1-cutoff)
predY_KS = numpy.empty_like(Y)
for i in range(nY):
    if (predProbY[i] > cutoff):
        predY_KS[i] = 'Commercial'
    else:
        predY_KS[i] = 'Private'
acc_KS = accuracy_score(Y,predY_KS)
print(acc_KS)
mis_KS = 1-acc_KS
print(mis_KS)
import matplotlib.pyplot as plt
Y = np.array(leafNodes['CAR_USE'].tolist())
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')
OneMinusSpecificity = np.append([0], OneMinusSpecificity)
Sensitivity = np.append([0], Sensitivity)
OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])
plt.plot(OneMinusSpecificity, Sensitivity)
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("Receiver Operating Characteristic curve")
plt.show()