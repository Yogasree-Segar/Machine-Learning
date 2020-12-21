Q1b-d
import numpy
import pandas
import scipy
import sympy
import statsmodels.api as stats
def create_interaction(inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)
def build_mnlogit(fullX,y,debug = 'N'):
    nFullParam = fullX.shape[1]
    y_category = y.cat.categories
    nYCat = len(y_category)
    
    reduce_form,inds = sympy.Matrix(fullX.values).rref()
    if(debug == 'Y'):
        print('Column numbers of non-red columns:')
        print(inds)
        
    X = fullX.iloc[:,list(inds)]
    
    thisDF = len(inds) * (nYCat - 1)
    
    logit = stats.MNLogit(y,X)
    thisFit = logit.fit(method = 'Newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)
    
    if(debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n",thisParameter)
        print("Log Likelihood:\n",thisLLK)
        print("Number of Free parameters:\n",thisDF)
        
    workParams = pandas.DataFrame(numpy.zeros(shape =(nFullParam,(nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullPArams = fullParams.drop(columns = '0_x').fillna(0.0)
    
    return(thisLLK, thisDF, fullParams)
    
hmeq = pandas.read_csv('Purchase_Likelihood.csv', delimiter = ',',usecols = ['group_size','homeowner','married_couple','insurance'])
hmeq = hmeq.dropna()
y=hmeq['insurance'].astype('category')
print(y)
xG = pandas.get_dummies(hmeq[['group_size']].astype('category'))
xH = pandas.get_dummies(hmeq[['homeowner']].astype('category'))
xM = pandas.get_dummies(hmeq[['married_couple']].astype('category'))
print(xG)
print(xH)
print(xM)
designX = pandas.DataFrame(y.where(y.isnull(),1))
LLK0,DF,fullParams0 = build_mnlogit(designX,y,debug='Y')
##Intercept + group_size

designX = stats.add_constant(xG, prepend = True)
LLK_1G,DF_1G,fullParams_1G = build_mnlogit(designX,y,debug = 'N')
testDev = 2* (LLK_1G - LLK0)
testDF = DF_1G - DF
testPValue = scipy.stats.chi2.sf(testDev,testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('Degree of Freedom = ', testDF)
print('Significance = ', testPValue)
import numpy

print('Log likelihood',LLK_1G)
print('No. of free parameters', DF_1G)
print("The significance is: 4.347870389027117e-210 ")
Imp1 = numpy.log10(testPValue)
print("Importance :", -Imp1)
##Intercept + group_size + homeover

designX = xG
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
import numpy

print('Log likelihood',LLK_1G_1H)
print('No. of free parameters', DF_1G_1H)
print("The significance is: 0.0 ")
Imp2 = numpy.log10(testPValue)
print("Importance :", -Imp2)
##Intercept + group_size + homeowner + married_couple

designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H_1M = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H)
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
import numpy

print('Log likelihood',LLK_1G_1H_1M)
print('No. of free parameters', DF_1G_1H_1M)
print("The significance is: 4.3064572185369587e-19 ")
Imp3 = numpy.log10(testPValue)
print("Importance :", -Imp3)
##Intercept + group_size + homeowner + married_couple + group_size * homeowner

designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

xGH = create_interaction(xG,xH)
designX = designX.join(xGH)

designX = stats.add_constant(designX, prepend=True)
LLK_2GH, DF_2GH, fullParams_2GH = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GH - LLK_1G_1H_1M)
testDF = DF_2GH - DF_1G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

import numpy

print('Log likelihood',LLK_2GH)
print('No. of free parameters', DF_2GH)
print("The significance is: 5.512105967934721e-52 ")
Imp4 = numpy.log10(testPValue)
print("Importance :", -Imp4)
##Intercept + group_size + homeowner + married_couple + group_size * married_couple

designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

xGH = create_interaction(xG,xH)
designX = designX.join(xGH)

xGM = create_interaction(xG,xM)
designX = designX.join(xGM)

designX = stats.add_constant(designX, prepend=True)
LLK_2GM, DF_2GM, fullParams_2GH = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GM - LLK_2GH)
testDF = DF_2GM - DF_2GH
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
import numpy

print('Log likelihood',LLK_2GM)
print('No. of free parameters', DF_2GM)
print("The significance is: 1.4597001212103711e-295 ")
Imp5 = numpy.log10(testPValue)
print("Importance :", -Imp5)

##Intercept + group_size + homeowner + married_couple + group_size * married_couple + group_size * homeowner + homeowner * married_couple

designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

xGH = create_interaction(xG,xH)
designX = designX.join(xGH)

xGM = create_interaction(xG,xM)
designX = designX.join(xGM)

xHM = create_interaction(xH,xM)
designX = designX.join(xHM)

designX = stats.add_constant(designX, prepend=True)
LLK_2HM, DF_2HM, fullParams_2HM = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2HM - LLK_2GM)
testDF = DF_2HM - DF_2GM
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
import numpy

print('Log likelihood',LLK_2HM)
print('No. of free parameters', DF_2HM)
print("The significance is: 1.4597001212103711e-295 ")
Imp6 = numpy.log10(testPValue)
print("Importance :", -Imp6)
 
Q1 - a

designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

xGH = create_interaction(xG,xH)
designX = designX.join(xGH)

xGM = create_interaction(xG,xM)
designX = designX.join(xGM)

xHM = create_interaction(xH,xM)
designX = designX.join(xHM)

designX = stats.add_constant(designX, prepend=True)

reduce_form,inds = sympy.Matrix(designX.values).rref()
print("The aliased parameters found in model are \n")
for i in range(0,len(designX.columns)):
    if i not in inds:
        print(designX.columns[i])

Q2
fin= []
grp = [1,2,3,4]
home = [0,1]
mar = [0,1]

for i in grp:
    for j in home:
        for k in mar:
            fin.append([i,j,k])

df = pandas.DataFrame(fin, columns=['group_size','homeowner','married_couple'])

df_grp = pandas.get_dummies(df[['group_size']].astype('category'))
X = df_grp

df_home = pandas.get_dummies(df[['homeowner']].astype('category'))
X = X.join(df_home)

df_mar = pandas.get_dummies(df[['married_couple']].astype('category'))
X = X.join(df_mar)

df_grp_home = create_interaction(df_grp, df_home)
df_grp_home = pandas.get_dummies(df_grp_home)
X = X.join(df_grp_home)

df_grp_mar = create_interaction(df_grp, df_mar)
df_grp_mar = pandas.get_dummies(df_grp_mar)
X = X.join(df_grp_mar)

df_home_mar = create_interaction(df_home, df_mar)
df_home_mar = pandas.get_dummies(df_home_mar)
X = X.join(df_home_mar)


X = stats.add_constant(X, prepend=True)


model = stats.MNLogit(y, designX)
Fit = model.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)           
prob_insurance = Fit.predict(X)
print(prob_insurance)
prob = pandas.DataFrame((prob_insurance[1]/prob_insurance[0])).max()
print(prob)
v11 = float(prob_insurance.loc[[8]][0])
v12 = float(prob_insurance.loc[[8]][2])
ans = v12/v11
print(ans)

v21 = float(prob_insurance.loc[[0]][0])
v22 = float(prob_insurance.loc[[0]][2])
ans2 = v22/v21
print(ans2)

Ratio = ans/ans2
print(Ratio)
v80 = float(prob_insurance.loc[[2]][0])
v81 = float(prob_insurance.loc[[2]][1])
v82 = float(prob_insurance.loc[[2]][2])
num1 = v80/v82
den1 = v81/v82
ans1 = num1/den1

v00 = float(prob_insurance.loc[[0]][0])
v01 = float(prob_insurance.loc[[0]][1])
v02 = float(prob_insurance.loc[[0]][2])
num2 = v00/v02
den2 = v01/v02
ans2 = num2/den2

ans = ans1/ans2
print(ans1)
print(ans2)
print(ans)

#print("Required Odds ratio for part g is " + str(ans))


Q3 - a
import pandas as pd
import numpy as np
hmeq = pd.read_csv("Purchase_Likelihood.csv", delimiter = ",")
print("Count of group insurance")
count = hmeq.groupby('insurance').size()
print(count)
print("Class probabilities of group insurance")
prob = count/ len(hmeq)
print(prob)

Q3b-d
def rowwithcolumn(rowvar, columnvar, show = 'ROW'):
    counttable = pd.crosstab(index = rowvar, columns = columnvar, margins = False, dropna = True)
    print("frequency table :\n",counttable)
    print()
    
    if (show == 'ROW' or show == 'BOTH'):
        rowFraction = counttable.div(counttable.sum(1), axis = 'index')
        print("Row fraction table:\n", rowFraction)
        print()
        
    if (show == 'COLUMN' or show == 'BOTH'):
        columnFraction = counttable.div(counttable.sum(0), axis = 'columns')
        print("Row fraction table:\n", columnFraction)
        print()
        
    return
rowwithcolumn(hmeq['group_size'],hmeq['insurance'])
rowwithcolumn(hmeq['homeowner'],hmeq['insurance'])
rowwithcolumn(hmeq['married_couple'],hmeq['insurance'])

Q3 - e
def cramerV(xCat, yCat):
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    chiSqStat = ((obsCount - expCount)**2/expCount).to_numpy().sum()
    cramerV = chiSqStat / nTotal
    if(cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)
    return(cramerV)

cramer_group_size = cramerV(hmeq['group_size'], hmeq['insurance'])
cramer_homeowner = cramerV(hmeq['homeowner'], hmeq['insurance'])
cramer_married_couple = cramerV(hmeq['married_couple'], hmeq['insurance'])
print(cramer_group_size)
print(cramer_homeowner)
print(cramer_married_couple)

Q3 - f

count = hmeq.groupby('insurance').count()['group_size']
prop = count/ hmeq.shape[0]

def probab(x):
    probab0 = ((pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'][0] / pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'].sum()) * 
                   (pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False)[x[0]][0] / pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False).loc[[0]].sum(axis=1)[0]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False)[x[1]][0] / pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False).loc[[0]].sum(axis=1)[0]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False)[x[2]][0] / pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False).loc[[0]].sum(axis=1)[0]))
    probab1 = ((pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'][1] / pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'].sum()) * 
                   (pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False)[x[0]][1] / pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False).loc[[1]].sum(axis=1)[1]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False)[x[1]][1] / pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False).loc[[1]].sum(axis=1)[1]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False)[x[2]][1] / pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False).loc[[1]].sum(axis=1)[1]))
    probab2 = prob2 = ((pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'][2] / pd.DataFrame({'insurance': count.index, 'Count': count.values,'Class Probability': prop.values})['Count'].sum()) * 
                   (pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False)[x[0]][2] / pd.crosstab(hmeq.insurance, hmeq.group_size, margins = False, dropna = False).loc[[2]].sum(axis=1)[2]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False)[x[1]][2] / pd.crosstab(hmeq.insurance, hmeq.homeowner, margins = False, dropna = False).loc[[2]].sum(axis=1)[2]) * 
                   (pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False)[x[2]][2] / pd.crosstab(hmeq.insurance, hmeq.married_couple, margins = False, dropna = False).loc[[2]].sum(axis=1)[2]))
    probabs_sum = probab0 + probab1 + probab2
    probabfin0 = probab0 / probabs_sum
    probabfin1 = probab1 / probabs_sum
    probabfin2 = probab2 / probabs_sum
    return [probabfin0, probabfin1, probabfin2]
uni_group = sorted(list(hmeq.group_size.unique()))
uni_home = sorted(list(hmeq.homeowner.unique()))
uni_mar = sorted(list(hmeq.married_couple.unique()))
import itertools
 
c = list(itertools.product(uni_group, uni_home, uni_mar))
n = []
for i in c:
    t = [probab(i)]
    n.extend(t)
    
	
pred=pd.DataFrame(n,columns=['insurance=0','insurance=1','insurance=2'])

Test=pd.DataFrame();
g=[]
h=[]
m=[]
for i in range(1,5):
	for j in range(2):
		for k in range(2):
			g.append(i)
			h.append(j)
			m.append(k)

Test['group_size']=g
Test['homeowner']=h
Test['married_couple']=m

Test=pd.concat([Test,pred],axis=1)
print('Predicted Probability:\n',Test )

Q3 - g
m=[]
for i in range(len(n)):
    temp=n[i][1]/n[i][0]
    m.append([temp])    
print(np.array(m).max())

np.array(m).max()
c[np.where(m == np.array(m).max())[0][0]]