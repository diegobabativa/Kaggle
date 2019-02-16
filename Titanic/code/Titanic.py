import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

data_top= train.head()

#print(data_top)
#print("Size of data_top: ", data_top.size)
#print("Train shape (complete data)", train.shape)
#print("train describe  ********" )
#print(train.describe())
#print(train.describe(include=['O']))
#print(train.info())
#print(train.isnull().sum())
#print(test.head())
#print(test.isnull().sum())
survived= train[train['Survived']==1]
not_survived= train[train['Survived']==0]

#print("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
#print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
#print ("Total: %i"%len(train))

# PCClass Vs Survival
#print(train.Pclass.value_counts())

#train grup by Pclass
#print(train.groupby('Pclass').Survived.value_counts())

#probability to survival
#print(train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())

#Plot the information
#train.groupby('Pclass').Survived.mean().plot(kind='bar')
#sns.barplot(x='Pclass', y='Survived', data=train)

#quantity survivals by sex
#print(train.Sex.value_counts())
#print(train.groupby('Sex').Survived.value_counts())
#print(train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())
#Plot data
#sns.barplot(x='Sex', y='Survived', data=train)

'''
Pclass & Sex vs. Survival

Below, we just find out how many males and females are there
in each Pclass. We then plot a stacked bar diagram with that information. 
We found that there are more males among the 3rd Pclass passengers.
'''

#tab = pd.crosstab(train['Pclass'], train['Sex'])
#print(tab)


#tab.div(tab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
#plt.xlabel('Pclass')
#plt.ylabel('Percentage')

'''
Pclass, Sex & Embarked vs. Survival

Women from 1st and 2nd Pclass have almost 100% survival chance.
Men from 2nd and 3rd Pclass have only around 10% survival chance.
'''
#sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=4, data=train)


'''
Pclass, Sex & Embarked vs. Survival
'''
#sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)

'''
Embarked vs. Survived
'''
#print(train.Embarked.value_counts())
#print(train.groupby('Embarked').Survived.value_counts())
#print(train[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean()) 

# Plot the information
#train.groupby('Embarked').Survived.mean().plot(kind='bar')
#sns.barplot(x='Embarked', y='Survived', data=train)

'''
Parch Vs Survival
'''
#print(train.Parch.value_counts())
#print(train.groupby('Parch').Survived.value_counts())
#print(train[['Parch','Survived']].groupby(['Parch'], as_index=True).mean())

#Plot the Information
#train.groupby('Parch').Survived.mean().plot(kind='bar')
#sns.barplot(x='Parch', y='Survived', ci=None, data=train) # ci=None will hide the error bar

'''
SibSp vs. Survival
'''
#print(train.SibSp.value_counts())
#print(train.groupby('SibSp').Survived.value_counts())
#print(train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean())

#Plot the information
#train.groupby('SibSp').Survived.mean().plot(kind='bar')
#sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar

'''
Age Vs Survival
'''
#fig = plt.figure(figsize=(15,5))
#ax1 = fig.add_subplot(131)
#ax2 = fig.add_subplot(132)
#ax3 = fig.add_subplot(133)

#Hat diagrams

#sns.violinplot(x='Embarked', y='Age', hue='Survived', data=train, split=True, ax=ax1)
#sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
#sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)


'''
Combining Male anf Female by Age
'''

'''
total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived= train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived= train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived= train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived= train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
'''

'''
Feature Extraction
'''

train_test_data = [train, test] #Combining train and test data
for dataset in train_test_data:
    dataset['Title']= dataset.Name.str.extract('([A-Za-z]+)\.')

#print(train.head)
print(pd.crosstab(train['Title'], train['Sex']))


