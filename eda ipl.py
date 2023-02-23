# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:07:39 2022

@author: kanik
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#importing data
df=pd.read_csv("ipl3.csv")
df["Set Name"].unique()
df["New Franchise"].unique()\
df["Country"].unique()
df["New Franchise"].unique()
df["Batting"].unique()
df["Bowling"].unique()
#fixing data entry error
df["New Franchise"]= df["New Franchise"].replace(["Gujarat Titan","Luknow Super Giants"],["Gujarat Titans","Lucknow Super Giants"])
df.drop("State Association",inplace=True,axis=1)
df.value_counts()

df.isnull().sum()
df.head()
df.shape
df.info()
y= df.describe()

#star variable
df['Star Variable'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)

#correlation
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

df.columns.values
8
df.boxplot(by ='Team', column =['Sold Price (in CR)'], grid = False, fontsize=10)
df.boxplot(by = "Team", column=["Runs"], grid= False)
df.boxplot(by ='Team', column =['IPL(Matches Played)'], grid = False, fontsize=4)


ax = sns.boxplot(x="Team", y="IPL(Matches Played)", data=df)
ax = sns.boxplot(x="Team", y="Sold Price (in CR)", data=df)
ax = sns.boxplot(x="Team", y="Runs", data=df)
ax = sns.boxplot(x="Team", y="Strike Rate", data=df)
ax = sns.boxplot(x="Team", y="Economy", data=df)
ax = sns.boxplot(x="Team", y="Wickets", data=df)

plt.ylim(-15, 240)
plt.show()
sns.boxplot("IPL(Matches Played)", data=df)
sns.boxplot('Sold Price (in CR)',data=df)


#plt.subplot(200,10)
sns.distplot(df["Sold Price (in CR)"],kde=True)
y1=np.log(df["Sold Price (in CR)"])
sns.distplot(y1,kde=True)
sns.distplot(df["IPL(Matches Played)"],kde=False)
sns.distplot(df["Runs"],kde=False)
sns.distplot(df["Age"],kde=False)
sns.distplot(df["Base Price"],kde=False)

pd.crosstab(df["Team"],df['Star Variable'],margins=True)
sns.countplot('Team',hue='Star Variable',data=df)
plt.show()

#sns.factorplot('Team','Sold Price (in CR)',hue="Star Variable",data=df)
#plt.show()

f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Team","Sold Price (in CR)", hue="Star Variable", data=df,split=True)
sns.violinplot("Team","Age", hue="Star Variable", data=df,split=True)
sns.violinplot("Team","IPL(Matches Played)", hue="Star Variable", data=df,split=True)

df.groupby('Team')['Sold Price (in CR)'].count()
df.groupby('Team')['Age'].mean() 
df.groupby('Team')['Star Variable'].mean()

sns.countplot('Team',hue='Star Variable',data=df)

sns.scatterplot(x="IPL(Matches Played)",y="Sold Price (in CR)", data=df)
plt.plot
sns.regplot(x="IPL(Matches Played)",y="Sold Price (in CR)", data=df)
sns.regplot(x="Age",y="Sold Price (in CR)", data=df)
sns.regplot(x="Base Price",y="Sold Price (in CR)", data=df)
sns.regplot(x="Runs",y="Sold Price (in CR)", data=df)
sns.regplot(x="Strike Rate",y="Sold Price (in CR)", data=df)
sns.regplot(x="Wickets",y="Sold Price (in CR)", data=df)
sns.regplot(x="Economy",y="Sold Price (in CR)", data=df)

sns.barplot(x="Team", y="Sold Price (in CR)", data=df, hue="Star Variable")#, estimator=sum)

sns.scatterplot(x="Base Price", y="Sold Price (in CR)",data=df)
sns.scatterplot(x="Age", y="Sold Price (in CR)",data=df)

df2=pd.read_csv("Book1 country wise.csv")

print("Frequency % of each State\n",df["Set Name"].value_counts()/len(df))
fe = df.groupby('Set Name').size()/len(df)
df.loc[:,'Set Name'] = df['Set Name'].map(fe)
print(df)

df["Specialism"].unique()
df["Country"].unique()
df["Bowling"].unique()
#encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Education'.
df['2021 Team']= label_encoder.fit_transform(df['2021 Team'])
print(df['Specialism'].unique())
print(df['Specialism'].unique())

print(df["Country"].value_counts())
country_dict = {'India':1,'Australia':2,'South Africa':3,'West Indies':4,'New Zealand':5,'England':6,'Sri Lanka':7,'Bangladesh':8,'Afghanistan':9}
df['Country'] = df["Country"].map(country_dict)
print(df[['Country']])

df["Bowling"]= df["Bowling"].replace(["-"],[0])

from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Education'.
df['New Franchise']= label_encoder.fit_transform(df['New Franchise'])
print(df['Education'].unique())
print(df['Education_label_encoder'].unique())

df.drop("State Association", inplace=True, axis=1)
df.drop("Bid", inplace=True, axis=1)
print(df["Batting"].value_counts())
print(df["Bowling"].value_counts())
df["Batting"].replace(('RHB','LHB'),(1,0),inplace=True)
df["C/U/A"].replace(('Capped','Uncapped'),(1,0),inplace=True)

print("Frequency % of each State\n",df["Bowling"].value_counts()/len(df))
fe1 = df.groupby('Bowling').size()/len(df)
df.loc[:,'Bowling'] = df['Bowling'].map(fe1)
print(df)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Education'.
df['New Franchise']= label_encoder.fit_transform(df['New Franchise'])


df.to_csv("ipl.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df['Set Name','Country','Age','Specialism','Batting','Bowling']))
data_num = pd.DataFrame(df[['Set Name','Country','Age','Specialism','Batting','Bowling']])  

data_num = pd.DataFrame(df[['Set No.','Set Name','Country','Age','Specialism','Batting','Bowling','IPL(Matches Played)','2021 Team','C/U/A','Base Price','Sold Price (in CR)','New Franchise','Runs','Strike Rate','Wickets','Economy']])  
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(data_num))
df1.columns=['Set No.','Set Name','Country','Age','Specialism','Batting','Bowling','IPL(Matches Played)','2021 Team','C/U/A','Base Price','Sold Price (in CR)','New Franchise','Runs','Strike Rate','Wickets','Economy']
df3=df.append(df1,ignore_index=True)
#['Set Name','Country','Age','Specialism','Batting','Bowling','IPL(Matches Played)','C/U/A','Base Price','Sold Price (in CR)','Runs','Strike Rate','Wickets','Economy','Star Variable']
ez=df["Player"]
df1.insert(2,"Player",ez)
df1.drop("Player",inplace=True,axis=1)

ez=df["Star Variable"]
df1.insert(19,'Star Variable',ez)

df1.to_csv("ipl0404.csv")



df1.columns.values
#checking linearity

sns.regplot(x="IPL(Matches Played)",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Set No.",y="Sold Price (in CR)", data=df1) #odd
sns.regplot(x="Set Name",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Country",y="Sold Price (in CR)", data=df1) #odd
sns.regplot(x="Age",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Base Price",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Runs",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Strike Rate",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Wickets",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Economy",y="Sold Price (in CR)", data=df1)
sns.regplot(x="Specialism",y="Sold Price (in CR)", data=df1)#
sns.regplot(x="Batting",y="Sold Price (in CR)", data=df1)#
sns.regplot(x="Bowling",y="Sold Price (in CR)", data=df1)#
sns.regplot(x="C/U/A",y="Sold Price (in CR)", data=df1)#
sns.regplot(x="New Franchise",y="Sold Price (in CR)", data=df1)#
sns.regplot(x="Star Variable",y="Sold Price (in CR)", data=df1)

sns.distplot(df1["Sold Price (in CR)"],kde=True)
y1=np.log(df1["Sold Price (in CR)"])

df['Specialism'].unique()
df4=pd.get_dummies(df['Specialism'])
