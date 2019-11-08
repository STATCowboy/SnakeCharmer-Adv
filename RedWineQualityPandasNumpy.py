import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import OS and set CWD
import os
# cd = os.getcwd()
cd = "c:/Users/jj/OneDrive/documents/SQL Server/SQL Summit 2019/Python Presentation - Advanced/"

# Pandas
#

# Load the Wine Data using Pandas
winePD = pd.read_csv(cd+"/winequality-red.csv", sep=';')
print(type(winePD))

# Print info on wine
print(winePD.info())

# First rows of wind
winePD.head()

# Last rows of wine
winePD.tail()

# Take a sample of 5 rows
winePD.sample(5)

# Describe wine Dataset
winePD.describe()

# Remove null values (don't have any nulls)
print(pd.isnull(winePD))
print(pd.notnull(winePD))

winePD.dropna()


# Extract one column from Panda Dataframe by name
winePD[['fixed acidity']]

# Extract from row 2 until end using indexes (indexes start at 0)
# eg. dataframe[row,column]
winePD.iloc[1:,]

# Extract column 1 and 2 (indexes start at 0)
winePD.iloc[:,[0,1]]
winePD[['fixed acidity', 'volatile acidity']]

# Extract the first 10 rows and first 2 columns 
# (Start is always included, End always excluded, so we have to add one more to the row and column)
winePD.iloc[0:10, 0:2]

# Melt/gather columns into rows
winePD.melt()

# Pivot
winePD.groupby('quality').mean()
pd.pivot_table(winePD, columns='quality', aggfunc="mean")

# Bring in white wine data to concat to red
winePDwhite = pd.read_csv(cd+"/winequality-white.csv", sep=';')

# Concat for Union of Dataframes
pd.concat([winePD, winePDwhite])

# Concat for Join Columns (for demo we use subset of columns from same dataframe)
pd.concat([winePD.iloc[:,0:4], winePD.iloc[:,4:11]], axis=1)

# Merge (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)
df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                  'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                  'value': [5, 6, 7, 8]})

df1.merge(df2, left_on='lkey', right_on='rkey')

winePD.merge(winePDwhite, how='outer')


# Plot histogram and scatter plot
winePD[['fixed acidity']].plot.hist()
winePD[['alcohol']].plot.hist()
winePD.plot.scatter(x='fixed acidity', y='volatile acidity')

# Correlation plots
corr = winePD.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

sns.pairplot(winePD.iloc[:,0:4])
plt.show()

g = sns.PairGrid(winePD.iloc[:,0:4], diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
plt.show()


# NumPy
# 

# Load the Wine Data using NumPy
wineNP = np.loadtxt(cd+"winequality-red.csv", delimiter=';', skiprows=1)
print(type(wineNP))

# Slice the wine data into X (independent variable) and y (dependent variable)
X = wineNP[:, 0:11]
Y = wineNP[:, 11]

# Create NumPy Arrays from Dataframe
X = np.array(winePD.drop(['quality'],axis=1))
Y = np.array(winePD['quality'])

# Create Panda Dataframe from NumPy ndarray
df = pd.DataFrame.from_records(wineNP)
df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Print Histogram in NumPy Array
print(np.histogram(winePD.alcohol, bins=[7,8,9,10,11,12,13,14,15]))




# Create Pandas Dataframe from Dictionary
wineDict = {
    'fixed acidity':[7.4, 7.8, 7.8, 11.2, 7.4]
    ,'volatile acidity':[0.7, 0.88, 0.76, 0.28, 0.7]
}

wineDFfromDict = pd.DataFrame(wineDict, columns = ['fixed acidity', 'volatile acidity'])
