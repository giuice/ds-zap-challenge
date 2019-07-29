import json 
import pandas as pd 
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Funções de Exploração
## le os arquivo json e retorna um pandas dataframe
def readAndFlatDataSet(filename):
    data = [json.loads(line) for line in open(filename, 'r', encoding='utf8')]
    return json_normalize(data)

def showCatsNotInTest(train,test, column):
    arrtrain = train[column].value_counts().index
    arrtest = test[column].value_counts().index
    print('{0} dados não existem na base de treino.'.format(len(set(arrtest).difference(arrtrain))))

def plotPivotTableTop10(ds, target):
    for col in ds.columns:
        if ds[col].dtype == object:
            fig = plt.figure()
            ds.groupby(col)[target].mean().sort_values(ascending=False).head(10).plot()
            fig.set_xlabel(col)
            fig.set_ylabel('Media de ' + target)
            plt.xticks(rotation=90)
            plt.show()

##  retorna alguns status importantes no dataset
def getSomeStats(df, target=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    null_ratio = round((df.isnull().sum()/obs) * 100,3)
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if target is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'null ratio', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, null_ratio, uniques, skewness, kurtosis], axis = 1, sort=True)

    else:
        corr = df.corr()[target]
        str = pd.concat([types, counts, distincts, nulls, null_ratio, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + target
        cols = ['types', 'counts', 'distincts', 'nulls', 'null_ratio', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('ooooooooooooooooooooooooooo\nData types:\n',str.types.value_counts())
    print('ooooooooooooooooooooooooooo')
    return str

def nullsInfo(ds):
    print('total sem NA(s)', ds.dropna().shape[0])
    print('total de linhas no dataset: ',  ds.shape[0])
    print('percentagem sem missings: ',  ds.dropna().shape[0]/ np.float(ds.shape[0]))

# FUNÇÕES DE TRANSFORMAÇÃO
def createMonotonicFeature(train, test, feature, target):
    ordered_labels = train.groupby([feature])[target].mean().sort_values().index
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    train['{0}_ordered'.format(feature)] = train.Cabin.map(ordinal_label)
    train['{0}_ordered'.format(feature)] = test.Cabin.map(ordinal_label)


'''FUNÇÕES DE PLOT'''

def plotDsNullPercent(ds, plotsize=12):
    plt.ylabel = 'Porcentagem de NA''s'''
    ds.isnull().mean().sort_values(ascending=False).head(plotsize).plot.bar()

def plotBarCategories(ds, barLimit=30):
    for col in ds.columns:
        if ds[col].dtype == object:
            if ds[col].nunique() < barLimit:
                #plt.title('Categoria: ' + col)
                print(ds[col].value_counts())
                plt.xticks(rotation=90)
                sns.countplot(ds[col])
                plt.show()

def plotScatters(ds, targetvar='pricingInfos.price'):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    columns = list(ds.select_dtypes(include=numerics).columns)
    
    if targetvar in columns:
        columns.remove(targetvar)
    
    length = len(columns)
    counter = 0
    
    for i in range(0,length):
        if counter == length: break
        col = columns[counter]
        figure, axes = plt.subplots(1, 2,  figsize=(15,10))
        if ds[col].dtype == np.float64:
            if not col == targetvar:
                #ax = fig.add_subplot(1,2,1)
                plt.xticks(rotation=90)
                data = pd.concat([ ds[targetvar], ds[col]], axis=1)
                sns.scatterplot(data[col],data[targetvar] , ax=axes[0])
        counter += 1
        if counter == length: break
        col = columns[counter]
        if ds[col].dtype == np.float64:
            if not col == targetvar:
                #ax2 = fig.add_subplot(1,2,2)
                plt.xticks(rotation=90)
                data = pd.concat([ ds[targetvar], ds[col]], axis=1)
                sns.scatterplot(data[col],data[targetvar] , ax=axes[1])
        counter += 1
        if counter == length: break
        col = columns[counter]
        
                
def plotBoxCategories(ds, targetvar="pricingInfos.price"):
    for col in ds.columns:
        if ds[col].dtype == object:
            if ds[col].nunique() < 30:
                #plt.title('Categoria: ' + col)
                f, ax = plt.subplots(figsize=(8, 6))
                fig = sns.boxplot(x=col, y=targetvar, data=ds)
                plt.xticks(rotation=90)
                fig.set_yscale('log')
                #fig.set_xscale('log')
                #fig.axis(ymin=0,ymax=800000);
                plt.xticks(rotation=90)
                #ds[col].value_counts().plot.bar()
                #sns.boxplot(ds[col])
                plt.show()

def plotDistAndLog(var):
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax.set_title('Normal')
    ax2.set_title('log10+1')
    sns.distplot(var, ax=ax)
    sns.distplot(np.log10(var+1), ax=ax2)
    #sns.distplot(np.log10(var+1), ax=ax3)
    #sns.regplot(var, ax=ax2)
    #sns.kdeplot(np.log10(var), ax=ax2)

def plotRegressionPlot(ds, targetvar='pricingInfos.price'):
    for col in ds.columns:
        if ds[col].dtype == np.float64:
            if not col == targetvar:
                fig = plt.figure(figsize=(20,8))
                ax = fig.add_subplot(1,2,1)
                ax2 = fig.add_subplot(1,2,2)
                data = pd.concat([ ds[targetvar], ds[col]], axis=1)
                ax.set_title="Normalized"
                ax2.set_title = "log10"
                sns.regplot(data[col],data[targetvar], ax=ax)
                x = (ds[col] -ds[col].mean())/ds[col].std()
                y = np.log10(data[targetvar]+1)
                sns.regplot(x,y,ax=ax2)
                #data.plot.scatter(x=col, y=targetvar);
                plt.show()


def train_rf(X_train, y_train, X_test, y_test, columns):
    # function to train the random forest
    # and test it on train and test sets

    rf = RandomForestRegressor(n_estimators=800, random_state=39)

    if type(
            columns) == str:  # if we train using only 1 variable (pass a string instead of list in the "columns" argument of the function)
        rf.fit(X_train[columns].to_frame(), y_train.values)
        pred_train = rf.predict(X_train[columns].to_frame())
        pred_test = rf.predict(X_test[columns].to_frame())

    else:  # if we train using multiple variables (pass a list in the argument "columns")
        rf.fit(X_train[columns], y_train.values)
        pred_train = rf.predict(X_train[columns])
        pred_test = rf.predict(X_test[columns])

    print('Train set')
    print('Random Forest mse: {}'.format(mean_squared_error(y_train, pred_train)))
    print('Test set')
    print('Random Forest mse: {}'.format(mean_squared_error(y_test, pred_test)))

