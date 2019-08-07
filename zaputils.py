import json 
import pandas as pd 
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.core.display import display, HTML
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# Funções de Exploração
## le os arquivo json e retorna um pandas dataframe
def readAndFlatDataSet(filename):
    data = [json.loads(line) for line in open(filename, 'r', encoding='utf8')]
    return json_normalize(data)

def showCatsNotInTest(train,test, column):
    arrtrain = train[column].value_counts().index
    arrtest = test[column].value_counts().index
    print('{0} dados não existem na base de treino.'.format(len(set(arrtest).difference(arrtrain))))


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


def categorical_to_counts(df_train, df_test, arr_rows, target):
    # make a temporary copy of the original dataframes
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()
    
    for col in arr_rows:
        # make the dictionary mapping label to counts
        X_frequency_map = df_train_temp[col].value_counts().to_dict()
        
        # remap the labels to their counts
        df_train_temp[col+'count_target'] = df_train_temp[col].map(X_frequency_map)
        df_test_temp[col+'count_target'] = df_test_temp[col].map(X_frequency_map)
    
    
    return df_train_temp, df_test_temp



def impute_categorical_to_median(df_train, df_test, y, arr_rows, target):
    # make a temporary copy of the datasets 
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()
    df_train_temp[target] = y
    
    for col in arr_rows:
        # order the labels according to target mean
        #ordered_labels = df_train_temp.groupby([col])[target].median().sort_values().index
        ordered_labels = df_train_temp.groupby([col])[target].median().sort_values().index
        # create the dictionary to map the ordered labels to an ordinal number
        ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 
        # remap the categories  to these ordinal numbers
        df_train_temp[col+'median_target'] = df_train[col].map(ordinal_label)
        
        #df_test_temp[col].where(df_test_temp[col].isin(ordinal_label), 0, inplace=True)
        #df_test_temp[col+'new'] = df_test_temp[col].apply(lambda row: row if row in ordinal_label else 0 )
        df_test_temp[col+'median_target'] = df_test[col].map(ordinal_label, na_action='ignore')
        df_train_temp[col+'median_target'] = pd.to_numeric(df_train_temp[col+'median_target'])
        df_test_temp[col+'median_target'] = pd.to_numeric(df_test_temp[col+'median_target'].fillna(0))
    
    # remove the target
    df_train_temp.drop([target], axis=1, inplace=True)
    return df_train_temp, df_test_temp



def impute_na_random(df_train, df_test, variable):
    # add additional variable to indicate missingness
    df_train.loc[:, variable+'_NA'] = np.where(df_train[variable].isnull(), 1, 0)
    df_test.loc[:, variable+'_NA'] = np.where(df_test[variable].isnull(), 1, 0)
    
    # random sampling
    df_train.loc[:,variable+'_random'] = df_train[variable]
    df_test.loc[:,variable+'_random'] = df_test[variable]
    
    # extract the random sample to fill the na
    random_sample_train = df_train[variable].dropna().sample(df_train[variable].isnull().sum(), random_state=0).copy()
    random_sample_test = df_train[variable].dropna().sample(df_test[variable].isnull().sum(), random_state=0).copy()
    
    # pandas needs to have the same index in order to merge datasets
    random_sample_train.index = df_train[df_train[variable].isnull()].index
    random_sample_test.index = df_test[df_test[variable].isnull()].index
    
    df_train.loc[df_train[variable].isnull(), variable+'_random'] = random_sample_train
    df_test.loc[df_test[variable].isnull(), variable+'_random'] = random_sample_test
    del random_sample_train
    del random_sample_test


'''FUNÇÕES DE PLOT'''

def plotDistributions(train,test, var='price', show_test=True):
    traincounts = train[var].value_counts(normalize=True)
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    ax[0].set_title('Treino')
    ax[1].set_title('Teste')
    ax[0].scatter(traincounts.index, traincounts)
    if show_test:
        testcounts = test[var].value_counts(normalize=True)
        ax[1].scatter(testcounts.index, testcounts)
    plt.show()


def plotMeanPivotTableTop10(ds, target):
    for col in ds.columns:
        if ds[col].dtype == object:
            fig = plt.figure()
            ds.groupby(col)[target].mean().sort_values(ascending=False).head(10).plot()
            fig.set_xlabel(col)
            fig.set_ylabel('Media de ' + target)
            plt.xticks(rotation=90)
            plt.show()

def plotMedianPivotTableTop10(ds, target):
    for col in ds.columns:
        if ds[col].dtype == object or ds[col].dtype.name == 'category':
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(col)
            ax.set_ylabel('Media de ' + target)
            #ax.set_xticklabels(90)
            ds.groupby(col)[target].median().sort_values(ascending=False).head(10).plot.barh(ax=ax)
            plt.show()


def plotDsNullPercent(ds, plotsize=12):
    plt.ylabel = 'Porcentagem de NA''s'''
    ds.isnull().mean().sort_values(ascending=False).head(plotsize).plot.bar()

def plotBarCategories(train,test, barLimit=30):
    for col in train.columns:
        if train[col].dtype == object or train[col].dtype.name == 'category':
            if train[col].nunique() < barLimit:
                fig, ax = plt.subplots(1,2,figsize=(14,8))
                ax[0].set_title('Categoria: ' + col)
                #print(ds[col].value_counts())
                ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=90) 
                tr = sns.countplot(train[col],ax=ax[0], orient='v')
                #tr.set_xticklabels(rotation=45)
                #plt.xticks(rotation='vertical')
                #ds[col].value_counts().plot.bar()
                ax[1].set_title('Test Categoria: ' + col)
                ts =sns.countplot(test[col],ax=ax[1], orient='v')
                ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=90)
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
    ax = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax.set_title('Normal')
    ax2.set_title('johnsonsu')
    ax3.set_title('log normal')
    sns.distplot(var, ax=ax, fit=stats.norm)
    sns.distplot(var, ax=ax2, fit=stats.johnsonsu)
    sns.distplot(var, ax=ax3, fit=stats.lognorm)
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
                ax2.set_ylim(0,1)
                sns.regplot(x,y,ax=ax2)
                #data.plot.scatter(x=col, y=targetvar);
                plt.show()

def hook(var):
    return var

def QQ_plot(df, variable, func=hook, **kwargs):
    title1 = kwargs.get("title1")
    title2 = kwargs.get("title2")
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(title1)
    df2 = func(df[variable])
    #df2.hist(bins=30)
    sns.distplot(df2,fit=stats.norm, ax=ax)
    ax2 = fig.add_subplot(1, 2, 2)
    stats.probplot(df2, dist="norm", plot=ax2)
    if title2:
        ax2.set_title(title2)
    plt.show()
    del df2

def removeOutliers(ds, columns, ratio=3):
    df = ds.copy()
    for col in columns:
        if df[col].dtype == np.number:
            x = df[col]
            if np.issubdtype(x.dtype, np.number):
                q75 = x.quantile(0.75)
                IQR = q75 - x.quantile(0.25)
                Upper_fence = q75 + (IQR * ratio)
                Bottom_fence = q75 - (IQR * ratio)
                df = df[(x < Upper_fence) & (x > Bottom_fence)]
    return df

def plotBoxandHist(data, columns, func=hook):
    datatemp = data.copy()
    for var in columns:
        datatemp[var] = func(datatemp[var])
        plt.figure(figsize=(15,6))
        plt.subplot(1, 2, 1)
        fig = datatemp.boxplot(column=var)
        fig.set_title('')
        fig.set_ylabel(var)

        plt.subplot(1, 2, 2)
        fig = datatemp[var].hist(bins=20)
        fig.set_ylabel('Numero de anuncios')
        fig.set_xlabel(var)

        plt.show()
    del datatemp

### MISSING VALUES FUNCTIONS
def fillGeolocationWithStreetMode(ds):
    ends = []
    for index, row in ds[ds['address.geoLocation.location.lat'].isnull()][['address.neighborhood',
                                                                       'address.street']].iterrows():
        ends.append([row.name, row[0], row[1]])
    for arr in ends:
        query = (ds['address.neighborhood'] == arr[1]) & \
        (ds['address.street'] == arr[2]) & (ds['address.geoLocation.location.lat'].notnull())
        ret = ds[query]
        if len(ret):
            ds['address.geoLocation.location.lat'].iloc[arr[0]] = ret['address.geoLocation.location.lat'].mode().values[0]
            ds['address.geoLocation.location.lon'].iloc[arr[0]] =ret['address.geoLocation.location.lon'].mode().values[0]
            ds['address.geoLocation.precision'].iloc[arr[0]] =ret['address.geoLocation.precision'].mode().values[0]
        else: #se não achou pela rua, vai pelo bairro
            query = (ds['address.neighborhood'] == arr[1]) & (ds['address.geoLocation.location.lat'].notnull())
            ret = ds[query]
            if len(ret):
                ds['address.geoLocation.location.lat'].iloc[arr[0]] = ret['address.geoLocation.location.lat'].mode().values[0]
                ds['address.geoLocation.location.lon'].iloc[arr[0]] =ret['address.geoLocation.location.lon'].mode().values[0]
                ds['address.geoLocation.precision'].iloc[arr[0]] =ret['address.geoLocation.precision'].mode().values[0]

def impute_na(ds, variable, median):
    ds[variable+'_median'] = ds[variable].fillna(median)
    ds[variable+'_zero'] = ds[variable].fillna(0) 
    
def fillNas(train,test):
    isNullNumericCols = train.loc[:,train.isna().any()].select_dtypes(include=np.number).columns
    for col in isNullNumericCols:
        median = train[col].median()
        impute_na(train,col,median)
        #a moda de teste deve ser passada para teste
        impute_na(test,col,median)

def printNullRatio(ds):
    for var in ds.columns:
        if ds[var].isnull().sum()>0:
            print(var, ds[var].isnull().mean())

def getCategoricalVars(ds):
    categorical = [var for var in ds.columns if ds[var].dtype=='O' or ds[var].dtype.name == "category"]
    print('Há {} variáveis categoricas'.format(len(categorical)))
    return categorical

def getBoolVars(ds):
    booleans = [var for var in ds.columns if ds[var].dtype == np.bool]
    print('Há {} variáveis booleans'.format(len(booleans)))
    return booleans

def getNumericalVars(ds):
    num_vars =['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical = [var for var in ds.columns if ds[var].dtype in num_vars ]
    print('Há {} variáveis numéricas'.format(len(numerical)))
    return numerical

def median_Encode(train,test,col, target):
    median = train.groupby(col)[target].median()
    train[col+'median_target'] = train[col].map(median)
    test[col+'median_target'] = test[col].map(median)

def removeOutliers(ds, columns, ratio=3):
    df = ds.copy()
    for col in columns:
        x = df[col]
        if np.issubdtype(x.dtype, np.number):
            IQR = x.quantile(0.75) - x.quantile(0.25)
            Upper_fence = x.quantile(0.75) + (IQR * ratio)
            Bottom_fence = x.quantile(0.75) - (IQR * ratio)
            df = df[(x < Upper_fence) & (x > Bottom_fence)]
    return df

def train_rf(X_train, y_train, X_test, y_test, columns):
   
    rf = RandomForestRegressor(n_estimators=800, random_state=39)

    if type(columns) == str:  # if we train using only 1 variable (pass a string instead of list in the "columns" argument of the function)
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

def runSimpleRegression(df, var, func = hook):
    newdf = df[[var,'price']].copy()
    newdf[var] = func(newdf[var])
    lmod = smf.ols(formula='price ~  {0}'.format(var), data=newdf).fit()
    display(lmod.summary())
    del newdf

def runMultiRegression(df, form):
    lmod = smf.ols(formula=form, data=df).fit()
    display(lmod.summary())

def plotMetricsImportance(X, y):
    from sklearn.feature_selection import SelectFromModel

    columns = list(X.columns)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(n_estimators=400, random_state=0)
    rf.fit(X_train, y_train.values)
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = columns,
                                    columns=['rf_importance']).sort_values('rf_importance',ascending=False)
    
    Y_ = rf.predict(X_test)
    
    
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
            'learning_rate': 0.01, 'loss': 'ls','random_state':0}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    Ygb_ = clf.predict(X_test)
    plt.figure(figsize=(14,10))
    ax1 = sns.distplot(y_test, hist=False, color="r", label="Valores Atuais")
    sns.distplot(Y_, hist=False, color="b", label="Valores Inferidos RF" , ax=ax1)
    sns.distplot(Ygb_, hist=False, color="g", label="Valores Inferidos GBM" , ax=ax1)

    rfmetrics = pd.DataFrame([
                            mean_absolute_error(y_test,Y_),
                            mean_squared_error(y_test,Y_),
                            r2_score(y_test, Y_),
                            mean_absolute_error(y_test,Ygb_),
                            mean_squared_error(y_test, Ygb_),
                            r2_score(y_test,Ygb_)],
        index=["RF-MAE","RF-MSE","RF-R2","GBM-MAE","GBM-MSE","GBM-R2"],
        columns=['Valores'])
    display(rfmetrics)
    display(feature_importances)
    display(pd.DataFrame(clf.feature_importances_,
                                   index = columns,
                                    columns=['rf_importance']).sort_values('rf_importance',ascending=False))
    #feature_select.plot.barh()
    #feature_importances.plot.barh()

def applyBoxCox(features):
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index
    bc_alphas = {}
    for i in skew_index:
        alpha = boxcox_normmax(features[i]+1)
        bc_alphas[i] = alpha
        features[i] = boxcox1p(features[i], alpha)
    return features, bc_alphas

def featureImportanceLasso(X,y):
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    imp_coef = coef.sort_values()
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance usando Lasso Model")

def runGBMRegressor(X,y):
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
            'learning_rate': 0.01, 'loss': 'ls','random_state':0}
    clf = ensemble.GradientBoostingRegressor(**params)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    print("R2: %.4f" % r2_score(y_test,clf.predict(X_test)))
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def RMSLE(y, y_pred):
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    print("RMSLE: % 4f" % (sum(terms_to_sum) * (1.0/len(y))) ** 0.5)