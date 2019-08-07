# Reference
# 
import numpy as np
import pandas as pd
import warnings


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from base_transformers import BaseCategoricalEncoder,BaseCategoricalTransformer, _define_variables, BaseNumericalTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer

class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class DFRobustScaler(BaseNumericalTransformer):
    # RobustScaler but for pandas DataFrames

    def __init__(self, variables=None):
        self.variables = _define_variables(variables)
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        var = self.variables
        self.rs.fit(X[var])
        self.center_ = pd.Series(self.rs.center_, index=X[var].columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X[var].columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        #for feature in self.variables:
        var = self.variables
        X[var] = self.rs.transform(X[var])
            #Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return X


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz

class CountFrequencyCategoricalEncoder(BaseCategoricalEncoder):
    """ 
    The CountFrequencyCategoricalEncoder() replaces categories by the count of
    observations per category or by the percentage of observations per category.
    
    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.
    
    The CountFrequencyCategoricalEncoder() will encode only categorical variables
    (type 'object'). A list of variables can be passed as an argument. If no 
    variables are passed as argument, the encoder will only encode categorical
    variables (object type) and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The encoder then transforms the categories to those mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default='count'
        Desired method of encoding.
        'count': number of observations per category
        'frequency' : percentage of observations per category
    
    variables : list
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and transform all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {count / frequency: category} pairs used
        to replace categories for every variable.    
    """
    
    def __init__(self, encoding_method = 'count', variables = None):
        
        if encoding_method not in ['count', 'frequency']:
            raise ValueError("encoding_method takes only values 'count' and 'frequency'")
            
        self.encoding_method = encoding_method       
        self.variables = _define_variables(variables)
           

    def fit(self, X, y = None):
        """
        Learns the numbers that should be used to replace the categories in
        each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            if self.encoding_method == 'count':
                self.encoder_dict_[var] = X[var].value_counts().to_dict()
                
            elif self.encoding_method == 'frequency':
                n_obs = np.float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()   
        
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
        
        return self



class OrdinalCategoricalEncoder(BaseCategoricalEncoder):
    """ 
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers 
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.
    
    For the ordered ordinal encoding for example in the variable colour, if the
    mean of the target for blue, red and grey is 0.5, 0.8 and 0.1 respectively,
    blue is replaced by 1, red by 2 and grey by 0.
    
    For the arbitrary ordinal encoding the numbers will be assigned arbitrarily
    to the categories, on a first seen first served basis.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default='ordered' 
        Desired method of encoding.
        'ordered': the categories are numbered in ascending order according to
        the target mean per category.
        'arbitrary' : categories are numbered arbitrarily.
        
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {ordinal number: category} pairs used
        to replace categories for every variable.
        
    """    
    def __init__(self, encoding_method  = 'ordered', variables = None):
        
        if encoding_method not in ['ordered', 'arbitrary']:
            raise ValueError("encoding_method takes only values 'ordered' and 'arbitrary'")
            
        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)
           

    def fit(self, X, y=None):
        """ Learns the numbers that should be used to replace the labels in each
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : Target. Can be None if selecting encoding_method = 'arbitrary'. 
        Otherwise, needs to be passed when fitting the transformer.
       
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        if self.encoding_method == 'ordered':
            if y is None:
                raise ValueError('Please provide a target (y) for this encoding method')
                            
            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns)+['target']

        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            
            if self.encoding_method == 'ordered':
                t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index
                
            elif self.encoding_method == 'arbitrary':
                t = X[var].unique()
                
            self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)}
            
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
        
        return self



class MeanCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The MeanCategoricalEncoder() replaces categories by the mean of the target. 
    
    For example in the variable colour, if the mean of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------  
    
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {target mean: category} pairs used
        to replace categories for every variable
        
    """    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
           

    def fit(self, X, y):
        """
        Learns the numbers that should be used to replace the labels in each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : Target
       
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)

        if y is None:
            raise ValueError('Please provide a target (y) for this encoding method')
            
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
            
        return self
    def transform(self, X):
        """ Replaces categories with the estimated numbers.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features].
            The input samples.
        
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing categories replaced by numbers.
       """
        # Check that the method fit has been called
        check_is_fitted(self, ['encoder_dict_'])
        
        # Check that the input is of the same shape as the training set passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from train set used to fit the encoder')
        
        # encode labels     
        X = X.copy()
        for feature in self.variables:
            X[feature+'_mean'] = X[feature].map(self.encoder_dict_[feature], na_action='ignore')
            
            if X[feature+'_mean'].isnull().sum() > 0:
                X[feature+'_mean'] = X[feature+'_mean'].fillna(0)
                warnings.warn("NaN values were introduced by the encoder due to labels in variable {} not present in the training set. Try using the RareLabelCategoricalEncoder.".format(feature) )       
        return X
    

class MedianCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The MedianCategoricalEncoder() replaces categories by the mean of the target. 
    
    For example in the variable colour, if the medan of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------  
    
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {target mean: category} pairs used
        to replace categories for every variable
        
    """    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
           

    def fit(self, X, y):
        """
        Learns the numbers that should be used to replace the labels in each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : Target
       
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)

        if y is None:
            raise ValueError('Please provide a target (y) for this encoding method')
            
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            self.encoder_dict_[var] = temp.groupby(var)['target'].median().to_dict()

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
            
        return self

    def transform(self, X):
        """ Replaces categories with the estimated numbers.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features].
            The input samples.
        
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing categories replaced by numbers.
       """
        # Check that the method fit has been called
        check_is_fitted(self, ['encoder_dict_'])
        
        # Check that the input is of the same shape as the training set passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from train set used to fit the encoder')
        
        # encode labels     
        X = X.copy()
        for feature in self.variables:
            X[feature+'_median'] = X[feature].map(self.encoder_dict_[feature], na_action='ignore')
            
            if X[feature+'_median'].isnull().sum() > 0:
                X[feature+'_median'] = X[feature+'_median'].fillna(0)
                warnings.warn("NaN values were introduced by the encoder due to labels in variable {} not present in the training set. Try using the RareLabelCategoricalEncoder.".format(feature) )       
        return X

class RareLabelCategoricalEncoder(BaseCategoricalEncoder):
    """
    The RareLabelCategoricalEncoder() groups rare / infrequent categories in
    a new category called "Rare".
    
    For example in the variable colour, if the percentage of observations
    for the categories magenta, cyan and burgundy are < 5 %, all those
    categories will be replaced by the new label "Rare".
       
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first finds the frequent labels for each variable (fit).
    The encoder then groups the infrequent labels under the new label 'Rare'
    (transform).
    
    Parameters
    ----------
    
    tol: float, default=0.05
        the minimum frequency a label should have to be considered frequent
        and not be removed.
    n_categories: int, default=10
        the minimum number of categories a variable should have in order for 
        the encoder to find frequent labels. If the variable contains 
        less categories, all of them will be considered frequent.
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containg the frequent categories (that will be kept)
        for each variable. Categories not present in this list will be replaced
        by 'Rare'.
    """
  
    def __init__(self, tol = 0.05, n_categories = 10, variables = None):
        
        if tol <0 or tol >1 :
            raise ValueError("tol takes values between 0 and 1")
            
        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")
            
        self.tol = tol
        self.n_categories = n_categories
        self.variables = _define_variables(variables)
        

    def fit(self, X, y = None):
        """
        Learns the frequent categories for each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter. You can leave y as None, or pass it as an
            argument.
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            if len(X[var].unique()) > self.n_categories:
                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts() / np.float(len(X)))
                # non-rare labels:
                self.encoder_dict_[var] = t[t>=self.tol].index
            else:
                # if the total number of categories is smaller than the indicated
                # the encoder will consider all categories as frequent.
                self.encoder_dict_[var]=  X[var].unique()
        
        self.input_shape_ = X.shape
                   
        return self
    

    def transform(self, X):
        """
        Groups rare labels under separate group 'Rare'.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe where rare categories have been grouped.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_'])
            
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
            
        return X