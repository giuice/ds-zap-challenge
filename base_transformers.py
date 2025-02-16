import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

def _define_variables(variables):
    '''
    checks that variable names are passed in a list
    '''
    
    if not variables or isinstance(variables, list):
       variables = variables
    else:
       variables = [variables]
    return variables


class BaseCategoricalTransformer(BaseEstimator, TransformerMixin):
    '''
    Finds categorical variables in the train set, or checks that variables indicated
    by the user are categorical.
    '''
    
    def fit(self, X, y = None):
        
        if not self.variables:
            # select all categorical variables
            self.variables = [var for var in X.columns if X[var].dtypes==['O','category']]
        else:
            # variables indicated by user
            if len(X[self.variables].select_dtypes(exclude=['O','category']).columns) != 0:
#            for var in self.variables:
#                if X[var].dtypes != 'O':
                raise TypeError("variable {} is not of type object, check that all indicated variables are of type object before calling the transformer")
            self.variables = self.variables
        
        return self.variables



class BaseCategoricalEncoder(BaseCategoricalTransformer):  

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
            X[feature] = X[feature].map(self.encoder_dict_[feature], na_action='ignore')
            
            if X[feature].isnull().sum() > 0:
                X[feature] = X[feature].fillna(0)
                warnings.warn("NaN values were introduced by the encoder due to labels in variable {} not present in the training set. Try using the RareLabelCategoricalEncoder.".format(feature) )       
        return X

class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    '''
    Finds numerical variables in the train set, or checks that variables indicated
    by the user are numerical.
    '''
    
    def fit(self, X, y = None):
        
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not self.variables:
            # select all numerical variables
            #self.variables = list(X.select_dtypes(include=numerics).columns)  
            self.variables = list(X.select_dtypes(include='number').columns)
        else:
            # variables indicated by user
            #if len(X[self.variables].select_dtypes(exclude=numerics).columns) != 0:
            if len(X[self.variables].select_dtypes(exclude='number').columns) != 0:
               raise ValueError("Some of the selected variables are not numerical. Please cast them as numerical before calling the imputer")
            
            self.variables = self.variables
