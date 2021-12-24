'''
Udacity machine learning devops project 1
Encapsulate a model fitting pipeline in production worthy script

author: Jeremy Smith
date: 12/21/2021
'''

from numpy.core.records import array
from pandas.core.frame import DataFrame, Series
from pandas.api.types import is_numeric_dtype
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from constants import *
import logging
from pathlib import Path

logging.basicConfig(
    filename='./logging.log'
)

def import_data(pth:str):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: str - a path to the csv
    output:
        customer_churn: df - pandas dataframe containing X,y for customer churn
    '''	
    try:
        customer_churn = pd.read_csv(pth).drop(columns=['Unnamed: 0'])
        customer_churn['Churn'] = customer_churn['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        customer_churn = customer_churn.drop(columns=['CLIENTNUM','Attrition_Flag'])
    except FileNotFoundError:
        logging.error('ERROR:%s, does not exist; check path...'%pth)
        raise FileNotFoundError
    return customer_churn


def perform_eda(customer_churn:DataFrame, impth:str):
    '''
    perform eda on df and save figures to images folder

    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        impth:str - location of folder to save image in 
    output:
        None
    '''
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError
    try:
        for col in customer_churn.columns:
            if is_numeric_dtype(customer_churn[col]):
                plot_histogram(customer_churn, col, impth)    
                plot_distogram(customer_churn, col, impth)
            else:
                plot_bar(customer_churn, col, impth)
        
        plot_heatmap(customer_churn, impth)
    except AssertionError:
        logging.error('ERROR: Plots failed to render, check dataframe is well formed!')
        raise AssertionError
    pass

def plot_bar(customer_churn:DataFrame, col:str, impth:str):
    '''
    plot and save a bar chart of value counts of a column in a dataframe

    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        col:str - one of the columns in customer_churn, chosen for plotting
        impth:str - location of folder to save image in 
    output:
        None
    '''
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError
    try:
        assert not customer_churn[col].empty
    except AssertionError:
        logging.error('ERROR: No data for plotting!')
        raise AssertionError
    try:
        assert len(customer_churn[col].unique()) < 10
    except AssertionError:
        logging.error('ERROR: Categorical column contains many categories, should it be cast to numeric type?')
        raise AssertionError
    plt.figure(figsize=(20,10)) 
    customer_churn[col].value_counts('normalize').plot(kind='bar')
    plt.savefig(impth + '%s_bar.png'%col)
    plt.close();
    pass

def plot_histogram(customer_churn:DataFrame, col:str, impth:str):
    '''
    plot and save a histogram of a column in a dataframe

    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        col:str - one of the columns in customer_churn, chosen for plotting
        impth:str - location of folder to save image in 
    output:
        None
    '''
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError
    try:
        assert not customer_churn[col].empty
    except AssertionError:
        logging.error('ERROR: No data for plotting!')
        raise AssertionError
    plt.figure(figsize=(20,10)) 
    customer_churn[col].hist()
    plt.savefig(impth + '%s_histogram.png'%col)
    plt.close();
    pass

def plot_distogram(customer_churn:DataFrame, col:str, impth:str):
    '''
    plot and save a distogram of a column in a dataframe
    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        col:str - one of the columns in customer_churn, chosen for plotting
        impth:str - location of folder to save image in 
    output:
        None
    '''
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError
    try:
        assert not customer_churn[col].empty
    except AssertionError:
        logging.error('ERROR: No data for plotting!')
        raise AssertionError
    plt.figure(figsize=(20,10))  
    sns.distplot(customer_churn[col]);
    plt.savefig(impth + '%s_distogram.png'%col)
    plt.close();
    pass

def plot_heatmap(customer_churn:DataFrame, impth:str):
    '''
    plot and save a histogram of a column in a dataframe
    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        col:str - one of the columns in customer_churn, chosen for plotting
        impth:str - location of folder to save image in
    output:
        None 
    '''
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError
    try:
        assert not customer_churn.empty
    except AssertionError:
        logging.error('ERROR: No data for plotting!')
        raise AssertionError
    plt.figure(figsize=(20,10))  
    sns.heatmap(customer_churn.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(impth + 'correlation_heatmap.png')
    plt.close();
    pass

def filter_columns(customer_churn:DataFrame, category_lst:list, quant_lst:list, response:str):
    '''
    split dataframe into valid inputs and outputs for fitting

    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        category_lst: list - a list of column names for categorical variables
        qunat_lst: list - a list of column names for quantitative variables
        response: str - name of column containg y
    output:
        X: DataFrame - filtered and encoded input variables
        y: Series - reponse variable series
    '''
    try:
        X = customer_churn[quant_lst+category_lst]
        X = encoder_helper(customer_churn, category_lst)
    except KeyError:
        logging.error('ERROR: Some of columns defined in constants are not in data!')
        raise KeyError
    try:
        y = customer_churn[response]
    except KeyError:
        logging.error('ERROR: %s, response column, not in data!'%response)
        raise KeyError
    X = X.drop(columns=response)
    return X, y

def encoder_helper(customer_churn:DataFrame, category_lst:list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        one_hot_variables = pd.get_dummies(customer_churn[category_lst], drop_first=True)
    except KeyError:
        logging.error('ERROR: Some of columns defined in constants are not in data!')
        raise KeyError
    encoded_customer_churn = customer_churn.drop(columns=category_lst).join(one_hot_variables)
    return encoded_customer_churn


def perform_feature_engineering(customer_churn:DataFrame, category_lst:list, quant_lst:list, response:str):
    '''
    input:
        customer_churn: df - pandas dataframe containing X,y for customer churn
        category_lst: list - a list of column names for categorical variables
        qunat_lst: list - a list of column names for quantitative variables
        response: str - name of column containg y

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    try:
        X, y = filter_columns(customer_churn, category_lst, quant_lst, response)
    except KeyError:
        logging.error('ERROR: Could not filter data and engineer columns!')
        raise KeyError
    try:        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    except ValueError:
        # Means that dataset passed was too small probably
        logging.error('ERROR: Could not split data, with input shapes of X:%s,%s y:%s'%(X.shape[0],X.shape[1],len(y)))
        raise ValueError
    return X_train, X_test, y_train, y_test

def plot_classification_report(report_str_test:str, report_str_train:str, report_name:str, impth:str):
    '''
    given plotting DataFrame, generate a heatmap plot, and save
    
    input:
        classification_report_df: DataFrame - output of sklearn classification report, converted to dataframe
        full_impth:str - path to save classification report image 
    output:
        None
    '''
    
    try:
        assert len(report_str_test) > 0
    except AssertionError:
        logging.error('ERROR: Empty test string ' + report_str_test)
        raise AssertionError
    try:
        assert len(report_str_train) > 0
    except AssertionError:
        logging.error('ERROR: Empty train string ' + report_str_train)
        raise AssertionError
    try:
        assert len(report_name) > 0
    except AssertionError:
        logging.error('ERROR: Empty name string ' + report_name)
        raise AssertionError
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('%s Train'%report_name), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(report_str_train), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('%s Test'%report_name), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(report_str_test), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(impth+'%s_class_report.png'%report_name)
    plt.close;


def classification_report_image(y_train:Series,
                                y_test:Series,
                                y_train_preds_lr:array,
                                y_train_preds_rf:array,
                                y_test_preds_lr:array,
                                y_test_preds_rf:array,
                                impth:str):
    '''
    produces classification report for training and testing results and stores report as image
    input:
            y_train: Series - training response values
            y_test: Series - test response values
            y_train_preds_lr: array - training predictions from logistic regression
            y_train_preds_rf: array - training predictions from random forest
            y_test_preds_lr: array - test predictions from logistic regression
            y_test_preds_rf: array - test predictions from random forest
            impth: str - location of folder to save image in

    output:
             None
    '''
    
    try:
        assert not y_train.empty or y_test.empty
    except AssertionError:
        logging.error('ERROR: Empty y reference data passed to classification report')
    try:
        assert len(y_train_preds_lr) == len(y_train) and len(y_test_preds_lr) == len(y_test)
    except AssertionError:
        logging.error('ERROR: Length of reference data does not match length of prediction data for logistic regression!')
    try:
        assert len(y_train_preds_rf) == len(y_train) and len(y_test_preds_rf) == len(y_test)
    except AssertionError:
        logging.error('ERROR: Length of reference data does not match length of prediction data for random forest!')

    report_str_rf_test = classification_report(y_test, y_test_preds_rf)
    report_str_rf_train = classification_report(y_train, y_train_preds_rf)
    report_str_lr_test = classification_report(y_test, y_test_preds_lr)
    report_str_lr_train = classification_report(y_train, y_train_preds_lr)

    plot_classification_report(report_str_rf_test, report_str_rf_train, 'RandomForest', impth)
    plot_classification_report(report_str_lr_test, report_str_lr_train, 'LogisticRegression', impth)

    pass

def roc_curve_plot(lrc:LogisticRegression, rfc:RandomForestClassifier, X_test:DataFrame, y_test:Series, impth:str):
    '''
    given trained logistic regressor and random forest classifier, make roc plot
    
    input:
        lrc: LogisticRegression - tranined logistic regressor
        rfc: RandomForestClassifier - trained random forest classifier
        X_test: DataFrame - test predictors
        y_test: Series - test labels
        impth:str - location of folder to save image in 
    output:
        None
    '''

    try:
        check_is_fitted(lrc)
    except NotFittedError:
        logging.error('ERROR: Logistic regressor is not fitted!')
    try:
        check_is_fitted(rfc)
    except NotFittedError:
        logging.error('ERROR: Random forest is not fitted!')
    try:
        assert len(X_test) == len(y_test)
    except AssertionError:
        logging.error('ERROR: X_train and y_train are different length for roc plot!')
        raise AssertionError
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(impth + 'roc_curve.png')
    plt.close();


def feature_importance_plot(rfc:RandomForestClassifier, X_data:DataFrame, impth:str):
    '''
    creates and stores the feature importances in pth
    input:
        rfc: RandomForestClassifier - trained random forest classifier
        X_test: DataFrame - test predictors
        impth:str - location of folder to save image in 

    output:
             None
    '''

    try:
        check_is_fitted(rfc)
    except NotFittedError:
        logging.error('ERROR: Random forest is not fitted!')
        raise NotFittedError
    try:
        assert not X_data.empty
    except AssertionError:
        logging.error('ERROR: Empty X_data passed to feature importance!')
        raise AssertionError
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError

    # Calculate feature importances
    importances = rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(impth + 'feature_importance.png')
    plt.close();

def explainer_plot(rfc:RandomForestClassifier, X_test:DataFrame, impth:str):
    '''
    generate feature importance plot using shap

    inputs:
        rfc: RandomForestClassifier - trained random forest classifier
        X_test: DataFrame - test predictors
        impth:str - location of folder to save image in 
    outputs:
        None
    '''

    try:
        check_is_fitted(rfc)
    except NotFittedError:
        logging.error('ERROR: Random forest is not fitted!')
        raise NotFittedError
    try:
        assert not X_test.empty
    except AssertionError:
        logging.error('ERROR: Empty X_data passed to feature importance!')
        raise AssertionError
    try:
        rfc.predict(X_test)
    except ValueError:
        logging.error('ERROR: X_test is malformed for fitting with random forest classifier!')
        raise ValueError
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError

    plt.figure(figsize=(8, 12))
    explainer = shap.TreeExplainer(rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(impth + 'shap_plot.png')
    plt.close();

def train_models(X_train:DataFrame, X_test:DataFrame, y_train:DataFrame, y_test:DataFrame, param_grid:dict, impth:str, modelpth:str):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              param_grid: dict - specifies grid search parameters
              impth:str - location of folder to save image in
              modelpth:str - location of folder to save models in
    output:
              None
    '''

    try:
        assert len(X_train) == len(y_train)
    except AssertionError:
        logging.error('ERROR: X_train and y_train have different lengths, cannot train models!')
        raise AssertionError
    try:
        assert len(X_test) == len(y_test)
    except AssertionError:
        logging.error('ERROR: X_test and y_test have different lengths, cannot test models!')
        raise AssertionError
    try:
        assert len(param_grid) > 0
    except AssertionError:
        logging.error('ERROR: Empty parameter grid passed to grid search!')
        raise AssertionError
    try:
        assert Path(modelpth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s model save path does not exist!'%modelpth)
        raise AssertionError
    try:
        assert Path(impth).is_dir()
    except AssertionError:
        logging.error('ERROR: %s image save path does not exist!'%impth)
        raise AssertionError

    # Fit the models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    
    # Save trained models
    joblib.dump(cv_rfc, modelpth + 'rfc_model.pkl')
    joblib.dump(lrc, modelpth + 'logistic_model.pkl')

    # Make predictions, to evaluate performance
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate plots to summarize performance
    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, impth)
    roc_curve_plot(lrc, cv_rfc, X_test, y_test, impth)
    feature_importance_plot(cv_rfc, X_train.append(X_test), impth)
    explainer_plot(cv_rfc, X_train.append(X_test), impth)

    pass

if __name__ == '__main__':

    save_test_data = False

    customer_churn = import_data(path_to_data)
    perform_eda(customer_churn, image_save_path)
    X_train, X_test, y_train, y_test = perform_feature_engineering(customer_churn, cat_columns, quant_columns, resp_col)

    if save_test_data:
        X_train.to_csv('./data/X_train.csv')
        X_test.to_csv('./data/X_test.csv')
        y_train.to_csv('./data/y_train.csv')
        y_test.to_csv('./data/y_test.csv')
    train_models(X_train, X_test, y_train, y_test, param_grid, image_save_path, model_save_path)
