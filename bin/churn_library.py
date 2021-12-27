'''
Udacity machine learning devops project 1
Encapsulate a model fitting pipeline in production worthy script

author: Jeremy Smith
date: 12/21/2021
'''

from pathlib import Path
from constants import *
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy.core.records import array
from pandas.core.frame import DataFrame, Series
from pandas.api.types import is_numeric_dtype
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth: str):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: str - a path to the csv
    output:
        customer_churn: df - pandas dataframe containing X,y for customer churn
    '''
    try:
        customer_churn = pd.read_csv(pth).drop(columns=['Unnamed: 0'])
        customer_churn['Churn'] = customer_churn['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        customer_churn = customer_churn.drop(
            columns=['CLIENTNUM', 'Attrition_Flag'])
    except FileNotFoundError as err:
        print('ERROR:%s, does not exist; check path...', pth)
        raise err
    return customer_churn


def perform_eda(customer_churn: DataFrame, impth: str):
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
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise FileNotFoundError from err
    try:
        for col in customer_churn.columns:
            if is_numeric_dtype(customer_churn[col]):
                plot_histogram(customer_churn, col, impth)
                plot_distogram(customer_churn, col, impth)
            else:
                plot_bar(customer_churn, col, impth)

        plot_heatmap(customer_churn, impth)
    except AssertionError as err:
        print(
            'ERROR: Plots failed to render, check dataframe is well formed!')
        raise err
    pass


def plot_bar(customer_churn: DataFrame, col: str, impth: str):
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
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err
    try:
        assert not customer_churn[col].empty
    except AssertionError as err:
        print('ERROR: No data for plotting!')
        raise err
    try:
        assert len(customer_churn[col].unique()) < 10
    except AssertionError as err:
        print(
            'ERROR: Categorical column contains too many categories, max 10')
        raise err
    plt.figure(figsize=(20, 10))
    customer_churn[col].value_counts('normalize').plot(kind='bar')
    plt.savefig(impth + '%s_bar.png' % col)
    plt.close();


def plot_histogram(customer_churn: DataFrame, col: str, impth: str):
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
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err
    try:
        assert not customer_churn[col].empty
    except AssertionError as err:
        print('ERROR: No data for plotting!')
        raise err
    plt.figure(figsize=(20, 10))
    customer_churn[col].hist()
    plt.savefig(impth + '%s_histogram.png' % col)
    plt.close();


def plot_distogram(customer_churn: DataFrame, col: str, impth: str):
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
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err
    try:
        assert not customer_churn[col].empty
    except AssertionError as err:
        print('ERROR: No data for plotting!')
        raise err
    plt.figure(figsize=(20, 10))
    sns.distplot(customer_churn[col])
    plt.savefig(impth + '%s_distogram.png' % col)
    plt.close();


def plot_heatmap(customer_churn: DataFrame, impth: str):
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
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err
    try:
        assert not customer_churn.empty
    except AssertionError as err:
        print('ERROR: No data for plotting!')
        raise err
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        customer_churn.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(impth + 'correlation_heatmap.png')
    plt.close();


def filter_encode_columns(
        customer_churn: DataFrame,
        category_lst: list,
        quant_lst: list,
        response: str):
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
        X = customer_churn[quant_lst + category_lst]
        X = encoder_helper(customer_churn, category_lst)
    except KeyError as err:
        print(
            'ERROR: Some of columns defined in constants are not in data!')
        raise err
    try:
        y = customer_churn[response]
    except KeyError as err:
        print('ERROR: %s, response column, not in data!', response)
        raise err
    X = X.drop(columns=response)
    return X, y


def encoder_helper(customer_churn: DataFrame, category_lst: list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        one_hot_variables = pd.get_dummies(
            customer_churn[category_lst], drop_first=True)
    except KeyError as err:
        print(
            'ERROR: Some of columns defined in constants are not in data!')
        raise err
    encoded_customer_churn = customer_churn.drop(
        columns=category_lst).join(one_hot_variables)
    return encoded_customer_churn


def perform_feature_engineering(
        customer_churn: DataFrame,
        category_lst: list,
        quant_lst: list,
        response: str):
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
        X, y = filter_encode_columns(
            customer_churn, category_lst, quant_lst, response)
    except KeyError as err:
        print('ERROR: Could not filter data and engineer columns!')
        raise err
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    except ValueError as err:
        # Means that dataset passed was too small probably
        print(
            'ERROR: Could not split data, with input shapes of X:%s,%s y:%s' %
            (X.shape[0], X.shape[1], len(y)))
        raise err
    return X_train, X_test, y_train, y_test


def plot_classification_report(
        report_str_test: str,
        report_str_train: str,
        report_name: str,
        impth: str):
    '''
    given plotting DataFrame, generate a heatmap plot, and save

    input:
        report_str_test: str - output of sklearn class report on test data
        report_str_train: str - output of sklearn class report on train data
        report_name: str - name to be applied to report as a label
        impth:str - path to save classification report image
    output:
        None
    '''

    try:
        assert len(report_str_test) > 0
    except AssertionError as err:
        print('ERROR: Empty test string %s', report_str_test)
        raise err
    try:
        assert len(report_str_train) > 0
    except AssertionError as err:
        print('ERROR: Empty train string %s', report_str_train)
        raise err
    try:
        assert len(report_name) > 0
    except AssertionError as err:
        print('ERROR: Empty name string %s', report_name)
        raise err
    try:
        assert Path(impth).is_dir()
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err

    plt.figure(figsize=(5, 5))
    plt.text(
        0.01, 1.25, str(
            '%s Train' %
            report_name), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(report_str_train), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.6, str(
            '%s Test' %
            report_name), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(report_str_test), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(impth + '%s_class_report.png' % report_name)
    plt.close();


def classification_report_image(y_train: Series,
                                y_test: Series,
                                y_train_preds_lr: array,
                                y_train_preds_rf: array,
                                y_test_preds_lr: array,
                                y_test_preds_rf: array,
                                impth: str):
    '''
    classification report based on results and stores report as image
    input:
            y_train: Series - training response values
            y_test: Series - test response values
            y_train_preds_lr: array - training predictions, logistic regression
            y_train_preds_rf: array - training predictions, random forest
            y_test_preds_lr: array - test predictions from logistic regression
            y_test_preds_rf: array - test predictions from random forest
            impth: str - location of folder to save image in

    output:
             None
    '''

    try:
        assert not y_train.empty or y_test.empty
    except AssertionError as err:
        print(
            'ERROR: Empty y reference data passed to classification report')
        raise err
    try:
        assert len(y_train_preds_lr) == len(
            y_train) and len(y_test_preds_lr) == len(y_test)
    except AssertionError as err:
        print(
            'ERROR: Length of ref data does not match prediction data for logistic regression!')
        raise err
    try:
        assert len(y_train_preds_rf) == len(
            y_train) and len(y_test_preds_rf) == len(y_test)
    except AssertionError as err:
        print(
            'ERROR: Length of ref data does not match prediction data for random forest!')
        raise err
    report_str_rf_test = classification_report(y_test, y_test_preds_rf)
    report_str_rf_train = classification_report(y_train, y_train_preds_rf)
    report_str_lr_test = classification_report(y_test, y_test_preds_lr)
    report_str_lr_train = classification_report(y_train, y_train_preds_lr)

    plot_classification_report(
        report_str_rf_test,
        report_str_rf_train,
        'RandomForest',
        impth)
    plot_classification_report(
        report_str_lr_test,
        report_str_lr_train,
        'LogisticRegression',
        impth)


def roc_curve_plot(
        lrc: LogisticRegression,
        rfc: RandomForestClassifier,
        X_test: DataFrame,
        y_test: Series,
        impth: str):
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
    except NotFittedError as err:
        print('ERROR: Logistic regressor is not fitted!')
        raise err
    try:
        check_is_fitted(rfc)
    except NotFittedError as err:
        print('ERROR: Random forest is not fitted!')
        raise err
    try:
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        print(
            'ERROR: X_train and y_train are different length for roc plot!')
        raise err
    try:
        assert Path(impth).is_dir()
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise FileNotFoundError from err

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(impth + 'roc_curve.png')
    plt.close();


def feature_importance_plot(
        rfc: RandomForestClassifier,
        X_data: DataFrame,
        impth: str):
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
    except NotFittedError as err:
        print('ERROR: Random forest is not fitted!')
        raise err
    try:
        assert not X_data.empty
    except AssertionError as err:
        print('ERROR: Empty X_data passed to feature importance!')
        raise err
    try:
        assert Path(impth).is_dir()
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err

    # Calculate feature importances
    importances = rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(impth + 'feature_importance.png')
    plt.close();


def explainer_plot(rfc: RandomForestClassifier, X_test: DataFrame, impth: str):
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
    except NotFittedError as err:
        print('ERROR: Random forest is not fitted!')
        raise err
    try:
        assert not X_test.empty
    except AssertionError as err:
        print('ERROR: Empty X_data passed to feature importance!')
        raise err
    try:
        rfc.predict(X_test)
    except ValueError as err:
        print(
            'ERROR: X_test is malformed for random forest classifier!')
        raise err
    try:
        assert Path(impth).is_dir()
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise err

    plt.figure(figsize=(8, 12))
    explainer = shap.TreeExplainer(rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(impth + 'shap_plot.png')
    plt.close();


def train_models(
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: DataFrame,
        y_test: DataFrame,
        param_grid: dict,
        impth: str,
        modelpth: str):
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
    except AssertionError as err:
        print('ERROR: X_train and y_train have different lengths, cannot train!')
        raise err
    try:
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        print('ERROR: X_test and y_test have different lengths, cannot test!')
        raise err
    try:
        assert len(param_grid) > 0
    except AssertionError as err:
        print('ERROR: Empty parameter grid passed to grid search!')
        raise err
    try:
        assert Path(modelpth).is_dir()
    except AssertionError as err:
        print('ERROR: %s model save path does not exist!', modelpth)
        raise FileNotFoundError from err
    try:
        assert Path(impth).is_dir()
    except AssertionError as err:
        print('ERROR: %s image save path does not exist!', impth)
        raise FileNotFoundError from err

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
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        impth)
    roc_curve_plot(lrc, cv_rfc, X_test, y_test, impth)
    feature_importance_plot(cv_rfc, X_train.append(X_test), impth)
    explainer_plot(cv_rfc, X_train.append(X_test), impth)


if __name__ == '__main__':

    churn_dataframe = import_data(path_to_data)
    perform_eda(churn_dataframe, image_save_path)
    train_x, test_x, train_y, test_y = perform_feature_engineering(
        churn_dataframe, cat_columns, quant_columns, resp_col)
    train_models(
        train_x,
        test_x,
        train_y,
        test_y,
        grid_search_parameters,
        image_save_path,
        model_save_path)
