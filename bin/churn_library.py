'''
Notes
author:
date:
'''

from pandas.core.frame import DataFrame
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



def import_data(pth:str):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: str - a path to the csv
    output:
            customer_churn: df - pandas dataframe containing X,y for customer churn
    '''	
    customer_churn = pd.read_csv(r"../data/bank_data.csv").drop(columns=['Unnamed: 0'])
    customer_churn['Churn'] = customer_churn['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return customer_churn


def perform_eda(customer_churn:DataFrame):
    '''
    perform eda on df and save figures to images folder
    input:
            customer_churn: df - pandas dataframe containing X,y for customer churn

    output:
            None
    '''

    for col in customer_churn.col:
        plot_histogram(customer_churn, col)    
        plot_bar(customer_churn, col)
        plot_distogram(customer_churn, col)
    
    plot_heatmap(customer_churn)

def plot_bar(customer_churn:DataFrame, col:str, impth:str='../images/'):
    '''
    plot and save a bar chart of value counts of a column in a dataframe
    input:
    customer_churn: df - pandas dataframe containing X,y for customer churn
    col:str - one of the columns in customer_churn, chosen for plotting
    impth:str - location of folder to save image in 
    '''
    plt.figure(figsize=(20,10)) 
    customer_churn[col].value_counts('normalize').plot(kind='bar')
    plt.savefig(impth + '%s_bar.png'%col);

def plot_histogram(customer_churn:DataFrame, col:str, impth:str='../images/'):
    '''
    plot and save a histogram of a column in a dataframe
    input:
            customer_churn: df - pandas dataframe containing X,y for customer churn
            col:str - one of the columns in customer_churn, chosen for plotting
            impth:str - location of folder to save image in 
    '''
    plt.figure(figsize=(20,10)) 
    customer_churn[col].hist()
    plt.savefig(impth + '%s_histogram.png'%col);

def plot_distogram(customer_churn:DataFrame, col:str, impth:str='../images/'):
    '''
    plot and save a histogram of a column in a dataframe
    input:
            customer_churn: df - pandas dataframe containing X,y for customer churn
            col:str - one of the columns in customer_churn, chosen for plotting
            impth:str - location of folder to save image in 
    '''
    plt.figure(figsize=(20,10))  
    sns.distplot(customer_churn[col]);
    plt.savefig(impth + '%s_distogram.png'%col);

def plot_heatmap(customer_churn:DataFrame, impth:str='../images/'):
        '''
        plot and save a histogram of a column in a dataframe
        input:
                customer_churn: df - pandas dataframe containing X,y for customer churn
                col:str - one of the columns in customer_churn, chosen for plotting
                impth:str - location of folder to save image in 
        '''
        plt.figure(figsize=(20,10))  
        sns.heatmap(customer_churn.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig(impth + 'correlation_heatmap.png');

def encoder_helper(df, category_lst, response):
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
    # gender encoded column
    gender_lst = []
    gender_groups = df.groupby('Gender').mean()['Churn']

    for val in df['Gender']:
        gender_lst.append(gender_groups.loc[val])

    df['Gender_Churn'] = gender_lst    

    #education encoded column
    edu_lst = []
    edu_groups = df.groupby('Education_Level').mean()['Churn']

    for val in df['Education_Level']:
    edu_lst.append(edu_groups.loc[val])

    df['Education_Level_Churn'] = edu_lst

    #marital encoded column
    marital_lst = []
    marital_groups = df.groupby('Marital_Status').mean()['Churn']

    for val in df['Marital_Status']:
    marital_lst.append(marital_groups.loc[val])

    df['Marital_Status_Churn'] = marital_lst

    #income encoded column
    income_lst = []
    income_groups = df.groupby('Income_Category').mean()['Churn']

    for val in df['Income_Category']:
    income_lst.append(income_groups.loc[val])

    df['Income_Category_Churn'] = income_lst

    #card encoded column
    card_lst = []
    card_groups = df.groupby('Card_Category').mean()['Churn']
    
    for val in df['Card_Category']:
    card_lst.append(card_groups.loc[val])

    df['Card_Category_Churn'] = card_lst
        
        


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass