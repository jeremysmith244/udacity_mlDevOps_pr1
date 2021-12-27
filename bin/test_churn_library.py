'''
Tests for churn_library.py

author: Jeremy Smith
date: 12/24/2021
'''

from constants import *
import pandas as pd
import joblib
from pathlib import Path
import logging

from churn_library import import_data, perform_eda, perform_feature_engineering, filter_encode_columns, train_models, classification_report_image, roc_curve_plot, feature_importance_plot, explainer_plot

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w')

LOGGER = logging.getLogger()

def delete_images():
    '''
    delete all images in image directory
    '''
    image_pth = Path(image_save_path)
    for im_file in image_pth.glob('*.png'):
        im_file.unlink()


def delete_models():
    '''
    delete all models in model directory
    '''
    model_pth = Path(model_save_path)
    for mod_file in model_pth.glob('*.pkl'):
        mod_file.unlink()


def delete_test_data():
    '''
    delete all intermediate test data in data directory
    '''
    data_pth = Path(path_to_data).parent
    for data_file in data_pth.glob('*.csv'):
        if data_file.name != Path(path_to_data).name:
            data_file.unlink()


def test_import_data():
    '''
    test data import function
    '''
    try:
        customer_churn = import_data(path_to_data)
        LOGGER.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        LOGGER.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert customer_churn.shape[0] > 0
        assert customer_churn.shape[1] > 0
    except AssertionError as err:
        LOGGER.error(
            "Testing import_data: Not enough rows and columns")
        raise err

    try:
        expected_columns = cat_columns + quant_columns + [resp_col]
        for col in expected_columns:
            assert col in customer_churn.columns
        LOGGER.info("Testing import_data: All expected columns are present")
    except AssertionError as err:
        LOGGER.error(
            "Testing import_data: Columns in data do not match constants")
        raise err


def test_perform_eda():
    '''
    test perform eda function
    '''
    # Delete all images, so we are sure we made them here
    delete_images()

    try:
        customer_churn = import_data(path_to_data)
        perform_eda(customer_churn, image_save_path)
        LOGGER.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        LOGGER.error(
            "Testing perform_eda: Data file or image save directory not found")
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing perform_eda: Plots failed to render, check input data")
        raise err

    # Assert expected images were created
    image_pth = Path(image_save_path)
    try:
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images > 0
        LOGGER.info(
            "Testing perform_eda: At least one image was created for eda")
    except AssertionError as err:
        LOGGER.error("Testing perform_eda: No images created")
        raise err

    # Delete all images again
    delete_images()


def test_filter_encode_columns():
    '''
    test column type casting and encoding
    '''
    try:
        customer_churn = import_data(path_to_data)
        X, y = filter_encode_columns(
            customer_churn, cat_columns, quant_columns, resp_col)
        assert not X.empty or y.empty
        LOGGER.info('Testing filter_encode_columns: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error("Testing filter_encode_columns: Data file not found")
        raise err
    except KeyError as err:
        LOGGER.error(
            "Testing filter_encode_columns: Columns are missing, check input")
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing filter_encode_columns: Filtered data empty, check inputs")
        raise err


def test_feature_engineering():
    '''
    test feature engineering of input data
    '''
    # Delete existing intermediate data, so we know we made it here
    delete_test_data()

    try:
        customer_churn = import_data(path_to_data)
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            customer_churn, cat_columns, quant_columns, resp_col)

        # Save the data, so as not to repeat below
        X_train.to_csv('./data/X_train.csv')
        X_test.to_csv('./data/X_test.csv')
        y_train.to_csv('./data/y_train.csv')
        y_test.to_csv('./data/y_test.csv')

        assert not X_train.empty or y_train.empty or X_test.empty or y_test.empty
        LOGGER.info('Testing feature_engineering: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error("Testing feature_engineering: Data file not found")
        raise err
    except KeyError as err:
        LOGGER.error(
            "Testing feature_engineering: Check filter_encode function")
        raise err
    except ValueError as err:
        LOGGER.error(
            "Testing feature_engineering: Input data too few rows to split")
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing feature_engineering: Filtered data appears empty")
        raise err


def test_train_models():
    '''
    test model training
    '''
    # Delete all models, so we are sure we made them here
    delete_models()

    try:
        # Load models for plotting roc curve
        X_train = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
        X_test = pd.read_csv('./data/X_test.csv', index_col='Unnamed: 0')
        y_train = pd.read_csv('./data/y_train.csv', index_col='Unnamed: 0')
        y_test = pd.read_csv('./data/y_test.csv', index_col='Unnamed: 0')
        train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            grid_search_parameters,
            image_save_path,
            model_save_path)
        LOGGER.info('Testing test_train_model: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error(
            '''Testing test_train_model: Check feature_engineering complete
               and image,model directory exists''')
        raise err
    except AssertionError as err:
        LOGGER.error(
            '''Testing test_train_model: Could not fit model,
               check input data shapes and parameters''')
        raise err

    try:
        # Assert expected models were created
        mod_pth = Path(model_save_path)
        num_models = len(list(mod_pth.glob('*.pkl')))
        assert num_models == 2
        LOGGER.info(
            "Testing test_train_model: Models were saved successfully")
    except AssertionError as err:
        LOGGER.error(
            "Testing test_train_model: Expected 2 model files saved, found %s",
            num_models)
        raise err


def test_classification_report_image():
    '''
    test classification report image
    '''
    # Delete all images, so we are sure we made them here
    delete_images()

    try:
        # Load dummy data
        y = pd.read_csv('./data/y_train.csv', index_col='Unnamed: 0')
        classification_report_image(y, y, y, y, y, y, image_save_path)

        # Assert expected images were created
        image_pth = Path(image_save_path)
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images == 2
        LOGGER.info('Testing classification_report_image: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error(
            '''Testing classification_report_image: Data file not found,
               check feature_engineering''')
        raise err
    except AssertionError as err:
        LOGGER.error(
            '''Testing classification_report_image:
               Images were not created as expected''')

    # Delete all images
    delete_images()


def test_roc_curve_plot():
    '''
    test roc curve plotting
    '''
    # Delete all images, so we are sure we made them here
    delete_images()

    try:
        # Load models for plotting roc curve
        rfc = joblib.load('./models/rfc_model.pkl')
        lrc = joblib.load('./models/logistic_model.pkl')
        X = pd.read_csv('./data/X_test.csv', index_col='Unnamed: 0')
        y = pd.read_csv('./data/y_test.csv', index_col='Unnamed: 0')
        roc_curve_plot(lrc, rfc, X, y, image_save_path)

        # Assert expected images were created
        image_pth = Path(image_save_path)
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images == 1
        LOGGER.info('Testing roc_curve_plot: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error(
            '''Testing roc_curve_plot: Data files not found,
               check feature_engineering and model training''')
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing roc_curve_plot: Images were not created as expected")

    # Delete all images
    delete_images()


def test_feature_importance_plot():
    '''
    test feature importance plot
    '''
    # Delete all images, so we are sure we made them here
    delete_images()

    try:
        # Load models for plotting roc curve
        rfc = joblib.load('./models/rfc_model.pkl')
        X = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
        feature_importance_plot(rfc, X, image_save_path)

        # Assert expected images were created
        image_pth = Path(image_save_path)
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images == 1
        LOGGER.info('Testing feature_importance_plot: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error(
            '''Testing feature_importance_plot: Data files not found,
               check feature_engineering and model training''')
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing feature_importance_plot: Images were not created")
        raise err

    # Delete all images
    delete_images()


def test_feature_explainer_plot():
    '''
    test feature explainer plot
    '''
    # Delete all images, so we are sure we made them here
    delete_images()

    try:
        # Load models for plotting roc curve
        rfc = joblib.load('./models/rfc_model.pkl')
        X = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
        explainer_plot(rfc, X, image_save_path)

        # Assert expected images were created
        image_pth = Path(image_save_path)
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images == 1
        LOGGER.info('Testing feature_explainer_plot: SUCCESS')
    except FileNotFoundError as err:
        LOGGER.error(
            '''Testing feature_explainer_plot: Data files not found,
               check feature_engineering and model training''')
        raise err
    except AssertionError as err:
        LOGGER.error(
            "Testing feature_explainer_plot: Images were not created")
        raise err

    # Clean up intermediate data
    delete_images()
    delete_models()
    delete_test_data()
