'''
Tests for churn_library.py

author: Jeremy Smith
date: 12/24/2021
'''
from churn_library import *
from constants import *
from pathlib import Path

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert customer_churn.shape[0] > 0
        assert customer_churn.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        expected_columns = cat_columns + quant_columns + [resp_col]
        for col in expected_columns:
            assert col in customer_churn.columns
        logging.info("Testing import_data: All expected columns are present")
    except AssertionError as err:
        logging.error("Testing import_data: Columns do not match expected columns in constants.oy")
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
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: Data file or image save directory not found")
        raise err
    except AssertionError as err:
        logging.error("Testing perform_eda: One or more plots failed to render, check input data")
        raise err

    # Assert expected images were created
    image_pth = Path(image_save_path)
    try:
        num_images = len(list(image_pth.glob('*.png')))
        assert num_images > 0
        logging.info("Testing perform_eda: At least one image was created for eda")
    except AssertionError as err:
        logging.error("Testing perform_eda: No images created")
        raise err

    # Delete all images again
    delete_images()

def test_filter_encode_columns():
    '''
    test column type casting and encoding
    '''
    try:
        customer_churn = import_data(path_to_data)
        X, y = filter_encode_columns(customer_churn, cat_columns, quant_columns, resp_col)
        assert not X.empty or y.empty
        logging.info('Testing filter_encode_columns: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing filter_encode_columns: Data file not found")
        raise err
    except KeyError as err:
        logging.error("Testing filter_encode_columns: Some columns are missing, check input data and columns")
        raise err
    except AssertionError as err:
        logging.error("Testing filter_encode_columns: Filtered data appears empty, check inputs")
        raise err
    
def test_feature_engineering():
    '''
    test feature engineering of input data
    '''
    # Delete existing intermediate data, so we know we made it here
    delete_test_data()

    try:
        customer_churn = import_data(path_to_data)
        X_train, X_test, y_train, y_test = perform_feature_engineering(customer_churn, cat_columns, quant_columns, resp_col)

        # Save the data, so as not to repeat below
        X_train.to_csv('./data/X_train.csv')
        X_test.to_csv('./data/X_test.csv')
        y_train.to_csv('./data/y_train.csv')
        y_test.to_csv('./data/y_test.csv')

        assert not X_train.empty or y_train.empty or X_test.empty or y_test.empty
        logging.info('Testing feature_engineering: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing feature_engineering: Data file not found")
        raise err
    except KeyError as err:
        logging.error("Testing feature_engineering: Check filter_encode function")
        raise err
    except ValueError as err:
        logging.error("Testing feature_engineering: Input data is extremely small, check inputs")
        raise err
    except AssertionError as err:
        logging.error("Testing feature_engineering: Filtered data appears empty, check inputs")
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
        train_models(X_train, X_test, y_train, y_test, param_grid, image_save_path, model_save_path)
        logging.info('Testing test_train_model: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing test_train_model: Check if feature_engineering completed and model, image directory exist")
        raise err
    except AssertionError as err:
        logging.error("Testing test_train_model: Could not fit model, check input data shapes and parameters")
        raise err

    try:
        # Assert expected models were created
        mod_pth = Path(model_save_path)
        num_models = len(list(mod_pth.glob('*.pkl')))
        assert num_models == 2 
        logging.info("Testing test_train_model: Models were saved successfully")
    except AssertionError as err:
        logging.error("Testing test_train_model: Expected 2 model files to be saved, found %s"%num_models)
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
        logging.info('Testing classification_report_image: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing classification_report_image: Data file not found, check feature_engineering")
        raise err
    except AssertionError as err:
        logging.error("Testing classification_report_image: Images were not created as expected")

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
        logging.info('Testing roc_curve_plot: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing roc_curve_plot: Data files not found, check feature_engineering and model training")
        raise err
    except AssertionError as err:
        logging.error("Testing roc_curve_plot: Images were not created as expected")

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
        logging.info('Testing feature_importance_plot: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing feature_importance_plot: Data files not found, check feature_engineering and model training")
        raise err
    except AssertionError as err:
        logging.error("Testing feature_importance_plot: Images were not created as expected")
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
        logging.info('Testing feature_explainer_plot: SUCCESS')
    except FileNotFoundError as err:
        logging.error("Testing feature_explainer_plot: Data files not found, check feature_engineering and model training")
        raise err
    except AssertionError as err:
        logging.error("Testing feature_explainer_plot: Images were not created as expected")
        raise err

    # Clean up intermediate data
    delete_images()
    delete_models()
    delete_test_data()