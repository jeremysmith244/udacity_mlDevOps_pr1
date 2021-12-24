'''

'''
from churn_library import *
from constants import *
from pathlib import Path

def delete_images():
    image_pth = Path(image_save_path)
    for im_file in image_pth.glob('*'):
        im_file.unlink()

def delete_models():
    model_pth = Path(model_save_path)
    for mod_file in model_pth.glob('*'):
        mod_file.unlink()

def test_import_data():
    customer_churn = import_data(path_to_data)
    expected_columns = cat_columns + quant_columns + [resp_col]
    for col in expected_columns:
        assert col in customer_churn.columns

def test_perform_eda():

    # Delete all images, so we are sure we made them here
    delete_images()

    # Load all data, will throw assertion errors if individual image plot fails    
    customer_churn = import_data(path_to_data)
    perform_eda(customer_churn, image_save_path)

    # Assert expected images were created
    image_pth = Path(image_save_path)
    num_images = len(list(image_pth.glob('*.png')))
    assert num_images == 36 # This is total amount of images expected, need update with code change

def test_filter_columns():
    customer_churn = import_data(path_to_data)
    X, y = filter_columns(customer_churn, cat_columns, quant_columns, resp_col)
    assert not X.empty or y.empty

def test_feature_engineering():
    customer_churn = import_data(path_to_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(customer_churn, cat_columns, quant_columns, resp_col)
    assert not X_train.empty or y_train.empty or X_test.empty or y_test.empty

def test_plot_classification_report():

    # Delete all images, so we are sure we made them here
    delete_images()

    # Make the test report
    plot_classification_report('report_test', 'report_train', 'test_report', image_save_path)

    # Assert expected images were created
    image_pth = Path(image_save_path)
    num_images = len(list(image_pth.glob('*.png')))
    assert num_images == 1

def test_classification_report_image():

    # Delete all images, so we are sure we made them here
    delete_images()

    # Load dummy data
    y = pd.read_csv('./data/y_train.csv', index_col='Unnamed: 0')

    # Make plot
    classification_report_image(y, y, y, y, y, y, image_save_path)

    # Assert expected images were created
    image_pth = Path(image_save_path)
    num_images = len(list(image_pth.glob('*.png')))
    assert num_images == 2

def test_roc_curve():

    # Delete all images, so we are sure we made them here
    delete_images()

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

def test_feature_importance_plot():

    # Delete all images, so we are sure we made them here
    delete_images()

    # Load models for plotting roc curve 
    rfc = joblib.load('./models/rfc_model.pkl')
    X = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
    feature_importance_plot(rfc, X, image_save_path)

    # Assert expected images were created
    image_pth = Path(image_save_path)
    num_images = len(list(image_pth.glob('*.png')))
    assert num_images == 1 

def test_feature_explainer_plot():

    # Delete all images, so we are sure we made them here
    delete_images()

    # Load models for plotting roc curve 
    rfc = joblib.load('./models/rfc_model.pkl')
    X = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
    explainer_plot(rfc, X, image_save_path)

    # Assert expected images were created
    image_pth = Path(image_save_path)
    num_images = len(list(image_pth.glob('*.png')))
    assert num_images == 1 


def test_train_models():

    # Delete all images, so we are sure we made them here
    delete_models()

    # Load models for plotting roc curve
    X_train = pd.read_csv('./data/X_train.csv', index_col='Unnamed: 0')
    X_test = pd.read_csv('./data/X_test.csv', index_col='Unnamed: 0')
    y_train = pd.read_csv('./data/y_train.csv', index_col='Unnamed: 0')
    y_test = pd.read_csv('./data/y_test.csv', index_col='Unnamed: 0')     
    train_models(X_train, X_test, y_train, y_test, param_grid, image_save_path, model_save_path)

    # Assert expected models were created
    mod_pth = Path(model_save_path)
    num_models = len(list(mod_pth.glob('*.pkl')))
    assert num_models == 2 