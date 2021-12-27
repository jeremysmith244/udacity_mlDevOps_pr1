# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Goal of this project is to take a messy (but functional) jupyter notebook, and turn it into a maintainable module.

## Project Files

Original notebook, and some exploratory stuff is located in notebooks folder.

All of the production (completed) code is located in bin folder. Importantly, churn_library.py is the completed code that runs model and analysis. 

test_churn_library.py is file which runs all unit tests for functions in churn_library.py. It is run using pytest module.

After running code, images and models folders will get populated with the analysis and fitted model files. 


## Running Files

In order to run the model, run it from python as a script:

python churn_library.py

After running, two trained models will get saved into models folder, one for a random forest,
and another for logistic regression. These models can be loaded for making predictions using joblib.load(). Additionally, exploratory data analysis plots will be saved into images folder, along with roc curves, classification report summary and feature importance plots. 

Instead, if one wishes to run tests, simply navigate to ~ and run run_tests.bat or run_tests.sh. If code is run this way, then model and images will not saved (or at least, they will be deleted at the end), but a log file will get saved into logs folder, summarizing results of tests. Note flags used for pytest call, in general pytest overides log file saving, so instead log output saving is specified by use of these flags in the call.

