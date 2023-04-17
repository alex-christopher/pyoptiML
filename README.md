# pyoptiML
This is a package that is developed to automate all the process of Machine Learning to prepare a optimal model on its own

# What is it?
automate_ml is a Python package that provides fast model generation for any dataset that contains numerical values in them. It aims to help everyone to create model on their own without most knowledge about Machine Learning and how it works. This package runs on its own and suggest the best algorithm to choose to fit the data that has been processed. 

# Main Features
Things that automate_ml does:
1. Importing the data - is way easier now, you can just select the file from the path not to copy the path and paste them to make them run.
2. Preprocessing - It fills the nan values / NA values /  empty values on its own not with mean median or mode alternatively it uses prediction algorithm to predict the missing values. 
Reasons : 
Rather than filling with mean, median or mode values this could help to fill the empty values more accurately.
Model with the best accuracy will be used to fill the nan values and so the accuracy increases in filling the missing values
3. Saving - Saving the files is the most important feature that has been processed. This package asks for yes or no kinda questions to save train data, test data, standardized data and even the model everything will be saved in a new folder that is created from where the data has been selected.
4. Handled outliers with the help of IQR (Interquaratile Range)
5. Numerical data are stored in csv files
6. Models are stored in the same folder so we can just load them to test the accuraccy and can be shared
7. Testing - After the model has been saved it also asks for testing to the user. So here user can just enter the values and the model predicts the output ie the target feature the model is trained with
8. It shows the total time taken to execute the whole process
9. This package intracts with the user so it can be more comfortable than coding and creating a model out of it

# Where to get it

The source code is currently hosted on Github at : https://github.com/alex-christopher/pyoptiML

pip install 

# Dependencies
1. Numpy
2. Pandas
3. Scikit-learn

These are the very basic libraries that are needed for machine learning and this package is built with all these to reduce the work that is done by the users

# License

# Documentation
The official documentation is hosted an PyData.org : 

# Contributing to automate_ml

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcomed.

Or maybe through using pandas you have an idea of your own or are looking for something in the documentation and thinking ‘this can be improved’...you can do something about it!

# Features to improve
1. Used only 4-5 libraries without hyperparameter tuning for the final model generation.
2. Works only for regression problems 
3. Works under progress for classification problems
4. Auto code generation for all the steps or all the progess that the user has done has to be generated on its own 
