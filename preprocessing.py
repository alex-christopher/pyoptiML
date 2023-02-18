import re

import pandas as pd
import numpy as np
import os
from import_file import data_cleaning

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


class preprocessing:
    def __init__(self):
        self.dc = data_cleaning()
        self.df = self.dc.import_data()
        self.print = self.dc.print_def

    def find_datatype(self):
        try:
            if not self.df.empty:
                self.print("PRINTING DATATYPES")
                # print(self.df.dtypes)
                float_vals = self.df.select_dtypes(include=['float'])
                int_vals = self.df.select_dtypes(include=['int'])
                # print(int_vals)
                return int_vals, float_vals
            return
        except Exception as e:
            print("Error occurred in find_datatype function", e)

    def missing_value_data(self):
        # int_vals, float_vals = self.find_datatype()
        try:
            self.null_columns = self.df.columns[self.df.isna().any()]
            if not self.null_columns.empty:
                for i in self.null_columns:
                    sum_of_missing_values = self.df[i].isna().sum()
                    # print("NO OF MISSING VALUES : ", sum_of_missing_values)
                    print("No of missing values in '{}' : ".format(i), self.df[i].isna().sum())

            else:
                print("NO NULL VALUES")
        except Exception as e:
            print("Missing value function", e)

    def categorical_to_numerical(self):
        categorical_cols = self.df.select_dtypes(include='object')
        numerical_cols = self.df.select_dtypes(exclude='object')

        if not categorical_cols.empty:
            self.cat_cols = pd.get_dummies(categorical_cols)
            #self.cat_cols = self.cat_cols.replace({0 : 1, 1 : 2, np.nan : 0})
            print(self.cat_cols)
        else:
            return self.df

        self.df = pd.concat([numerical_cols, self.cat_cols], axis=1)
        print(self.df)

    def cols_with_more_than_one_misisng_values(self):
        if self.df.isna().any() > 1:
            cc = self.df.dropna(axis=0)
            print("cccccccccccccccccccc\n", cc)


    def splitting_data(self):
        try:
            int_vals, float_vals = self.find_datatype()
            target = []
            for cols in self.null_columns:

                target.append(cols)
                print("TARGET : ", target)

                training_data = self.df.dropna(axis=0)
                print("TRAINING DATAAAAAAAAA : \n", training_data)
                testing_data = self.df[self.df.isna().any(axis=1)]
                print("TESTING DATAAAAA : \n", testing_data)

                x_train = training_data.drop(target, axis=1)
                print("X TRAI N : \n", x_train)
                y_train = training_data[target]
                print("Y TRAI N : \n", y_train)

                # print("XTRAIN : \n", x_train, "\nYTRAIN : \n", y_train)
                # print("TRAINING DATA\n", training_data)
                # print("TESTING DATA\n", testing_data)

                x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train,
                                                                                    random_state=42, test_size=0.3)

                print("X_val_train : \n", x_val_train, "X_val_test : \n", x_val_test)
                print(x_val_train.shape)
                print(x_val_test.shape)

                x_test = testing_data.drop(target, axis=1)
                x_test = x_test.dropna(axis=0)
                y_test = testing_data[target]
                print("X_TEST : \n", x_test, "\nY_TEST\n", y_test)

                lr = LinearRegression()

                model = lr.fit(x_val_train, y_val_train)
                prediction = model.predict(x_val_test)

                R2_score = r2_score(y_val_test, prediction)
                print("R2 Score for validation dataset : ", R2_score)

                '''HistGrad = HistGradientBoostingRegressor()
                modelHGB = HistGrad.fit(x_val_train, y_val_train.values.ravel())
                pred = modelHGB.predict(x_val_test)
                R2_score = r2_score(y_val_test, pred)
                mse = mean_squared_error(y_val_test, pred)
                print(pred)
                print("R2 : ", R2_score)
                print("MSE : ", mse)'''

                '''GradBoost = GradientBoostingRegressor()
                modelGBR = GradBoost.fit(x_val_train, y_val_train.values.ravel())
                GBR_Pred = modelGBR.predict(x_val_test)
                R2_score = r2_score(y_val_test, GBR_Pred)
                mse = mean_squared_error(y_val_test, GBR_Pred)
                print("R2 : ", R2_score, "\nGBR : ", GBR_Pred)'''
                if R2_score > 0.8:
                    print("MODEL FITTED WELL AND GOOD")
                elif 0.5 > R2_score < 0.8:
                    print("MODEL FITTED GOOD")

                self.print(f"FILLING MISSING VALUES IN COLUMN {cols}")


                missing_value_prediction = model.predict(x_test)
                print("MISSING VALUE TO BE REPLACED : ", missing_value_prediction)

                target = []

                # return "END OF NULL VALUES"

        except Exception as e:
            print("Splitting data function", e)


pre = preprocessing()
pre.missing_value_data()
pre.categorical_to_numerical()
pre.cols_with_more_than_one_misisng_values()
pre.splitting_data()
# pre.filling_missing_values()
