import re
import traceback

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
            # self.cat_cols = self.cat_cols.replace({0 : 1, 1 : 2, np.nan : 0})
            print(self.cat_cols)
        else:
            return self.df

        self.df = pd.concat([numerical_cols, self.cat_cols], axis=1)
        print(self.df)

    def cols_with_more_than_one_misisng_values(self):
        counter = 0
        self.df = self.df.reset_index(drop=True)
        for i in range(len(self.df.index)):
            print(self.df.index)
            print(len(self.df.values))
            print(i)
            c = self.df.iloc[i].isna().values.sum()
            if c >= 2:
                self.df = self.df.dropna(thresh=self.df.shape[1] - 1)
                counter += 1
            else:
                return self.df

        print(self.df)
        # print("PLSSSSSSSS", self.new_df)
        print("COUNTER : ", counter)
        print("DATA WITH MORE THAN ONE MISSING COLUMN HAS BEEN DROPPED")
        print("NO OF ROWS DROPPED : ", counter)
        print("NO OF ROWS REMAINING : ", len(self.df.values) - counter)
        print("NEW DF \n", self.df)
        return self.df

    def splitting_data(self):
        try:
            int_vals, float_vals = self.find_datatype()
            target = []
            self.df = self.cols_with_more_than_one_misisng_values()
            for cols in self.null_columns:

                target.append(cols)
                print("TARGET : ", target)
                training_data = self.df.dropna(axis=0)
                print("TRAINING DATAAAAAAAAA : \n", training_data)
                testing_data = self.df.drop(training_data.index)
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
                y_test = testing_data[target].loc[x_test.index]
                print("X_TEST : \n", x_test, "\nY_TEST\n", y_test)

                lr = LinearRegression()

                model = lr.fit(x_val_train, y_val_train)
                prediction = model.predict(x_val_test)

                R2_score = r2_score(y_val_test, prediction)
                mse = mean_squared_error(y_val_test, prediction)
                print("R2 Score for validation dataset : ", R2_score)
                print("MEAN SQUARED ERROR for validation dataset : ", mse)

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

                print(y_test)
                print(y_test.index)

                print("NEW Y TEST \n", self.df)
                print(f"{target} null values : ", self.df.isna().sum())

                '''if x_test.index == y_test.index:
                    for i in x_test.index:
                        for j in missing_value_prediction:
                            self.df[target].loc[i] = j'''
                print(type(missing_value_prediction))

                # predicted_series_1d = round(missing_value_prediction, 2)
                a = missing_value_prediction[0][0]
                print(missing_value_prediction[0][0])

                print("BEFORE UPDATION \n", testing_data)

                testing_data[target] = testing_data[target].fillna(a)
                print("AFTER UPDATION\n", testing_data)

                self.df = pd.concat([training_data, testing_data], axis=0)

                print("PRINT ALL", self.df)

                target = []

                self.cols_with_more_than_one_misisng_values()

                # return "END OF NULL VALUES"

        except Exception as e:
            traceback.print_exc()
            print("Splitting data function", e)


pre = preprocessing()
pre.categorical_to_numerical()
# pre.cols_with_more_than_one_misisng_values()
pre.missing_value_data()
pre.splitting_data()
# pre.filling_missing_values()
