import re

import pandas as pd
import numpy as np
import os
from import_file import data_cleaning

from sklearn.model_selection import train_test_split


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
            for i in self.null_columns:
                sum_of_missing_values = self.df[i].isna().sum()
                # print("NO OF MISSING VALUES : ", sum_of_missing_values)
                print("No of missing values in '{}' : ".format(i), self.df[i].isna().sum())
        except Exception as e:
            print("Missing value function", e)

    def splitting_data(self):
        try:
            int_vals, float_vals = self.find_datatype()
            target = []
            for cols in self.null_columns:
                target.append(cols)
                print("TARGET : ", target)

                training_data = self.df.dropna()
                testing_data = self.df[self.df.isna().any(axis=1)]

                x_train = training_data.drop(target, axis=1)
                y_train = training_data[target]

                #print("XTRAIN : \n", x_train, "\nYTRAIN : \n", y_train)
                #print("TRAINING DATA\n", training_data)
                #print("TESTING DATA\n", testing_data)

                x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, random_state=42, test_size=0.3)

                print("X_val_train : \n", x_val_train, "X_val_test : \n", x_val_test)
                print(x_val_train.shape)
                print(x_val_test.shape)


                target = []
        except Exception as e:
            print("Splittin data function", e)


pre = preprocessing()
pre.missing_value_data()
pre.splitting_data()
# pre.filling_missing_values()
