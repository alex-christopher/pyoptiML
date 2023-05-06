import traceback

import pandas as pd
import numpy as np

from import_file import data_loading

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


class Standard_preprocessing_data:
    def __init__(self):
        self.dc = data_loading()
        self.df, self.file_path = self.dc.import_data()
        self.print = self.dc.print_def

    '''def find_datatype(self):
        try:
            if self.df:
                self.print("PRINTING DATATYPES")
                # print(self.df.dtypes)
                float_vals = self.df.select_dtypes(include=['float'])
                int_vals = self.df.select_dtypes(include=['int'])
                # print(int_vals)
                return int_vals, float_vals


        except Exception as e:
            print("Error occurred in find_datatype function", e)'''

    def missing_value_data(self):
        """
        This function gets the number of missing values in the every column
        :return: Missing values
        """
        # int_vals, float_vals = self.find_datatype()
        try:
            self.null_columns = self.df.columns[self.df.isna().any()]
            if not self.null_columns.empty:
                for i in self.null_columns:
                    # print("NO OF MISSING VALUES : ", sum_of_missing_values)
                    print("No of missing values in '{}' : ".format(i), self.df[i].isna().sum())

            else:
                print("NO NULL VALUES")
        except Exception as e:
            print("Missing value function", e)

    def categorical_to_numerical(self):
        """
        This function converts all the categorical features into numerical features by creating dummies
        :return:
        """
        try:
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
        except Exception as e:
            print("ERROR IN CATEGORICAL TO NUMERICAL", e)

    def cols_with_more_than_one_missing_values(self):
        """
        This function drops all the rows that contains more than one missing values considering the worst case scenario
        :return: New dataframe self.df
        """
        try:
            counter = 0
            self.df = self.df.reset_index(drop=True)
            for i in range(len(self.df.index - 1)):
                '''print(self.df.index)
                print(len(self.df.values))
                print(i)'''
                c = self.df.iloc[i].isna().values.sum()
                if c > 1:
                    self.df = self.df.dropna(thresh=self.df.shape[1] - 1, axis=0)
                    counter += 1
                    return self.df

            print(self.df)
            # print("PLSSSSSSSS", self.new_df)
            print("COUNTER : ", counter)
            print("DATA WITH MORE THAN ONE MISSING COLUMN HAS BEEN DROPPED")
            print("NO OF ROWS DROPPED : ", counter)
            print("NO OF ROWS REMAINING : ", len(self.df.values) - counter)
            print("NEW DF \n", self.df)
            return self.df
        except Exception as e:
            print("ERROR IN COLS WITH MORE THAN ONE MISSING VALUE", e)

    def save_visualization(self):
        pass

    def splitting_data(self):
        """
        This function splits all the data as training and testing to fill the NA values
        The NA values are considered as the testing data and the model is prepared with XGBoost Regressor to fill in the values
        :return: New Dataframe self.df, File path from where the data is fetched self.file_path
        """
        try:
            # int_vals, float_vals = self.find_datatype()
            target = []
            self.df = self.cols_with_more_than_one_missing_values()

            self.categorical_to_numerical()
            self.missing_value_data()
            for cols in self.null_columns:

                target.append(cols)
                print("TARGET : ", target)
                training_data = self.df.dropna(axis=0)
                print("TRAINING DATAAAAAAAAA : \n", training_data)
                testing_data = self.df.drop(training_data.index)
                print("TESTING DATAAAAA : \n", testing_data)

                x_train = training_data.drop(target, axis=1)
                print("X TRAIN : \n", x_train)
                y_train = training_data[target]
                print("Y TRAIN : \n", y_train)

                # print("XTRAIN : \n", x_train, "\nYTRAIN : \n", y_train)
                # print("TRAINING DATA\n", training_data)
                # print("TESTING DATA\n", testing_data)

                x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train,
                                                                                    test_size=0.3, random_state=42)

                print("X_val_train : \n", x_val_train, "X_val_test : \n", x_val_test)
                print("Y_val_train : \n", y_val_train, "Y_val_test : \n", y_val_test)
                print(x_val_train.shape)
                print(x_val_test.shape)

                x_test = testing_data.drop(target, axis=1)
                x_test = x_test.dropna(axis=0)
                y_test = testing_data[target].loc[x_test.index]
                print("X_TEST : \n", x_test, "\nY_TEST\n", y_test)

                '''lr = LinearRegression()
                rid = Ridge()

                model_name = f'{target}_model'

                model_name = rid.fit(x_val_train, y_val_train)
                prediction = model_name.predict(x_val_test)

                R2_score = r2_score(y_val_test, prediction)
                mse = mean_squared_error(y_val_test, prediction)
                print("R2 Score for validation dataset : ", R2_score)
                print("MEAN SQUARED ERROR for validation dataset : ", mse)'''

                '''HistGrad = HistGradientBoostingRegressor()
                modelHGB = HistGrad.fit(x_val_train, y_val_train.values.ravel())
                pred = modelHGB.predict(x_val_test)
                R2_score = r2_score(y_val_test, pred)
                mse = mean_squared_error(y_val_test, pred)
                print(pred)
                print("R2 : ", R2_score)
                print("MSE : ", mse)'''

                GradBoost = GradientBoostingRegressor(loss='squared_error',
                                                      learning_rate=0.1,
                                                      n_estimators=100,
                                                      random_state=42)
                modelGBR = GradBoost.fit(x_val_train, y_val_train.values.ravel())
                GBR_Pred = modelGBR.predict(x_val_test)
                R2_score = r2_score(y_val_test, GBR_Pred)
                mse = mean_squared_error(y_val_test, GBR_Pred)
                print("R2 : ", R2_score, "\nGBR : ", GBR_Pred, "\nMSE : ", mse)

                if R2_score > 0.8:
                    print("MODEL FITTED WELL AND GOOD")
                elif 0.5 > R2_score < 0.8:
                    print("MODEL FITTED GOOD")

                self.print(f"FILLING MISSING VALUES IN COLUMN {cols}")

                missing_value_prediction = modelGBR.predict(x_test)
                c = self.df[target].loc[0][0]
                print(c)

                number_of_decimal_values = str(c)[::-1].find('.')
                print(number_of_decimal_values)

                print("MISSING VALUE TO BE REPLACED : ", missing_value_prediction)

                print(y_test)
                print(y_test.index)

                print("NEW Y TEST \n", self.df)
                print(f"{target} null values : ", self.df.isna().sum())

                print(type(missing_value_prediction))

                # predicted_series_1d = round(missing_value_prediction, 2)

                i = 0
                for j in y_test.index:
                    print("BEFORE UPDATION \n", testing_data)
                    testing_data.loc[[j], target] = np.round(missing_value_prediction[i], number_of_decimal_values)
                    print("AFTER UPDATION\n", testing_data)
                    i = i + 1

                self.df = pd.concat([training_data, testing_data], axis=0)

                print("PRINT ALL", self.df)

                target = []

                self.cols_with_more_than_one_missing_values()

            return self.df, self.file_path

        except Exception as e:
            traceback.print_exc()
            print("Splitting data function", e)

class Expert_preprocessing_data:
    def __int__(self):
        self.dc = data_loading()
        self.df, self.file_path = self.dc.import_data()
        self.print = self.dc.print_def
    def find_datatypes(self, data):
        try:
            if data:
                float_values = data.select_dtypes(include=["float"])
                integer_values = data.select_dtypes(include=["int"])
                date_pattern = r'\d{4}-\d{2}-\d{2}'
                return float_values, integer_values
        finally:
            print("Segregated with datatypes")

    def fill_missing_values(self):
        missing_vals = ["Mean", "Median", "Mode", "Prediction", "Drop Missing value"]
        for i in missing_vals:
            print(i)
        while True:
            missing_vals_input = str(input("Enter the method to fill the missing values : "))
            if missing_vals_input in missing_vals:
                if missing_vals.lower() == "mean":
                    for i in self.df.columns:
                        self.df[i].fillna(method="mean", inplace=True)





