import numpy as np
import pandas as pd
import os

from import_file import data_cleaning
from preprocessing import preprocessing
from savefile import savefile

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor, Lasso, ElasticNet

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from tqdm import tqdm

import pickle

sc = StandardScaler()


class model_creation:
    def __init__(self):
        self.save = savefile()
        self.df, self.path = self.save.csvfile()

    def new_save(self, process, data, file_format):
        new_save = str(input(f"Do you like to save the {process} values as {file_format} file? [y/n] : "))
        if new_save.lower() == 'y':
            folder_name = os.path.split(self.path)[1]
            if file_format == ".csv":
                file_name = f'{process}_{folder_name}{file_format}'
                print(file_name)
                os.chdir(self.path)
                csv_file = data.to_csv(file_name, index=False)

            elif file_format == ".pkl":
                self.model_file_name = f'{process}_{folder_name}{file_format}'
                pickle.dump(data, open(self.model_file_name, 'wb'))
                print('MODEL IS BEING SAVED...')
                print("MODEL SAVED")
            else:
                print("File NOT SAVED")


    def train_test_split_(self):
        for i in self.df.columns:
            print(i)
        while True:
            target = input("Select the target column : ")
            if target in self.df.columns:
                x = self.df.drop(target, axis=1)
                y = self.df[target]
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=42)
                print("Test size is set to 0.3 by default")
                training_data = pd.concat([self.x_train, self.y_train], axis=1)
                train_data = pd.DataFrame(training_data)
                testing_data = pd.concat([self.x_test, self.y_test], axis=1)
                test_data = pd.DataFrame(testing_data)
                self.new_save("Train", train_data, ".csv")
                print("Train data saved")
                self.new_save("Test", test_data, ".csv")
                print("Test data saved")
                break
            else:
                print("Enter proper column name")

        return x, y, target

    def standardization(self):
        self.df = sc.fit_transform(self.df)
        standard = pd.DataFrame(self.df)
        self.new_save("Standardized", standard, ".csv")

    def validation_model_generation(self):
        self.standardization()
        x_train = sc.fit_transform(self.x_train)
        x_test = sc.transform(self.x_test)

        GradBoost = GradientBoostingRegressor(loss='squared_error',
                                              learning_rate=0.1,
                                              n_estimators=100,
                                              random_state=42)
        LinearReg = LinearRegression()
        RidgeReg = Ridge()
        ElasticNetReg = ElasticNet()
        LassoReg = Lasso()

        models = [GradBoost,
                  LinearReg,
                  RidgeReg,
                  ElasticNetReg,
                  LassoReg]

        r2 = []
        mse_ = []

        for model in models:
            modelGBR = model.fit(x_train, self.y_train.values.ravel())
            Prediction = modelGBR.predict(x_test)
            R2_score = r2_score(self.y_test, Prediction)
            r2.append(R2_score)
            mse = mean_squared_error(self.y_test, Prediction)
            mse_.append(mse)
            print(f"{model}")
            print("R2 : ", R2_score, "\nMSE : ", mse)
        model_selection = ['Gradient Boosting Regressor',
                           'Linear Regression',
                           'Ridge Regression',
                           'Elastic Net Regression',
                           'Lasso Regression']
        greatest_index = r2.index(max(r2))
        best_model = models[greatest_index]
        suggest_model = model_selection[greatest_index]
        print("BEST MODEL : ", suggest_model)
        return greatest_index, best_model

    def model_generation(self):
        x, y, target = self.train_test_split_()
        index_values, validation = self.validation_model_generation()
        GradBoost = GradientBoostingRegressor(loss='squared_error',
                                              learning_rate=0.1,
                                              n_estimators=100,
                                              random_state=42)
        LinearReg = LinearRegression()
        RidgeReg = Ridge()
        ElasticNetReg = ElasticNet()
        LassoReg = Lasso()

        new_model = validation.fit(x, y.values.ravel())
        self.new_save("Model", new_model, ".pkl")

        return x, y, self.model_file_name, target

