import sys

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from savefile import savefileas

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import GradientBoostingRegressor

import pickle

sc = StandardScaler()


class model_creation:
    def __init__(self):
        self.save = savefileas()
        self.df, self.path = self.save.fixing_file_path()

    def new_save(self, process, data, file_format):
        '''
        This function in general is used to save the file or data in a particular format
        :param process: Standardization, testing, training etc.
        :param data: The data that has to be saved
        :param file_format: pkl format for models and csv for numerical data and jpg for visualization
        :return:
        '''
        self.save_file = str(input(f"Do you like to save the {process} as {file_format} file? [y/n] : "))
        if self.save_file.lower()[0] == 'y':
            folder_name = os.path.split(self.path)[1]
            if file_format == ".csv":
                file_name = f'{process}_{folder_name}{file_format}'
                print(file_name)
                os.chdir(self.path)
                data.to_csv(file_name, index=False)
                print(f'{process} saved')

            elif file_format == ".pkl":
                self.model_file_name = f'{process}_{folder_name}{file_format}'
                pickle.dump(data, open(self.model_file_name, 'wb'))
                print('MODEL IS BEING SAVED...')
                print("MODEL SAVED")

            elif file_format == ".jpg":
                plt.imsave(f'{process}_{data}{file_format}', data)
            else:
                print("FILE NOT SAVED")
                sys.exit()
        else:
            print("FILE NOT SAVED")

    def visualization(self, column, process):
        hist = plt.hist(self.df[column])
        plt.imsave(f'Before_Outliers{column}.jpg', hist)
        box = plt.box(self.df[column])
        self.new_save(process, box, ".jpg")

    def treating_outliers(self):
        '''
        This function treats the outliers with Interquaratile range 25% for the minimum value and 75% for maximum value
        :return: self.df The updated new dataframe
        '''
        try:
            for i in self.df.columns:
                Q1 = self.df[i].quantile(.25)
                Q3 = self.df[i].quantile(.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                print(f"LOWER LIMIT FOR {i} : ", round(lower, 2))
                print(f"UPPER LIMIT FOR {i} : ", round(upper, 2))

                self.df[i] = np.where(self.df[i] > upper, upper,
                                      np.where(self.df[i] < lower, lower, self.df[i]))
            return self.df
        except Exception as e:
            print("ERROR IN REMOVING OUTLIERS", e)

    def train_test_split_(self):
        """
        This function in splits the data for training and testing
        :return: New updated dataframe self.df
        """
        # self.df = self.treating_outliers()
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
                self.new_save("Train values", train_data, ".csv")
                # print("Train data saved")
                self.new_save("Test values", test_data, ".csv")
                # print("Test data saved")
                break
            else:
                print("Enter the appropriate column name")

        return x, y, target

    def standardization(self):
        self.df = sc.fit_transform(self.df)
        standard = pd.DataFrame(self.df)
        self.new_save("Standardized", standard, ".csv")

    def validation_model_generation(self):
        """
        This function creates a validation model that can be used to check which model fits well and that can be suggested to the user
        :return: Least MSE value as the greatest index and model as the best model will be suggested
        """
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

    def final_model_generation(self):
        """
        This function will generate the real model that can be reused
        :return:  x values as x
                  y values as y
                  model as model_file_name
                  target as target for cross checking purposes
        """
        try:
            x, y, target = self.train_test_split_()
            index_values, validation = self.validation_model_generation()

            new_model = validation.fit(x, y.values.ravel())
            self.new_save("Model", new_model, ".pkl")
            if self.save_file.lower() == 'y':
                return x, y, self.model_file_name, target
            else:
                return x, y, new_model, target
        except Exception as e:
            print("ERROR AT MODEL GENERATION", e)
