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

sc = StandardScaler()

class model_creation:
    def __init__(self):
        self.save = savefile()
        self.df, self.path = self.save.csvfile()

    def new_save(self, process, data):
        new_save = str(input(f"Do you like to save the {process} values as .csv file? [y/n] : "))
        if new_save.lower() == 'y':
            folder_name = os.path.split(self.path)[1]
            csv_path = f'{process}_{folder_name}.csv'
            print(csv_path)
            os.chdir(self.path)
            csv_file = data.to_csv(csv_path, index=False)
        else:
            print("FILE NOT SAVED")

    def train_test_split(self):
        for i in self.df.columns:
            print(i)
        target = input("Select the target column : ")
        if target in self.df.columns:
            x = self.df.drop(target, axis=1)
            y = self.df[target]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            print("Test size is set to 0.3 by default")
            training_data = pd.concat([self.x_train, self.y_train], axis=1)
            train_data = pd.DataFrame(training_data)
            testing_data = pd.concat([self.x_test, self.y_test], axis=1)
            test_data = pd.DataFrame(testing_data)
            self.new_save("Train", train_data)
            print("Train data saved")
            self.new_save("Test", test_data)
            print("Test data saved")
        else:
            print("Enter proper column name")


    def standardization(self):
        self.df = sc.fit_transform(self.df)
        standard = pd.DataFrame(self.df)
        self.new_save("Standardized", standard)

    def validation_model_generation(self):
        x_train = sc.fit_transform(self.x_train)
        x_test = sc.transform(self.x_test)


        GradBoost = GradientBoostingRegressor(loss='squared_error',
                                              learning_rate=0.1,
                                              n_estimators=100,
                                              random_state=42)
        LinearReg = LinearRegression()
        RidgeReg = Ridge()
        RANSAC = RANSACRegressor()
        ElasticNetReg = ElasticNet()
        LassoReg = Lasso()


        models = [GradBoost, LinearReg, RidgeReg, RANSAC,LassoReg, ElasticNetReg]

        for model in models:
            modelGBR = model.fit(x_train, self.y_train.values.ravel())
            Prediction = modelGBR.predict(x_test)
            R2_score = r2_score(self.y_test, Prediction)
            mse = mean_squared_error(self.y_test, Prediction)
            print(f"{model}")
            print("R2 : ", R2_score, "\nMSE : ", mse)


    def model_generation(self):
        x_train = sc.fit_transform(self.x_train)
        x_test = sc.transform(self.x_test)



model = model_creation()
model.train_test_split()
model.standardization()
model.validation_model_generation()
