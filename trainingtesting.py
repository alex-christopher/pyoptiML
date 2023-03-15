import numpy as np
import pandas as pd
import os

from import_file import data_cleaning
from preprocessing import preprocessing
from savefile import savefile

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tqdm
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

    def try_standardization(self):
        self.df = sc.fit_transform(self.df)
        standard = pd.DataFrame(self.df)
        self.new_save("Standardized", standard)

    def train_test_split(self):
        for i in self.df.columns:
            print(i)
        target = input("Select the target column : ")
        while target in self.df.columns:
            x = self.df.drop(target, axis=1)
            y = self.df[target]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            print("Test size is set to 0.3 by default")
            training_data = x_train + y_train
            train_data = pd.DataFrame(training_data)
            testing_data = x_test + y_test
            test_data = pd.DataFrame(testing_data)
            self.new_save("Train", train_data)
            print("Train data saved")
            self.new_save("Test", test_data)
            print("Test data saved")
        else:
            print("Enter proper column name")


model = model_creation()
model.train_test_split()
model.try_standardization()
