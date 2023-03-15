import os
import pandas as pd
import numpy as np
from preprocessing import preprocessing
from import_file import data_cleaning
import tqdm
import tkinter as tk
from tkinter import filedialog
import csv
import pickle

window = tk.Tk()
window.withdraw()
window.update()


class savefile:
    def __init__(self):
        pre = preprocessing()
        self.df, self.file_path = pre.splitting_data()

    def csvfile(self):
        save_option = str(input("Do you like to save the updated dataframe in .csv format? [y/n] : "))
        if save_option.lower() == 'y':
            print(self.file_path)
            folder_path = os.path.split(self.file_path)[0]
            folder_name = os.path.split(self.file_path)[1]

            complete_path = os.path.join(folder_path, folder_name)
            os.mkdir(complete_path)
            csv_path = f'Preprocessed_{folder_name}.csv'
            print(csv_path)

            os.chdir(complete_path)
            csv_file = self.df.to_csv(csv_path, index=False)
            print("SAVED")
            return self.df, complete_path
        else:
            print("FILE NOT SAVED")
        return self.df, None

    def save_model(self, model, model_name):
        save_option = str(input("Do you like to save the model? [y/n] : "))
        if save_option.lower() == 'y':
            pickle.dump(model, open(self.file_path / f"{model_name}.pkl"), "wb")
            print(f"MODEL SAVED IN {self.file_path}")
        else:
            print("MODEL NOT SAVED")
        return

