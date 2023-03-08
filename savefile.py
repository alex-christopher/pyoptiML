import os
import pandas as pd
import numpy as np
from preprocessing import preprocessing
from import_file import data_cleaning
import tqdm
import tkinter as tk
from tkinter import filedialog
import csv

window = tk.Tk()
window.withdraw()
window.update()

class savefile:
    def __init__(self):
        self.pre = preprocessing()
        self.df = self.pre.splitting_data()
        self.dc = data_cleaning
        self.print = self.dc.print_def

    def csvfile(self):
        save_option = str(input("Do you like to save the updated dataframe in .csv format? [y/n] : "))
        if save_option.lower() == 'y':
            file_path = filedialog.asksaveasfilename(defaultextension=".csv")
            new_save = self.df.to_csv(file_path, index=False)
            print("SAVED")
        else:
            print("FILE NOT SAVED")


save = savefile()
save.csvfile()