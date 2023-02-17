import pandas as pd
import numpy as np
import os
from import_file import data_cleaning


class preprocessing:
    def __init__(self):
        self.dc = data_cleaning()
        self.df = self.dc.import_data()
        self.print = self.dc.print_def

    def find_datatype(self):
        try:
            if not self.df.empty:
                self.print("PRINTING DATATYPES")
                #print(self.df.dtypes)
                float_vals = self.df.select_dtypes(include=['float'])
                int_vals = self.df.select_dtypes(include=['int'])
                #print(int_vals.values)
                return int_vals, float_vals
            return
        except Exception as e:
            print("Error occurred in find_datatype function", e)

    def splitting_data(self):
        int_vals, float_vals = self.find_datatype()
        try:
            null_columns = self.df.columns[self.df.isna().any()]
            for i in null_columns:
                sum_of_missing_values =  self.df[i].isna().sum()
                print("NO OF MISSING VALUES : ", sum_of_missing_values)
                print("No of missing values in '{}' : ".format(i), self.df[i].isna().sum())
        except Exception as e:
            print("SPLITTING DATA FUNCTION", e)



pre = preprocessing()
pre.splitting_data()
#pre.filling_missing_values()
