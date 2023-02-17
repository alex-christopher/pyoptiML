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
                self.float_vals = self.df.select_dtypes(include=['float'])
                self.int_vals = self.df.select_dtypes(include=['int'])
                #print(self.int_vals.values)
                return self.int_vals, self.float_vals
            return
        except Exception as e:
            print("Error occurred in find_datatype function", e)

    def splitting_data(self):
        try:
            print(self.df.isna().sum())
            null_columns = self.df.columns[self.df.isna().sum()]
            print("NULL VALUES AT", null_columns)
        except Exception as e:
            print("SPLITTING DATA FUNCTION", e)



pre = preprocessing()
pre.find_datatype()
pre.splitting_data()
#pre.filling_missing_values()
