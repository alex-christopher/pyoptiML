import pandas as pd
import numpy as np

from import_file import data_loading

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


class expert_preprocessing:
    def __int__(self):
        self.dc = data_loading()
        self.df, self.file_path = self.dc.import_data()
        self.print = self.dc.print_def
    def finding_datatypes(self):
        try:
            if self.df:
                float_values = self.df.select_dtypes(include=["float"])
                integer_values = self.df.select_dtypes(include=["int"])
                date_pattern = r'\d{4}-\d{2}-\d{2}'
                if self.df:
                    pass
                return float_values, integer_values
        finally:
            print("Segregated with datatypes")