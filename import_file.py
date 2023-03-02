import pandas as pd
import numpy as np
import os

import tkinter as tk
from tkinter import filedialog as fd
import pyfiglet

window = tk.Tk()
window.withdraw()
window.update()

text = "Automate ML"
ascii_art = pyfiglet.figlet_format(text, font="slant")
terminal_width = len(ascii_art.split('\n')[0])
ascii_art = ascii_art.center(terminal_width)
print(ascii_art)

class data_cleaning:
    def __init__(self):
        print("*"*50 +"AUTOMATED MACHINE LEARNING"+"*"*50)

    def import_data(self):
        try:
            import_file = input("Import file [y/n] : ")
            if import_file.lower() == 'y':
                file_dialog_box = fd.askopenfile(
                    filetypes=[ ("CSV file", "*.csv"), ("Excel file", "*.xlsx"), ("Excel file", "*.xls")])
                file_path, file_ext = os.path.splitext(file_dialog_box.name)
                if file_ext == ".csv":
                    df = pd.read_csv(file_dialog_box.name)
                    self.print_def("IMPORTED DATAFRAME")
                    print(df)
                    return df
                if file_ext == ".xls" or ".xlsx":
                    df = pd.read_excel(file_dialog_box.name)
                    print(df)
                    return df
                window.destroy()
                window.mainloop()
            else:
                print("THANK YOU")
                return False
        except Exception as e:
            print("Error occurred while importing file", e)
        finally:
            print("IMPORT_FILE MODULE EXECUTED")
            #window.destroy()
            #window.mainloop()

    def print_def(self, text, data=None):
        print("*"*50 + text + "*"*50)
