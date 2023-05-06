import sys

import pandas as pd
import os


import tkinter as tk
from tkinter import filedialog as fd
import pyfiglet

window = tk.Tk()
window.withdraw()
window.update()

text = "OPTI-ML"
ascii_art = pyfiglet.figlet_format(text, font="slant")
terminal_width = len(ascii_art.split('\n')[0])
ascii_art = ascii_art.center(terminal_width)
print(ascii_art)


class data_loading:
    def __init__(self):
        print("*" * 50 + "OPTI ML" + "*" * 50)

    def import_data(self):
        """
        Asks for yes of no question to import a csv or excle file
        :return: True or False
        """
        try:
            import_file = input("Import file [y/n] : ")
            if import_file.lower()[0] == 'y':
                file_dialog_box = fd.askopenfile(
                    filetypes=[("CSV file", "*.csv"), ("Excel file", "*.xlsx"), ("Excel file", "*.xls")])
                self.file_path, file_ext = os.path.splitext(file_dialog_box.name)
                if file_ext == ".csv":
                    df = pd.read_csv(file_dialog_box.name)
                    self.print_def("IMPORTED DATAFRAME")
                    print(df)
                    return df, self.file_path
                if file_ext == ".xls" or ".xlsx":
                    df = pd.read_excel(file_dialog_box.name)
                    print(df)
                    return df, self.file_path
                window.destroy()
                window.mainloop()

                return True

            else:
                print("THANK YOU")
                sys.exit()

        except Exception as e:
            print("Error occurred while importing file", e)
        finally:
            print("IMPORT_FILE MODULE EXECUTED")
            # window.destroy()
            # window.mainloop()

    def print_def(self, text, data=None):
        """
        Printing a line with asterisks
        :param text: Text that is to be inserted within the asterisks
        :param data: None
        :return: The line with asterisks
        """
        print("*" * 50 + text + "*" * 50)
