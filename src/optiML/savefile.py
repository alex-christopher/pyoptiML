import os
from preprocessing_data import Standard_preprocessing_data
import pickle


class savefileas:
    def __init__(self):
        pre = Standard_preprocessing_data()
        self.df, self.file_path = pre.splitting_data()

    def fixing_file_path(self):
        save_option = str(input("Do you like to save the updated dataframe in .csv format? [y/n] : "))
        if save_option.lower()[0] == 'y':
            print(self.file_path)
            folder_path = os.path.split(self.file_path)[0]
            folder_name = os.path.split(self.file_path)[1]

            i = 0
            while True:
                if i == 0:
                    complete_path = os.path.join(folder_path, folder_name)
                else:
                    complete_path = os.path.join(folder_path, f'{folder_name}({i})')
                if not os.path.exists(complete_path):
                    os.mkdir(complete_path)
                    break
                i += 1

            # os.mkdir(complete_path)
            csv_path = f'Preprocessed_{folder_name}.csv'
            print(csv_path)

            os.chdir(complete_path)
            self.df.to_csv(csv_path, index=False)
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
