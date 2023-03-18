from trainingtesting import model_creation
import pickle

class testing:

    def __init__(self):
        model = model_creation()
        self.x, self.y, self.generated_model, self.target = model.model_generation()

    def testing(self):
        try:
            test_input = str(input("Do you like to test your model [y/n] : "))
            if test_input.lower() == 'y':
                append_values = []
                for i in self.x.columns:
                    values = float(input(f"Enter the value for {i} : "))
                    append_values.append(values)

                load_model = pickle.load(open(self.generated_model, 'rb'))
                pred = load_model.predict([append_values])
                print("Predicted output : ", pred[0])
            else:
                print("THANK YOU")
        finally:
            print("MODEL EXECUTED")

