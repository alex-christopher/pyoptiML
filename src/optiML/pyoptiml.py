from testing import testing_module
import time


class pyoptiML:
    def __init__(self):
        start = time.time()
        mode_selection = str(input("Do you want to enter Standard mode / Expert mode : "))
        if mode_selection.lower()[0] == "s":
            print("Entered Standard Mode")
            test = testing_module()
            test.testing_user_input()
        elif mode_selection.lower()[0] == "e":
            print("Entered Expert Mode")
        else:
            print("Input Error")
        end = time.time()
        print("------%s seconds------ " % (round(end - start, 2)))
