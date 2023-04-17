from testing import testing_module
import time


class pyoptiML:
    def __init__(self):
        start = time.time()
        test = testing_module()
        test.testing_user_input()
        end = time.time()
        print("------%s seconds------ " % (round(end - start, 2)))
