from testing import testing
import time


class the_end:
    def __init__(self):
        start = time.time()
        test = testing()
        over = test.testing()
        end = time.time()
        print("------%s seconds------ " %(round(end-start, 2)))


end = the_end()