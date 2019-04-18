import sys
import time
import threading
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from pandas import DataFrame as DF
def train_test_val_split(Dat: DF, x: float, y: float, z: float): 
    '''dat: data
    x: train size
    y: validation size
    z: test size'''
    y_ = y / (1-x)
    z_ = z / (1-x)
    assert_almost_equal(x+y+z, 1)
    assert_almost_equal(y_+z_, 1)
    train, valtest = train_test_split(Dat, train_size=x, test_size=y+z)
    val, test = train_test_split(valtest, train_size=y_, test_size=z_)
    return train, val, test


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield '.'+cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)


