import pickle
import pandas as pd
import unittest
from surprise import accuracy
from sklearn.metrics import mean_absolute_error


class RecsTestCase(unittest.TestCase):
    def test_svd_recs(self):
        with open('lab2\svd.pkl', 'rb') as f:
            loaded_svd = pickle.load(f)
        td = pd.read_csv("lab2\svd_test.csv")
        predictions = loaded_svd.test(td.values)
        mae = accuracy.mae(predictions)
        print(mae)
        self.assertLessEqual(mae, 1.3)

    def test_linreg_recs(self):
        with open('lab2\linreg.pkl', 'rb') as f:
            loaded_linreg = pickle.load(f)
        td = pd.read_csv("lab2\linreg_test.csv")
        y = td.pop("y")
        predictions = loaded_linreg.predict(td)
        mae = mean_absolute_error(y, predictions)
        print(mae)
        self.assertLessEqual(mae, 1.5)
