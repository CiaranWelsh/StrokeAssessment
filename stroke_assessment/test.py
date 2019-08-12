import unittest
import os, glob
import pandas as pd
import numpy

from collections import Counter

import stroke_assessment
from stroke_assessment.explore import Plots
from stroke_assessment.preprocess import PreprocessData


class PreprocessTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(stroke_assessment.TRAIN_DATA)
        self.prep = PreprocessData(self.data)

    def test_one_hot_encoding_columns(self):
        expected = ['id', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status', 'stroke', 'Female', 'Male', 'Other',
                    'formerly smoked', 'never smoked', 'smokes']
        actual = self.prep.output_data_.columns
        self.assertListEqual(expected, list(actual))

    def test_one_hot_encoding_values(self):
        expected = {0: 36838, 1: 6562}
        actual = Counter(self.prep.output_data_['smokes'])
        self.assertEqual(expected, actual)

    def test_scale_data(self):
        self.assertTrue(self.prep.output_data_['age'].max() == 1)

    def test_bool(self):
        expected = Counter({1: 27938, 0: 15462})
        actual = Counter(self.prep.output_data_['ever_married'])
        self.assertEqual(expected, actual)


class ExploreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
        self.processed_data = PreprocessData(self.data).output_data_

    def tearDown(self) -> None:
        delete_plots = False
        if delete_plots and os.path.isdir(stroke_assessment.PLOTS_DIR):
            os.remove(stroke_assessment.PLOTS_DIR)

    def test_something2(self):
        ex = Plots(self.data)
        ex._plot_numerical()


if __name__ == '__main__':
    unittest.main()
