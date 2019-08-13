import unittest
import os, glob
import pandas as pd
import numpy

from collections import Counter

import stroke_assessment
from stroke_assessment.explore import Plots, Stats
from stroke_assessment.preprocess import PreprocessData, UnderSample


# todo write more rigerous tests for UnderSample methods
# todo finish Explore tests.


class PreprocessTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
        self.prep = PreprocessData(self.data)

    def test_columns(self):
        expected = ['age', 'hypertension', 'heart_disease', 'ever_married',
                    'avg_glucose_level', 'bmi', 'stroke', 'Female', 'Male', 'Govt_job',
                    'Never_worked', 'Private', 'Self-employed']
        actual = self.prep.output_data_.columns
        self.assertListEqual(sorted(expected), sorted(list(actual)))

    def test_one_hot_encoding_values(self):
        expected = Counter({0: 27104, 1: 2054})
        actual = Counter(self.prep.output_data_['heart_disease'])
        self.assertEqual(expected, actual)

    def test_scale_data(self):
        self.assertTrue(self.prep.output_data_['age'].max() == 1)

    def test_bool(self):
        expected = Counter({1: 25970, 0: 3188})
        actual = Counter(self.prep.output_data_['ever_married'])
        self.assertEqual(expected, actual)

    def test_under_sample(self):
        data = UnderSample.under_sample(self.prep.input_data)
        expected = [783, 783]
        actual = data['stroke'].value_counts().tolist()
        self.assertEqual(expected, actual)

    def test_under_sample_with_val_split(self):
        train, _ = UnderSample.under_sample_with_val_split(self.prep.input_data)
        expected = [626, 626]
        actual = train['stroke'].value_counts().tolist()
        self.assertEqual(expected, actual)

    def test_impute(self):
        x = self.prep.impute(self.prep.input_data)
        expected = 0
        actual = x.bmi.isna().sum()
        self.assertEqual(expected, actual)


class ExploreTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
        self.processed_data = PreprocessData(self.data).output_data_

    def tearDown(self) -> None:
        delete_plots = False
        if delete_plots and os.path.isdir(stroke_assessment.PLOTS_DIR):
            os.remove(stroke_assessment.PLOTS_DIR)

    def test_pca(self):
        print(self.processed_data.columns)
        ex = Plots(self.processed_data, savefig=True)
        ex.plot_multiple_pca()

    def test_freq(self):
        ex = Plots(self.data, savefig=True)
        ex.plot_frequencies()

    def test_pairplot(self):
        ex = Plots(self.data, savefig=True)
        ex.plot_scatter_mtrx()


class StatsTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
        self.stats = Stats(self.data)

    def test_summarise(self):
        self.stats.summarise()
        self.assertTrue(os.path.isfile(stroke_assessment.TRAIN_DATA_DESCRIPTION_FILE))


if __name__ == '__main__':
    unittest.main()
