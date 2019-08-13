import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# a note on why I've use static methods. It basically makes it easier to chop and change how I perform the
#  cleaning and preprocessing step and experiment with how it affects the accuracy scores

class PreprocessData:
    """
    class to store all operations relating to cleaning the raw data and
    preparing it for input to a ML model

    Usage
    -----
    >>> data = pd.read_csv(<data_file>) # read in data
    >>> processed_data = PreprocessData(data).output_data_ # processed data is stored in output_data_ attribute

    """

    def __init__(self, input_data, impute_strategy='median') -> None:
        self.input_data = input_data
        self.impute_strategy = impute_strategy
        self.output_data_ = self.process()

    def process(self):
        """
        The main method in this class which calls all other required methods
        Returns: pd.DataFrame
        """
        data = self.input_data
        # drop smoking status since ~30% are missing data. Can't impute. Might be able to do something with this afterwards
        data = self.drop_smoking_status(data)
        # remove residence column. EDA seems to suggest that it is not imformative for predicting strokes
        data = self.drop_residence(data)
        # Under 30's don't often have strokes. Remove from the model before under sampling
        data = self.discard_the_young(data)
        # male and female predominate the population. Lets stick with that.
        data = self.drop_other_category_from_gender(data)
        # Use median to impute bmi
        data = self.impute(data)
        # One hot encode categorical vars
        data = self.one_hot_encoding(data)
        # Scale the contineous vars between 0 and 1
        data = self.scale(data)
        # convert boolean vars to 1's and 0's
        data = self.convert_to_bool(data)
        # Drop rows which have nan in them. Should actually be none left after the above processing.
        data = self.dropna(data)
        return data

    @staticmethod
    def discard_the_young(data, age_discard=30):
        return data[data['age'] > age_discard]

    @staticmethod
    def drop_other_category_from_gender(data):
        return data[data['gender'] != 'Other']

    @staticmethod
    def one_hot_encoding(data):
        # categorical = ['gender', 'smoking_status', 'work_type']
        categorical = ['gender', 'work_type']
        df_list = []
        for i in categorical:
            df_list.append(pd.get_dummies(data[i]))
        data.drop(categorical, axis=1, inplace=True)
        return pd.concat([data] + df_list, sort=False, axis=1)

    @staticmethod
    def scale(data):
        scaler = MinMaxScaler()
        for i in ['age', 'bmi', 'avg_glucose_level']:
            data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))
        return data

    @staticmethod
    def convert_to_bool(data):
        boolean_vars = ['ever_married', 'Residence_type']
        data['ever_married'] = data['ever_married'].str.strip().str.lower()
        married = []
        resident = []
        for i in data['ever_married']:
            married.append(1) if i.strip().lower() == 'yes' else married.append(0)

        data['ever_married'] = married
        # I have removed residence type from the analysis
        # data['Residence_type'] = resident
        return data

    @staticmethod
    def dropna(data):
        return data.dropna(axis=0, how='any')

    @staticmethod
    def drop_smoking_status(data):
        return data.drop('smoking_status', axis=1)

    @staticmethod
    def drop_residence(data):
        return data.drop('Residence_type', axis=1)

    @staticmethod
    def impute(data):
        bmi = SimpleImputer(strategy='median').fit_transform(data['bmi'].values.reshape([-1, 1]))
        data['bmi'] = bmi
        return data


class UnderSample:

    def __init__(self):
        pass

    @staticmethod
    def under_sample(data) -> pd.DataFrame:
        """
        Implements a simple undersampling strategy. Keeps all of the stroke data and
        randomly the sam number of samples from the non stroke data.
        Args:
            data: pd.DataFrame to sample from

        Returns: training_data

        """
        strokes = data[data['stroke'] == 1]
        healthy = data[data['stroke'] == 0]
        data_healthy = healthy.sample(n=strokes.shape[0])
        data_strokes = strokes.sample(n=strokes.shape[0])
        data = pd.concat([data_strokes, data_healthy]).sample(frac=1).reset_index(drop=True)
        return data

    @staticmethod
    def under_sample_with_val_split(data, val_prop=0.2):
        """
        Performs the same sampling strategy as :py:meth:`UnderSample.under_sample`.
        except the data are addtionally split into train and validation sets
        with proportion specified by `val_prop`.
        Args:
            data: pd.DataFrame. Which dataframe to sample from
            val_prop: float. train, validation split

        Returns: train_data, val_data

        """
        train_prop = 1 - val_prop
        strokes = data[data['stroke'] == 1]
        healthy = data[data['stroke'] == 0]

        train_data_healthy = healthy.sample(n=int(strokes.shape[0] * train_prop), replace=False)
        train_data_strokes = strokes.sample(n=int(strokes.shape[0] * train_prop), replace=False)
        train_data = pd.concat([train_data_healthy, train_data_strokes]).sample(frac=1).reset_index(drop=True)

        val_data_healthy = healthy.sample(n=int(strokes.shape[0] * val_prop), replace=False)
        val_data_strokes = healthy.sample(n=int(strokes.shape[0] * val_prop), replace=False)
        val_data = pd.concat([val_data_healthy, val_data_strokes]).sample(frac=1).reset_index(drop=True)
        return train_data, val_data

    @staticmethod
    def under_sample_age_strat(data):
        """
        Under sample the non-stroke data with stratification on age, i.e. ensures
        when sampling from the non-stroke data that we have an equal age distribution
        and the same number of stroke and non-stroke samples.
        Args:
            data: pd.DataFrame to be sampled

        Returns: training_data

        """
        strokes = data[data['stroke'] == 1]
        not_stroke = data[data['stroke'] == 0]

        age_categorical = pd.cut(not_stroke.age, bins=range(0, 101, 10))
        age_categorical.name = 'age_categorical'
        not_stroke = pd.concat([not_stroke, age_categorical], axis=1)

        age_categorical = pd.cut(strokes.age, bins=range(0, 101, 10))
        age_categorical.name = 'age_categorical'
        strokes = pd.concat([strokes, age_categorical], axis=1)

        counts = {}
        for label, df in strokes.groupby('age_categorical'):
            counts[label] = df['stroke'].sum()

        not_stroke_samples = []
        for label, df in not_stroke.groupby('age_categorical'):
            not_stroke_samples.append(df.sample(n=counts[label]))

        not_stroke_samples = pd.concat(not_stroke_samples)
        train_data = pd.concat([strokes, not_stroke_samples])
        return train_data

    @staticmethod
    def under_sample_age_strat_with_val_split(data, val_prop=0.2) -> tuple:
        """
        The same as :py:meth:`UnderSample.under_sample_age_strat but additionally
        split the data into train and validation data sets.

        Args:
            data (pd.DataFrame): data to be sampled
            val_prop (float): proportion of data to use as validation data

        Returns (tuple of pd.DataFrames): (train, val)

        """
        # calculate training proportion
        train_prop = 1 - val_prop
        # isolate the stroke and non stroke datasets
        strokes = data[data['stroke'] == 1]
        not_stroke = data[data['stroke'] == 0]

        # create bins for age category on strokes
        age_categorical = pd.cut(not_stroke.age, bins=range(0, 101, 10))
        age_categorical.name = 'age_categorical'
        not_stroke = pd.concat([not_stroke, age_categorical], axis=1)

        # create bins for age category on non-strokes data
        age_categorical = pd.cut(strokes.age, bins=range(0, 101, 10))
        age_categorical.name = 'age_categorical'
        strokes = pd.concat([strokes, age_categorical], axis=1)

        # get the number of counts needed from strokes for non strokes
        counts = {}
        for label, df in strokes.groupby('age_categorical'):
            counts[label] = df['stroke'].sum()

        # turn counts into probabilities
        proportions = {}
        total = sum(counts.values())
        for label, num in counts.items():
            proportions[label] = num / total

        # calculate the number of train and validation samples we need
        nstrokes_tot = strokes.shape[0]
        num_train = int(np.floor(nstrokes_tot * train_prop))
        num_val = int(np.floor(nstrokes_tot * val_prop))

        # use the strokers age distribution to sample from non-strokes
        not_stroke_samples_train = []
        not_stroke_samples_val = []
        for label, df in not_stroke.groupby('age_categorical'):
            num1 = int(np.floor(num_train * proportions[label]))
            num2 = int(np.floor(num_val * proportions[label]))
            not_stroke_train = df.sample(n=num1)
            df = df.drop(not_stroke_train.index)
            # ensure we do not sample same row twice
            if df.empty:
                continue
            not_stroke_samples_val.append(df.sample(n=num2))
            not_stroke_samples_train.append(not_stroke_train)

        # concat
        not_stroke_samples_train = pd.concat(not_stroke_samples_train)
        not_stroke_samples_val = pd.concat(not_stroke_samples_val)

        # repeat with the strokes data
        stroke_samples_train = []
        stroke_samples_val = []
        for label, df in strokes.groupby('age_categorical'):
            num1 = int(np.floor(num_train * proportions[label]))
            num2 = int(np.floor(num_val * proportions[label]))
            stroke_train = df.sample(n=num1)
            df = df.drop(stroke_train.index)
            # ensure we do not sample same row twice
            if df.empty:
                continue
            stroke_samples_val.append(df.sample(n=num2))
            stroke_samples_train.append(stroke_train)

        # concat
        stroke_samples_train = pd.concat(stroke_samples_train)
        stroke_samples_val = pd.concat(stroke_samples_val)

        # concat and shuffle
        train = pd.concat([stroke_samples_train, not_stroke_samples_train])
        val = pd.concat([stroke_samples_val, not_stroke_samples_val])
        train = train.sample(frac=1.0)
        val = val.sample(frac=1.0)

        # remove the categorical age col
        train.drop('age_categorical', axis=1, inplace=True)
        val.drop('age_categorical', axis=1, inplace=True)

        return train, val
