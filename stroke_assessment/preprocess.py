import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class PreprocessData:

    def __init__(self, input_data):
        self.input_data = input_data
        self.output_data_ = self.process()

    def process(self):
        data = self._one_hot_encoding(self.input_data)
        data = self._scale(data)
        data = self._bool(data)
        data = self._dropna(data)
        return data

    @staticmethod
    def _one_hot_encoding(data):
        categorical = ['gender', 'smoking_status', 'work_type']
        df_list = []
        for i in categorical:
            df_list.append(pd.get_dummies(data[i]))
        data.drop(categorical, axis=1, inplace=True)
        return pd.concat([data] + df_list, sort=False, axis=1)

    @staticmethod
    def _scale(data):
        scaler = MinMaxScaler()
        for i in ['age', 'bmi', 'avg_glucose_level']:
            data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))
        return data

    @staticmethod
    def _bool(data):
        boolean_vars = ['ever_married', 'Residence_type']
        data['ever_married'] = pd.Series(np.where(data['ever_married'] == 'Yes', 1, 0))
        data['Residence_type'] = pd.Series(np.where(data['Residence_type'] == 'Urban', 1, 0))
        return data

    @staticmethod
    def _dropna(data):
        return data.dropna(axis=0, how='any')








