"""
This file is for exploration of the data only. Therefore there are a lot of experimental
graphs that were plotted that are not particuarly well documented.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycle
import seaborn as sns

sns.set_context(context='paper')

import stroke_assessment
from sklearn.decomposition import PCA

from stroke_assessment.preprocess import PreprocessData


# todo age stratified sampling
# todo evaluate the impact of removing bmi. Doesn't look like there is much of a relationship with strokds
# todo evaluate the impact of removing ever married column.

class Plots:
    """
    Class to hold methods pertaining to plotting data.

    Usage
    -----
    >>> p = Plots(data)
    >>> p.plot_multiple_pca()
    >>> p.plot_frequencies()        # note: deprecated after building `plot_scatter_matrices`
    >>> p.plot_scatter_matrices()   # replaces p.plot_frequencies()
    """

    def __init__(self, data, separate_strokers=False, plots_dir=None, savefig=False):
        self.data = data
        self.separate_strokers = separate_strokers
        self.savefig = savefig

        if plots_dir is None:
            plots_dir = stroke_assessment.PLOTS_DIR
        self.plots_dir = plots_dir

    @staticmethod
    def _savefig(fname) -> None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    def pca(self, colour_by='stroke') -> None:
        """
        Compute a principle component analysis and plot on 2d using the contineous variables. Colour by the
        categorical variables.
        Args:
            colour_by (str): a column heading to colour the plot by

        Returns:

        """
        data = self.data[['age', 'avg_glucose_level', 'bmi']]
        pca = PCA()
        pc = pca.fit_transform(data)
        pc = pd.DataFrame(pc, index=data.index)
        data = pd.concat([self.data, pc], axis=1)

        if isinstance(colour_by, str):
            n_colours = len(data[colour_by].unique())
        elif isinstance(colour_by, (tuple, list)):
            n_colours = len(data[colour_by[0]].unique())
        else:
            raise TypeError

        fig, ax = plt.subplots()
        plt.scatter(x=data[0], y=data[1], c=data['stroke'], alpha=0.3, cmap='viridis')  # , label=label)
        cbar = plt.colorbar()
        cbar.ax.set_title('stroke')
        sns.despine(fig=fig, top=True, right=True)
        plt.xlabel('PC1 ({:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100))
        plt.ylabel('PC2 ({:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100))
        if n_colours < 8:
            plt.legend()
        plt.title(f'PCA coloured by {colour_by}')
        if self.savefig:
            fname = os.path.join(stroke_assessment.PCA_PLOTS_DIR, f'{colour_by}.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_multiple_pca(self) -> None:
        """
        Wrapper around :py:meth:`Plot.pca` to colour by categorical variables
        Returns:

        """

        for i in [i for i in self.data.columns if i != 'stroke']:
            self.pca(i)

    def plot_frequency(self, x='age') -> None:
        """
        PLots a sns.distplot for `x` variable in self.data

        Args:
            x (str): variable to plot

        Returns:

        """
        data = PreprocessData.impute(self.data)
        strokers = data[data['stroke'] == 1]
        print(strokers.head())
        fig = plt.figure()
        sns.distplot(strokers[x], norm_hist=False, kde=False,
                     hist_kws=dict(edgecolor='black', linewidth=2),
                     color='green')
        sns.despine(fig=fig, top=True, right=True)
        plt.ylabel('Stroke Frequency')
        plt.title('Distribution of stroke incidence by {}'.format(x))

        if self.savefig:
            fname = os.path.join(stroke_assessment.HIST_PLOTS_DIR, f'{x}.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_frequencies(self)-> None:
        """
        wrapper around :py:meth:`Plot.plot_frequency`
        Returns:

        """
        for i in ['age', 'bmi', 'avg_glucose_level']:
            self.plot_frequency(i)

    def plot_scatter_mtrx(self, hue='stroke', prefix='')-> None :
        """
        plot scatter matrix of contineous variables that is coloured by hue. When hue != 'stroke'
        the variable is combined with the stroke column to visualise hue and stroke together.
        Args:
            hue (str): one of the categorical variables in self.data
            prefix (str): Prefixed onto the filename. Used to prevent overwriting plot files.

        Returns:

        """
        vars = ['age', 'bmi', 'avg_glucose_level']
        print(self.data.columns)
        if hue != 'stroke':
            unique_stroke_vals = self.data['stroke'].unique()
            unique_hue_vals = self.data[hue].unique()
            from itertools import product
            x = product(unique_hue_vals, unique_stroke_vals)
            lookup = {(i, j): f'{i}_{j}' for i, j in x}

            # todo: find the pythonic way of doing this. This takes far too long.
            new_col = {}
            for i in self.data.index:
                i_data = self.data.loc[i][[hue, 'stroke']]
                val = lookup[tuple(i_data.values)]
                new_col[i] = val
            self.data[f'{hue}_and_stroke'] = pd.Series(new_col)

        plt.figure()

        p = sns.pairplot(
            data=self.data,
            vars=vars,
            hue=f'{hue}_and_stroke' if hue != 'stroke' else hue,
            # markers='.',
            diag_kws=dict(
                linewidth=2
            ),
            plot_kws=dict(
                edgecolor='black',
                linewidth=0.5,
                alpha=0.4
            ),
        )
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        if self.savefig:
            fname = os.path.join(stroke_assessment.SCATTER_MATRIX_DIR, f'{prefix}.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight')

    def plot_scatter_matrices(self, prefix='') -> None:
        """
        wrapper around :py:meth:`plot_scatter_mtrx()`. Plots scatter matrices for several variables.

        Args:
            prefix:

        Returns:

        """
        vars = ['hypertension',
                'heart_disease',
                'ever_married',
                'Residence_type',
                'work_type',
                'Male',
                ]
        for i in vars:
            try:
                self.plot_scatter_mtrx(hue=i, prefix=f'{prefix}_{i}')
            except Exception:
                continue


class Stats:
    """
    class to compute stats on data. This is mostly empty for now, with a view to expand
    when needed in the future.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def summarise(self) -> pd.DataFrame:
        desc = self.data.describe()
        desc.to_csv(stroke_assessment.TRAIN_DATA_DESCRIPTION_FILE)
        print(f'saved to "{stroke_assessment.TRAIN_DATA_DESCRIPTION_FILE}"')
        return self.data.describe()


if __name__ == '__main__':
    train_data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
    test_data = pd.read_csv(stroke_assessment.TEST_DATA, index_col='id')

    # I used this code to figure out how imbalanced the instances of stroke and non stroke records we have
    percent_strokers = pd.DataFrame(train_data['stroke'].value_counts())
    percent_strokers.columns = ['counts']
    percent_strokers['percent'] = percent_strokers['counts'] / train_data.shape[0]
    print(percent_strokers)
    """
           counts   percent
        0   42617  0.981959
        1     783  0.018041
    """

    # This code was used to determine which features have a low enough number of missing values that can be imputed
    df = pd.DataFrame(train_data.isna().sum())
    df.columns = ['counts']
    df['percent'] = df['counts'] / train_data.shape[0]
    # all fields are 0 except bmi and smoking status
    print(df.loc[['bmi', 'smoking_status']])
    """
                    counts   percent
    bmi               1462  0.033687
    smoking_status   13292  0.306267
    """
    # This means we can probably safely impute the bmi, but not smoking status. Though it doens't make sense to
    # impute this anyway

    proc_data = PreprocessData(train_data).output_data_
    # Plots(train_data, savefig=False).plot_scatter_mtrx(hue='ever_married', prefix='raw')

    print(proc_data)

    # print(train_data[(train_data['ever_married'] == 'No') & (train_data['stroke'] == 1)])
    # Plots(proc_data, savefig=True).plot_scatter_mtrx(hue='stroke', prefix='proc')

    Plots(train_data, savefig=True).plot_scatter_matrices(prefix='raw')
    Plots(proc_data, savefig=True).plot_scatter_matrices(prefix='proc')



    # Plots(strokers_processed).plot_scatter_mtrx(prefix='strokers_proc')
    # Plots(non_strokers_processed).plot_scatter_mtrx(prefix='non_strokers_proc')

    # print(data[data['stroke'] == 1])

    # mar = train_data[['ever_married']]
    #
    # print(mar.iloc[:, 0].unique())

    # print(train_data[train_data['age'] > 30])

    # print(train_data)
    # lt_30 = train_data[train_data['age']  <  30]
    # print(lt_30['stroke'].value_counts())
    # get number of strokes as a function of age group.
    # train_data['age_range']  = pd.cut(train_data['age'], range(0, 101, 5))
    # x = train_data[(train_data['age'] < 5) & (train_data['stroke'] == 1)]
    #
    # sns.pairplot(train_data.drop(['']))
    #
    # plt.show()

    # look at the distribution of ages/bmis of strokers
    # explore the dataset more visually scatter mtix

    # plt.show()

    # print(strokes)
    # print('strokes :', strokes.shape)
    # print('healthy :', healthy.shape)
    #
    # print(train_data['stroke'].value_counts())
    # from sklearn.model_selection import train_test_split
    # X_train, X_val, y_train, y_test = train_test_split(train_data.drop('stroke', axis=1), train_data['stroke'],
    #                                                    test_size=0.2, stratify=train_data['stroke'],
    #                                                    shuffle=True)
    # print(X_train.shape)
    # print(X_val.shape)
