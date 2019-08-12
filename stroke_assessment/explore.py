import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import stroke_assessment


class Plots:

    def __init__(self, data, separate_strokers=False, plots_dir=None, savefig=False):
        self.data = data
        self.separate_strokers = separate_strokers
        self.savefig = savefig

        if plots_dir is None:
            plots_dir = stroke_assessment.PLOTS_DIR
        self.plots_dir = plots_dir

    @staticmethod
    def _savefig(fname):
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    def _plot_gender_dist(self):
        fig = plt.figure()
        data = self.data[['Male', 'Female']].count().to_dict()
        sns.barplot(x=list(data.keys()), y=list(data.values()))
        sns.despine(fig=fig, top=True, right=True)
        plt.title('Gender distribution in dataset')

        if self.savefig:
            fname = os.path.join(stroke_assessment.PLOTS_DIR, 'gender_distribution.png')
            self._savefig(fname)
        else:
            plt.show()

    def _plot_numerical(self):
        fig = plt.figure()
        sns.violinplot(y=['avg_glucose_level'], data=self.data, hue='stroke')

        plt.show()
        # print(self.data)

    def _plot_bmi_dist(self):
        pass

    def _plot_age_vs_bmi(self):
        pass


if __name__ == '__main__':
    train_data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
    test_data = pd.read_csv(stroke_assessment.TEST_DATA, index_col='id')

    trainx = train_data.drop('stroke')

    print(train_data)
