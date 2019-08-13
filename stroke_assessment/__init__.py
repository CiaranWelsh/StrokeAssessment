from stroke_assessment import explore
import os

# define some useful global variables to be used throughout the project
WORKING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(WORKING_DIR, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_2v.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_2v.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'Plots')
PCA_PLOTS_DIR = os.path.join(PLOTS_DIR, 'PCA_plots')
HIST_PLOTS_DIR = os.path.join(PLOTS_DIR, 'histograms')
SCATTER_MATRIX_DIR = os.path.join(PLOTS_DIR, 'scatter_matrix')
MODEL_FILE = os.path.join(WORKING_DIR, 'ffn_model.h5')
HISTORY_FILE = os.path.join(WORKING_DIR, 'history.csv')
BOOTSTRAP_HISTORY_FILE = os.path.join(WORKING_DIR, 'boot_history.csv')
HISTORY_PLOT_FILE = os.path.join(PLOTS_DIR, 'history.png')
TRAIN_DATA_DESCRIPTION_FILE = os.path.join(DATA_DIR, 'train_data_description.csv')

# make plots dir if not exists
os.makedirs(PLOTS_DIR) if not os.path.isdir(PLOTS_DIR) else None
os.makedirs(PCA_PLOTS_DIR) if not os.path.isdir(PCA_PLOTS_DIR) else None
os.makedirs(HIST_PLOTS_DIR) if not os.path.isdir(HIST_PLOTS_DIR) else None
os.makedirs(SCATTER_MATRIX_DIR) if not os.path.isdir(SCATTER_MATRIX_DIR) else None

# make sure we can find the data
for i in [TRAIN_DATA, TEST_DATA]:
    if not os.path.isfile(i):
        raise FileNotFoundError('Cannot find the data file "{}"'.format(i))





"""
Sources
-------
- for the imbalanced data problem:
  - https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
  - explain that with more time I would explore more imbalanced data strategies. Make sure the implementation 
    allows for easy plug in of different imbalanced data strategies. 
"""


















