from stroke_assessment import explore
import os

# define some useful global variables to be used throughout the project
WORKING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(WORKING_DIR, 'healthcare-dataset-stroke-data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_2v.csv')
TEST_DATA = os.path.join(DATA_DIR, 'test_2v.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'Plots')
# make plots dir if not exists
os.makedirs(PLOTS_DIR) if not os.path.isdir(PLOTS_DIR) else None

# make sure we can find the data
for i in [TRAIN_DATA, TEST_DATA]:
    if not os.path.isfile(i):
        raise FileNotFoundError('Cannot find the data file "{}"'.format(i))
























