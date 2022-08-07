import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))

TRAIN_DIRECTORY_PATH = os.path.join(SCRIPT_DIR, 'train_data')
TEST_DIRECTORY_PATH = os.path.join(SCRIPT_DIR, 'test_data')

# Mean and Std values for images in the dataset, get them using utils.get_mean_and_std
TRUCKS_DATASET_MEAN = [0.5244, 0.4904, 0.4781]  # TODO CHANGE
TRUCKS_DATASET_STD = [0.2642, 0.2608, 0.2561]  # TODO CHANGE
