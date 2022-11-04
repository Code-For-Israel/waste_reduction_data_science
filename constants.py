import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))

TRAIN_DIRECTORY_PATH = os.path.join(SCRIPT_DIR, 'train_data')  # TODO DEL
TEST_DIRECTORY_PATH = os.path.join(SCRIPT_DIR, 'test_data')  # TODO DEL

# Mean and Std values for images in the dataset, get them using utils.get_mean_and_std
TRUCKS_DATASET_MEAN = [0.4740, 0.4618, 0.4479]  # TODO CHANGE ACCORDING TO TRAIN DATA
TRUCKS_DATASET_STD = [0.1322, 0.1332, 0.1404]  # TODO CHANGE ACCORDING TO TRAIN DATA
