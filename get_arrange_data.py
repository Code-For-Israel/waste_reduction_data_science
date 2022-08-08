import zipfile
import os
from dataset import extract_bboxes_and_labels_from_annotations_txt
import numpy as np
from collections import Counter
import shutil


def extract_annotations_zipfile(path_to_yolo1_1_annotations_zip_file,
                                directory_to_extract_to):
    with zipfile.ZipFile(path_to_yolo1_1_annotations_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def download_images_from_s3_bucket(bukcet_name, directory_to_download_to):
    os.system(f'aws s3 cp s3://{bukcet_name}/ {directory_to_download_to}/ --recursive --exclude "*" --include "*.jpg"')


def get_only_relevant_annotations_filenames(annotations_directory):
    np.random.seed(42)
    all_labels = list()

    filenames = os.listdir(os.path.join(annotations_directory, 'obj_train_data'))
    filenames = [filename for filename in filenames if '.txt' in filename]

    relevant_filenames = list()
    for filename in filenames:
        boxes, labels = extract_bboxes_and_labels_from_annotations_txt(
            os.path.join(annotations_directory, 'obj_train_data', filename))

        # if 0 not in labels:
        #  In YOLO1.1 annotations files downloaded from CVAT label = 0 means General Truck
        #  (because how we defined the manual tagging project in CVAT)
        #  We don't want the images with General Truck (because it means we didn't manually tag this image)
        # and labels:
        #  We only want images with at least one box, because FasterRCNN / FPN  TODO - make sure this is not possible
        if 0 not in labels and labels:
            # Under sample the class `truck_not_relevant` (label = 3)
            if set(labels) == {3}:
                # Randomly get image that all the trucks in it are not relevant
                if np.random.randint(0, 6) == 1:  # TODO Depends on accuracy for not relevant trucks can change the high
                    relevant_filenames.append(filename.replace('.txt', ''))
                    all_labels.extend(labels if labels else [0])
            else:
                relevant_filenames.append(filename.replace('.txt', ''))
                all_labels.extend(labels if labels else [0])

    print(f'Amount of boxes in dataset per class: {Counter(all_labels)}')
    return sorted(relevant_filenames)


def split_data_to_train_test_and_move_to_separate_folders(train_size_ratio, relevant_annotations_filenames,
                                                          all_annotations_directory, all_images_directory,
                                                          train_directory, test_directory):
    np.random.seed(42)
    np.random.shuffle(relevant_annotations_filenames)

    train_files_amount = int(train_size_ratio * len(relevant_annotations_filenames))

    shutil.rmtree(train_directory, ignore_errors=True)
    shutil.rmtree(test_directory, ignore_errors=True)
    os.mkdir(train_directory)
    os.mkdir(test_directory)

    for i, filename in enumerate(relevant_annotations_filenames):
        if i < train_files_amount:
            directory = train_directory
        else:
            directory = test_directory
        print(os.path.join(all_annotations_directory, 'obj_train_data', filename + '.txt'))
        print(os.path.join(directory, filename + '.txt'))
        shutil.move(os.path.join(all_annotations_directory, 'obj_train_data', filename + '.txt'),
                    os.path.join(directory, filename + '.txt'))
        shutil.move(os.path.join(all_images_directory, filename + '.jpg'),
                    os.path.join(directory, filename + '.jpg'))


if __name__ == "__main__":
    path_to_yolo1_1_annotations_zip_file = 'tagged_trucks_07_08_2022.zip'
    annotations_directory = 'all_annotations'
    extract_annotations_zipfile(path_to_yolo1_1_annotations_zip_file=path_to_yolo1_1_annotations_zip_file,
                                directory_to_extract_to=annotations_directory)

    relevant_annotations_filenames = get_only_relevant_annotations_filenames(
        annotations_directory=annotations_directory)

    bucket_name_with_jpg_images = 'tomer-waste'
    images_directory = 'all_images'
    download_images_from_s3_bucket(bukcet_name=bucket_name_with_jpg_images, directory_to_download_to=images_directory)

    train_directory = 'train_data'
    test_directory = 'test_data'
    split_data_to_train_test_and_move_to_separate_folders(train_size_ratio=0.8,
                                                          relevant_annotations_filenames=relevant_annotations_filenames,
                                                          all_annotations_directory=annotations_directory,
                                                          all_images_directory=images_directory,
                                                          train_directory=train_directory,
                                                          test_directory=test_directory)
