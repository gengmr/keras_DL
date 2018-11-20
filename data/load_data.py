import numpy as np
import os

def Load_train_data(data_class, unsupervised_number=None, supervised_number=None):
    if data_class == 'unsupervised':
        unsupervised_data = []
        load_data_path = 'data/train_data/unsupervised_data'
        all_number = len(os.listdir(load_data_path))
        index_list = np.random.choice(all_number, unsupervised_number)
        for index in index_list:
            image = np.load(load_data_path + '/' + str(index) + '.npy')
            unsupervised_data.append(image)
        unsupervised_data = np.array(unsupervised_data)
        unsupervised_data = np.expand_dims(unsupervised_data, axis=-1)

        return normalization(unsupervised_data)

    if data_class == 'supervised':
        supervised_input = []
        supervised_ground_truth = []
        load_data_path = 'data/train_data/supervised_data'
        all_number = len(os.listdir(load_data_path + '/ground_truth'))
        index_list = np.random.choice(all_number, supervised_number)
        for index in index_list:
            input_image = np.load(load_data_path + '/input/' + str(index) + '.npy')
            ground_truth_image = np.load(load_data_path + '/ground_truth/' + str(index) + '.npy')
            supervised_input.append(input_image)
            supervised_ground_truth.append(ground_truth_image)
        supervised_input = np.array(supervised_input)
        supervised_input = np.expand_dims(supervised_input, axis=-1)
        supervised_ground_truth = np.array(supervised_ground_truth)
        supervised_ground_truth = np.expand_dims(supervised_ground_truth, axis=-1)

        return normalization(supervised_input), normalization(supervised_ground_truth)

    if data_class == 'semisupervised':
        unsupervised_data = []
        load_data_path = 'data/train_data/unsupervised_data'
        all_number = len(os.listdir(load_data_path))
        index_list = np.random.choice(all_number, unsupervised_number)
        for index in index_list:
            image = np.load(load_data_path + '/' + str(index) + '.npy')
            unsupervised_data.append(image)
        unsupervised_data = np.array(unsupervised_data)
        unsupervised_data = np.expand_dims(unsupervised_data, axis=-1)

        supervised_input = []
        supervised_ground_truth = []
        load_data_path = 'data/train_data/supervised_data'
        all_number = len(os.listdir(load_data_path + '/ground_truth'))
        index_list = np.random.choice(all_number, supervised_number)
        for index in index_list:
            input_image = np.load(load_data_path + '/input/' + str(index) + '.npy')
            ground_truth_image = np.load(load_data_path + '/ground_truth/' + str(index) + '.npy')
            supervised_input.append(input_image)
            supervised_ground_truth.append(ground_truth_image)
        supervised_input = np.array(supervised_input)
        supervised_input = np.expand_dims(supervised_input, axis=-1)
        supervised_ground_truth = np.array(supervised_ground_truth)
        supervised_ground_truth = np.expand_dims(supervised_ground_truth, axis=-1)

        return normalization(unsupervised_data), \
               normalization(supervised_input), \
               normalization(supervised_ground_truth)

def Load_validation_data():
    validation_input = np.load('data/validation_data/input/validation_input.npy')
    validation_ground_truth = np.load('data/validation_data/ground_truth/validation_ground_truth.npy')

    return normalization(validation_input), normalization(validation_ground_truth)

def Load_test_data():
    test_input = np.load('data/test_data/input/test_input.npy')
    test_ground_truth = np.load('data/test_data/ground_truth/test_ground_truth.npy')

    return normalization(test_input), normalization(test_ground_truth)

def normalization(image_batch):
    max_ = np.load('max.npy')
    min_ = np.load('min.npy')
    image_batch = (image_batch - min_) / (max_ - min_)

    return image_batch








