"""
File: titanic_level1.py
Name: Hui-Hsuan Chung
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
from util import *

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: an empty Python dictionary
    :param mode: str, indicating the mode we are using
    :param training_data: dict[str: list], key is the column name, value is its data
                          (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """
    ############################
    select_headers = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    with open(filename, 'r') as f:
        extract_hdr = 1
        for line in f:
            # Header information for dict keys
            if extract_hdr == 1:
                extract_hdr = 0
                headers = line.split('\n')[0].split(',')
                for header in headers:
                    if header in select_headers and (mode == 'Train' or mode == 'Test'):
                        data[header] = []
                    if mode == 'Train':
                        data['Survived'] = []
            # Data information for dict values
            else:
                tokens = line.split('\n')[0].split(',')
                if mode == 'Train':
                    i_select_tokens = [1, 2, 5, 6, 7, 8, 10, 12]  # Required indices selected by hand
                    check_nan = 0
                    # Check missing data
                    for i_select_token in i_select_tokens:
                        if not len(tokens[i_select_token]):
                            check_nan += 1
                    # if there is no missing data in the given row, store the data
                    if check_nan == 0:
                        data['Survived'].append(int(tokens[1]))
                        data['Pclass'].append(int(tokens[2]))
                        if tokens[5] == 'male':
                            data['Sex'].append(1)
                        else:
                            data['Sex'].append(0)
                        data['Age'].append(float(tokens[6]))
                        data['SibSp'].append(int(tokens[7]))
                        data['Parch'].append(int(tokens[8]))
                        data['Fare'].append(float(tokens[10]))
                        if tokens[12] == 'S':
                            data['Embarked'].append(0)
                        elif tokens[12] == 'C':
                            data['Embarked'].append(1)
                        else:
                            data['Embarked'].append(2)
                elif mode == 'Test':
                    data['Pclass'].append(int(tokens[1]))
                    if tokens[4] == 'male':
                        data['Sex'].append(1)
                    else:
                        data['Sex'].append(0)
                    # if there is missing data, fill up the average value of training data for the test data
                    if not tokens[5]:
                        data['Age'].append(round(sum(training_data['Age']) / len(training_data['Age']), 3))
                    else:
                        data['Age'].append(float(tokens[5]))
                    data['SibSp'].append(int(tokens[6]))
                    data['Parch'].append(int(tokens[7]))
                    # if there is missing data, fill up the average value of training data for the test data
                    if not tokens[9]:
                        data['Fare'].append(round(sum(training_data['Fare']) / len(training_data['Fare']), 3))
                    else:
                        data['Fare'].append(float(tokens[9]))
                    if tokens[11] == 'S':
                        data['Embarked'].append(0)
                    elif tokens[11] == 'C':
                        data['Embarked'].append(1)
                    else:
                        data['Embarked'].append(2)
                else:
                    print('Please put "Train" or "Test" for the mode parameter')
    return data


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    # Check how many unique values in the given data[feature]
    occurrence = []
    for value in data[feature]:
        if value not in occurrence:
            occurrence.append(value)
    occurrence = sorted(occurrence)

    # Create the new one-hot encoding dictionary keys
    for n_new_col in range(len(occurrence)):
        new_col_name = feature + '_' + str(n_new_col)
        data[new_col_name] = []

    # Start one-hot encoding by looping the values of data[feature]
    for value in data[feature]:
        for i_row in range(len(occurrence)):
            value_index = occurrence.index(value)
            key = feature + '_' + str(i_row)
            if i_row == value_index:
                data[key].append(1)
            else:
                data[key].append(0)

    # Pop the feature
    data.pop(feature)

    return data


def normalize(data: dict):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :return data: dict[str, list], key is the column name, value is its normalized data
    """
    for key in data:
        if max(data[key]) == 1:
            pass
        else:
            data[key] = list(((ele - min(data[key])) / (max(data[key]) - min(data[key]))) for ele in data[key])
    return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, known as step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    # Step 1 : Initialize weights
    weights = {}  # feature => weight
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0

    # Step 2 : Start training
    w_keys = weights.keys()
    for epoch in range(num_epochs):
        for i in range(len(labels)):
            label = labels[i]
            phi_x = {}
            k = 0
            # Step 3 : Feature Extract
            for k_i, key in enumerate(keys):
                phi_x_key_i = inputs[key][i]
                phi_x[key] = phi_x_key_i
                k += weights[key]*phi_x_key_i
            if degree == 2:
                for k_i in range(len(keys)):
                    for k_j in range(k_i, len(keys)):
                        deg2_key = keys[k_i] + keys[k_j]
                        phi_x_key_i = inputs[keys[k_i]][i]*inputs[keys[k_j]][i]
                        phi_x[deg2_key] = phi_x_key_i
                        k += weights[deg2_key] * phi_x_key_i

            h = 1 / (1 + math.exp(-1 * k))  # sigmoid function

            # Step 4 : Update weights
            for w_key in w_keys:
                weights[w_key] = weights[w_key] - alpha * ((h - label) * phi_x[w_key])

    return weights
