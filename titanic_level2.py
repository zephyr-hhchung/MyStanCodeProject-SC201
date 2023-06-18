"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be read into pandas
    :param mode: str, indicating the mode we are using (either Train or Test)
    :param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
                          (You will only use this when mode == 'Test')
    :return: Tuple(data, labels), if the mode is 'Train'
             data, if the mode is 'Test'
    """
    data = pd.read_csv(filename)

    if mode == 'Train':
        data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].dropna()
        # Changing 'male' to 1, 'female' to 0
        data.loc[data.Sex == 'female', 'Sex'] = 0
        data.loc[data.Sex == 'male', 'Sex'] = 1

        # Changing 'S' to 0, 'C' to 1, 'Q' to 2
        data.loc[data.Embarked == 'S', 'Embarked'] = 0
        data.loc[data.Embarked == 'C', 'Embarked'] = 1
        data.loc[data.Embarked == 'Q', 'Embarked'] = 2

        labels = data['Survived']
        data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

        return data, labels

    elif mode == 'Test':
        data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

        # Changing 'male' to 1, 'female' to 0
        data.loc[data.Sex == 'female', 'Sex'] = 0
        data.loc[data.Sex == 'male', 'Sex'] = 1

        # Changing 'S' to 0, 'C' to 1, 'Q' to 2
        data.loc[data.Embarked == 'S', 'Embarked'] = 0
        data.loc[data.Embarked == 'C', 'Embarked'] = 1
        data.loc[data.Embarked == 'Q', 'Embarked'] = 2
        data = data.fillna(round(training_data.mean(axis=0), 3))

        return data


def one_hot_encoding(data, feature):
    """
    :param data: DataFrame, key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: DataFrame, remove the feature column and add its one-hot encoding features
    """
    # Check how many unique values in the given data[feature]
    occurrence = []
    for value in data[feature]:
        if value not in occurrence:
            occurrence.append(value)
    occurrence = sorted(occurrence)
    # Create the new one-hot encoding dictionary keys
    for i_new_col in range(len(occurrence)):
        new_col_name = feature + '_' + str(i_new_col)
        data[new_col_name] = 0
        data.loc[data[feature] == occurrence[i_new_col], new_col_name] = 1
    # Pop the feature
    data.pop(feature)
    return data


def standardization(data, mode='Train'):
    """
    :param data: DataFrame, key is the column name, value is its data
    :param mode: str, indicating the mode we are using (either Train or Test)
    :return data: DataFrame, standardized features
    """
    normalizer = preprocessing.StandardScaler()
    data = normalizer.fit_transform(data)
    return data


def main():
    """
    You should call data_preprocess(), one_hot_encoding(), and
    standardization() on your training data. You should see ~80% accuracy
    on degree1; ~83% on degree2; ~87% on degree3.
    Please write down the accuracy for degree1, 2, and 3 respectively below
    (rounding accuracies to 8 decimals)
    TODO: real accuracy on degree1 -> 0.80196629
    TODO: real accuracy on degree2 -> 0.83707865
    TODO: real accuracy on degree3 -> 0.87640449
    """
    # Data processing
    train_data, labels = data_preprocess('titanic_data/train.csv')

    # One-hot encoding for training data
    train_data = one_hot_encoding(train_data, 'Sex')
    train_data = one_hot_encoding(train_data, 'Pclass')
    train_data = one_hot_encoding(train_data, 'Embarked')

    normalizer = preprocessing.StandardScaler()
    x_data = normalizer.fit_transform(train_data)

    #############################
    # Degree 1 Polynomial Model #
    #############################
    h = linear_model.LogisticRegression(max_iter=10000)
    classifier = h.fit(x_data, labels)
    acc = classifier.score(x_data, labels)
    print('Training Accuracy (deg 1):', acc)

    #############################
    # Degree 2 Polynomial Model #
    #############################
    poly_phi = preprocessing.PolynomialFeatures(degree=2)
    X_train = poly_phi.fit_transform(x_data)

    h = linear_model.LogisticRegression(max_iter=10000)
    classifier = h.fit(X_train, labels)
    acc = classifier.score(X_train, labels)
    print(f"Training accuracy (deg 2): {acc}")

    #############################
    # Degree 3 Polynomial Model #
    #############################
    poly_phi = preprocessing.PolynomialFeatures(degree=3)
    X_train = poly_phi.fit_transform(x_data)

    h = linear_model.LogisticRegression(max_iter=10000)
    classifier = h.fit(X_train, labels)
    acc = classifier.score(X_train, labels)
    print(f"Training accuracy (deg 3): {acc}")


if __name__ == '__main__':
    main()
