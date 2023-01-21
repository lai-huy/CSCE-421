#!/usr/bin/env python
# coding: utf-8

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def read_data(filename: str) -> pd.DataFrame:
    """This function reads from a csv file and converts its contents into a pandas DataFrame.

    Args:
        filename (str): Filename of the csv file to read from

    Returns:
        pd.DataFrame: the csv file read as a pandas DataFrame
    """
    return pd.read_csv(filename)


def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """This function determines the shape, or dimensions, of the inputed datafile

    Args:
        filename (str): Filename of the csv file to read from

    Returns:
        Tuple[int, int]: the dimension of the datafile
    """
    return df.shape


# Extract features "Lag1", "Lag2", and label "Direction"

# Question 8 sub 3


def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data: pd.DataFrame = pd.DataFrame()
    data["Lag1"] = df["Lag1"]
    data["Lag2"] = df["Lag2"]
    return data, pd.Series(df["Direction"])


# Split the data into a train/test split

# Question 8 sub 4


def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits data into train and test data

    Args:
        features (pd.DataFrame): feature data set
        label (pd.Series): label data set
        test_size (float): percentage of the data to go into the train and test data

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, y_train, x_test, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=test_size, random_state=0)
    return x_train, y_train, x_test, y_test


# Write a function that returns score on test set with KNNs
# (use KNeighborsClassifier class)

# Question 8 sub 5


def knn_test_score(
    n_neighbors: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    classifier: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    score: float = 0
    for test, pred in zip(y_test, y_pred):
        if test == pred:
            score += 1

    return score / len(y_test)


# Apply k-NN to a list of data
# You can use previously used functions (make sure they are correct)

# Question 8 sub 6


def knn_evaluate_with_neighbours(
    n_neighbors_min: int,
    n_neighbors_max: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> List[float]:
    result: List[float] = []
    for n_neighbors in range(n_neighbors_min, n_neighbors_max + 1):
        result.append(knn_test_score(n_neighbors,
                                     x_train, y_train, x_test, y_test))

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = read_data("./Smarket.csv")
    # assert on df
    shape = get_df_shape(df)
    # assert on shape
    features, label = extract_features_label(df)

    x_train, y_train, x_test, y_test = data_split(features, label, 0.33)

    print(knn_test_score(1, x_train, y_train, x_test, y_test))
    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    plt.plot(range(1, 11), acc)
    plt.show()
