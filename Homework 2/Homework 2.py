################
################
# Q1
################
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
import scipy.stats


# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    """Read train data and return dataframe

    Args:
        filename (str): name of the file to read from

    Returns:
        pd.DataFrame: data read from file as data frame
    """
    return pd.read_csv(filename)


def read_test_data(filename: str) -> pd.DataFrame:
    '''
        read test data and return dataframe
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Implement LinearRegression class


class LinearRegression:
    def __init__(self, learning_rate=0.00001, iterations=30):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # weight initialization
        # YOUR CODE HERE

        # YOUR CODE HERE

        # data
        # YOUR CODE HERE
        # YOUR CODE HERE

        # gradient descent learning
        # YOUR CODE HERE
        # YOUR CODE HERE
        pass

        # Helper function to update weights in gradient descent
    def update_weights(self):
        # predict on data and calculate gradients
        # YOUR CODE HERE
        # YOUR CODE HERE

        # update weights
        # YOUR CODE HERE
        # YOUR CODE HERE
        pass

        # Hypothetical function  h( x )
    def predict(self, X):
        # YOUR CODE HERE
        # YOUR CODE HERE

        ########################
        ## Your Solution Here ##
        ########################
        pass

# Build your model


def build_model(train_X: np.array, train_y: np.array):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Make predictions with test set


def pred_func(model, X_test):
    '''
        return numpy array comprising of prediction on test set using the model
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Calculate and print the mean square error of your prediction


def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

################
################
# Q2
################
################

# Download and read the data.


def read_training_data(filename: str) -> tuple:
    '''
        read train data into a dataframe df1, store the top 10 entries of the dataframe in df2
        and return a tuple of the form (df1, df2, shape of df1)   
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Prepare your input data and labels


def data_clean(df_train: pd.DataFrame) -> tuple:
    '''
        check for any missing values in the data and store the missing values in series s, drop the entries corresponding 
        to the missing values and store dataframe in df_train and return a tuple in the form: (s, df_train)
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def feature_extract(df_train: pd.DataFrame) -> tuple:
    '''
        New League is the label column.
        Separate the data from labels.
        return a tuple of the form: (features(dtype: pandas.core.frame.DataFrame), label(dtype: pandas.core.series.Series))
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def data_preprocess(feature: pd.DataFrame) -> pd.DataFrame:
    '''
        Separate numerical columns from nonnumerical columns. (In pandas, check out .select dtypes(exclude = ['int64', 'float64']) and .select dtypes(
        include = ['int64', 'float64']). Afterwards, use get dummies for transforming to categorical. Then concat both parts (pd.concat()).
        and return the concatenated dataframe.
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def label_transform(labels: pd.Series) -> pd.Series:
    '''
        Transform the labels into numerical format and return the labels
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass

################
################
# Q3
################
################


def data_split(features: pd.DataFrame, label: pd.Series, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set using the given random state
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, max_iter=1000000):
    '''
        Instantiate an object of LogisticRegression class, train the model object
        use provided max_iterations for training logistic model
        using training data and return the model object
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return the tuple consisting the coefficients for each feature for Linear Regression 
        and Logistic Regression Models respectively
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def linear_pred_and_area_under_curve(linear_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def logistic_pred_and_area_under_curve(logistic_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [log_reg_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def optimal_thresholds(linear_threshold: np.ndarray, linear_reg_fpr: np.ndarray, linear_reg_tpr: np.ndarray, log_threshold: np.ndarray, log_reg_fpr: np.ndarray, log_reg_tpr: np.ndarray) -> Tuple[float, float]:
    '''
        return the tuple consisting the thresholds of Linear Regression and Logistic Regression Models respectively
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def stratified_k_fold_cross_validation(num_of_folds: int, shuffle: True, features: pd.DataFrame, label: pd.Series):
    '''
        split the data into 5 groups. Checkout StratifiedKFold in scikit-learn
    '''

    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_test_folds(skf, num_of_folds: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    '''
        train and test in for loop with different training and test sets obatined from skf. 
        use a PENALTY of 12 for logitic regression model for training
        find features in each fold and store them in features_count array.
        populate auc_log and auc_linear arrays with roc_auc_score of each set trained on logistic regression and linear regression models respectively.
        populate f1_dict['log_reg'] and f1_dict['linear_reg'] arrays with f1_score of trained logistic and linear regression models on each set
        return features_count, auc_log, auc_linear, f1_dict dictionary
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def is_features_count_changed(features_count: np.array) -> bool:
    '''
       compare number of features in each fold (features_count array's each element)
       return true if features count doesn't change in each fold. else return false
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


def mean_confidence_interval(data: np.array, confidence=0.95) -> Tuple[float, float, float]:
    '''
        To calculate mean and confidence interval, in scipy checkout .sem to find standard error of the mean of given data (AUROCs/ f1 scores of each model, linear and logistic trained on all sets). 
        Then compute Percent Point Function available in scipy and mutiply it with standard error calculated earlier to calculate h. 
        The required interval is from mean-h to mean+h
        return the tuple consisting of mean, mean -h, mean+h
    '''
    ########################
    ## Your Solution Here ##
    ########################
    pass


if __name__ == "__main__":
    ################
    ################
    # Q1
    ################
    ################
    data_path_train = "/path/train.csv"
    data_path_test = "/path/test.csv"
    df_train, df_test = read_train_data(
        data_path_train), read_test_data(data_path_test)

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    model = build_model(train_X, train_y)

    # Make prediction with test set
    # YOUR CODE HERE
    preds = pred_func(model, test_X)
    # YOUR CODE HERE

    # Calculate and print the mean square error of your prediction
    # YOUR CODE HERE
    mean_square_error = MSE(test_y, preds)
    # YOUR CODE HERE

    # plot your prediction and labels, you can save the plot and add in the report
    # YOUR CODE HERE

    plt.plot(test_y, label='label')
    plt.plot(preds, label='pred')
    plt.legend()
    plt.show()
    # YOUR CODE HERE

    ################
    ################
    # Q2
    ################
    ################
    # YOUR CODE HERE
    data_path_training = "/path/Hitters.csv"
    # YOUR CODE HERE

    train_df, df2, df_train_shape = read_training_data(data_path_training)
    s, df_train_mod = data_clean(train_df)
    features, label = feature_extract(df_train_mod)
    final_features = data_preprocess(features)
    final_label = label_transform(label)

    ################
    ################
    # Q3
    ################
    ################

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {'log_reg': [], 'linear_reg': []}
    is_features_count_changed = True

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(
        linear_model, logistic_model)

    print(linear_coef)
    print(logistic_coef)

    linear_y_pred, linear_reg_fpr, linear_reg_tpr, linear_reg_area_under_curve, linear_threshold = linear_pred_and_area_under_curve(
        linear_model, X_test, y_test)

    log_y_pred, log_reg_fpr, log_reg_tpr, log_reg_area_under_curve, log_threshold = logistic_pred_and_area_under_curve(
        logistic_model, X_test, y_test)

    # YOUR CODE HERE
    # plot your prediction and labels
    plt.plot(log_reg_fpr, log_reg_fpr, label='logistic')
    plt.plot(linear_reg_fpr, linear_reg_tpr, label='linear')
    plt.legend()
    plt.show()

    linear_threshod, linear_threshod = optimal_thresholds(
        y_test, linear_y_pred, log_y_pred, linear_threshold, log_threshold)

    skf = stratified_k_fold_cross_validation(
        num_of_folds, final_features, final_label)
    features_count, auc_log, auc_linear, f1_dict = train_test_folds(
        skf, num_of_folds)

    print("Does features change in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    is_features_count_changed = is_features_count_changed(features_count)

    linear_threshold, log_threshold = optimal_thresholds(
        linear_threshold, linear_reg_fpr, linear_reg_tpr, log_threshold, log_reg_fpr, log_reg_tpr)
    print(is_features_count_changed)

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = 0, 0, 0
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = 0, 0, 0

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = 0, 0, 0
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = 0, 0, 0

    # Find mean and 95% confidence interval for the AUROCs for each model and populate the above variables accordingly
    # Hint: use mean_confidence_interval function and pass roc_auc_scores of each fold for both models (ex: auc_log)
    # Find mean and 95% confidence interval for the f1 score for each model.

    mean_confidence_interval(auc_log)
    mean_confidence_interval(auc_linear)
    mean_confidence_interval(f1_dict['log_reg'])
    mean_confidence_interval(f1_dict['linear_reg'])
