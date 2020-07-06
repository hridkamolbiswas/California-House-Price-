from sffs_class import SFFS
from monte_carlo_split import monte_carlo_split
from helpers import rmse, visualize_reults
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def model_results(METHOD, N_SPLITS_evaluation, X, y, PERCENTAGE):

    counter = 1
    error_array_test = []
    error_array_train = []

    split_dict = monte_carlo_split(N_SPLITS_evaluation, X, y, PERCENTAGE)

    for item in split_dict:
        #print(f"Split number = {item}")
        x_train = split_dict[item]['x_train']
        y_train = split_dict[item]['y_train']
        x_test = split_dict[item]['x_test']
        y_test = split_dict[item]['y_test']

        model_train = METHOD.fit(x_train, y_train)
        predictions = model_train.predict(x_test)

        test_error = rmse(predictions.flatten(), y_test)
        train_error = rmse(METHOD.predict(x_train).flatten(), y_train)

        error_array_test.append(test_error)
        error_array_train.append(train_error)

    #print(f"{error_array_test}      {np.mean(error_array_test)}")   
    mean_error = np.mean(error_array_test)
    return mean_error 


def data_transformation(df: pd.DataFrame, selected_index: list):

    transformed_df = df.iloc[:, selected_index]
    return transformed_df

def best_model_results(METHOD, X, y, PERCENTAGE, final_features_index, degree):
    split_dict = monte_carlo_split(1, X, y, PERCENTAGE)
    x_train = split_dict[1]['x_train']
    y_train = split_dict[1]['y_train']
    x_test = split_dict[1]['x_test']
    y_test = split_dict[1]['y_test']

    transformed_x_train = data_transformation(x_train, final_features_index)
    transformed_x_test = data_transformation(x_test, final_features_index)

    model_train = METHOD.fit(transformed_x_train, y_train)
    #save the trained model as pickle
    pickle.dump(model_train, open(f"./model_pickle/trained_model_using_moment_{degree}.pickle", 'wb'))

    predictions = model_train.predict(transformed_x_test)
    test_error = rmse(predictions.flatten(), y_test)
    #print(round(test_error,4))

    visualize_reults(y_test, predictions, degree)

    return round(test_error,4)




    


