from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os


def dataset(all=True):
    """
    TODO
    """
    cal_housing = fetch_california_housing()
    feature_df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    target = cal_housing.target

    
    if all:
        return feature_df, target
    else:
        return feature_df.iloc[:,0:3], target #only first 3 colms



def calculate_error(true_value, predicted_value, metric=None):
    """
    TODO
    """
    mae = sum(abs(predicted_value - true_value)) / len(true_value)
    return mae

def rmse(predicted_value, true_value):
    return np.sqrt(((predicted_value - true_value) ** 2).mean())


def visualize_reults(true_values, predicted_values, degree):
    n_samples = 500
    plt.figure(figsize=(12, 6))
    plt.plot(true_values[0:n_samples], label="True Values")
    plt.plot(predicted_values[0:n_samples], label="Predicted_values")
    plt.legend()
    #plt.tight_layout()
    plt.title(f"True Values and predicted values\n" +
    f"only {n_samples} are shown for better visualization\n"+
    f"Mean RMSE {round(rmse(predicted_values, true_values),4)} using moment = {degree}")
    plt.savefig(f"./images/image_for_moment_{degree}.png")
    #plt.show()


def N_features_vs_error(d : dict, degree):
    rearange_d={}
    for key, val in d.items():
        #print(key, val)
        rearange_d.update({len(val):key})

    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    plt.figure(figsize=(8,6))
    plt.bar(*zip(*rearange_d.items()), width=0.5)
    plt.title(f"NUmber of features vs the associated error for Moment = {degree}")
    plt.xlabel('Number of features')
    plt.ylabel('RMSE error')
    #plt.show()
    plt.savefig(f"./images/N_features_vs_Error_moment_{degree}.png")


def plot_moment_vs_best_features(information_desk):
    x = []
    y=[]
    z = []
    min_error = 1
    select_moment = 1
    for item in information_desk:
        x.append(f"Moment_{str(item)}") # moment
        y.append(information_desk[item]['min_error']) #error
        z.append(information_desk[item]['min_error_index']) #selected index

        moment = item
        error = information_desk[item]['min_error']
        index = information_desk[item]['min_error_index']

    
        if error < min_error:
            min_error = error
            min_index = index
            select_moment = moment

    plt.figure(figsize=(8,6))
    plt.bar(x,y, width=0.2, alpha= 0.5)    
    for i, j, k in zip(x,y,z):
        plt.text(i, j, k, horizontalalignment='center',verticalalignment='top', rotation=90)
    plt.title('Moments vs the best features index with minimum error', fontsize = 14)
    #plt.tight_layout()
    #plt.show()
    plt.savefig(f"./images/moment_vs_best_model.png")

    return [select_moment, min_index, min_error]


def delete_no_need_pkl_files(select_moment):
    """
    Keep the trained_pickle file that givis minimum to test the unseen data in future.
    delete all unnecessary files from the folder
    """
    for file in os.listdir('./model_pickle/'):
        #print(file.split('_'))
        if 'best' in file.split('_'):
            pass
        else:
            if str(select_moment)  in list(file):
                pass
            else: 
                os.remove('./model_pickle/'+file)
   




