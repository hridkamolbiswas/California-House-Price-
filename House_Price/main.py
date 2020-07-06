from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import random
import os
import sys
import shutil
import pickle

from monte_carlo_split import monte_carlo_split
from helpers import (dataset, rmse, visualize_reults, N_features_vs_error,
                    plot_moment_vs_best_features, delete_no_need_pkl_files)
from get_best_n_feature_index import get_best_features_index
from sffs_class import SFFS
from create_folders import create_folders
from evaluation import model_results, data_transformation, best_model_results
from polynomial_features import polynomial_features


def main():

    """
    TODO
    """
    # define few necessarry variables
    random.seed(44)
    N_SPLITS_feature_selection = 2
    N_SPLITS_model_selection = 3

    PERCENTAGE = 0.2
    METHOD = LinearRegression(fit_intercept=False)

    #create folder to ave the images
    IMAGE_FOLDER = "images"
    PICKLE_FOLDER = "model_pickle"
    create_folders(folder_name=IMAGE_FOLDER)
    create_folders(folder_name=PICKLE_FOLDER)

    #use higher moments
    moment_list = [1,2]

    information_desk ={}
    summary = open("summary.txt", "w")
    for degree in moment_list:
        # get the dataset
        X, y = dataset(all=True)
        #do transformation
        X = polynomial_features(X, degree)


        feature_length =  len(X.columns.tolist())
        print(feature_length)
        
        information_dict = {}
        for n_best_features in tqdm(range(1,feature_length+1)):

            best_feature_combi = get_best_features_index(n_best_features, METHOD, N_SPLITS_feature_selection, X, y, PERCENTAGE)
            mean_error = model_results(METHOD, N_SPLITS_model_selection, X, y, PERCENTAGE)

            information_dict.update({mean_error: best_feature_combi})
            #print(f" Using degree: {degree} feature_size: {len(best_feature_combi)} features index : {best_feature_combi} Error : {mean_error}")
        minimum_error = min(information_dict.items())
        final_features_index_error = minimum_error[0]
        final_features_index = minimum_error[1]

        N_features_vs_error(information_dict, degree)
        best_model_results(METHOD, X, y, PERCENTAGE, final_features_index, degree)
    
        summary.write(f"FINAL --> Using degree: {degree} index : {final_features_index}, error: {final_features_index_error} \n") 
        print(f"FINAL --> Using degree: {degree} index : {final_features_index}, error: {final_features_index_error}")
        N_features_vs_error(information_dict, degree)
        best_model_results(METHOD, X, y, PERCENTAGE, final_features_index, degree)

        information_desk.update({
            degree : {
            'min_error': final_features_index_error,
            'min_error_index': final_features_index
            }
        })
    summary.close()

    #print(information_desk)
    best_model_info = plot_moment_vs_best_features(information_desk) #returne the best [moment, featute index, associated_error]
    print(best_model_info)
    #save this info as pickle
    with open('./model_pickle/best_model_info.pickle', 'wb') as f:
        pickle.dump(best_model_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    delete_no_need_pkl_files(best_model_info[0])

    

if __name__ == "__main__":
    main()
