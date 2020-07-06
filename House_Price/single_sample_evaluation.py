import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from polynomial_features import polynomial_features
from evaluation import data_transformation


trained_model = pickle.load(open('./model_pickle/trained_model_using_moment_2.pickle', 'rb'))
model_info = pickle.load(open('./model_pickle/best_model_info.pickle', 'rb'))
#print(model_info)

moment = model_info[0]
feature_index = model_info[1]
print(f"Model moment = {moment} and selected index = {feature_index}")

one_sample = [4.1739,	40.0,	4.510638,	1.089362,	667.0,	2.838298,	37.79,	-122.20]

x = polynomial_features(one_sample, moment)
x_transformed = data_transformation(x, feature_index)

output = trained_model.predict(x_transformed)
print(f"The predicted output  = {round(output[0],4)}")

