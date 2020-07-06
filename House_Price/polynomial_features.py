import pandas as pd
import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(df: pd.DataFrame, degree: int):
    """
    TODO
    """
    required_features = ['MedInc',
                        'HouseAge',
                        'AveRooms',
                        'AveBedrms',
                        'Population',
                        'AveOccup',
                        'Latitude',
                        'Longitude']

    if isinstance(df, pd.DataFrame):
        df.insert(0, "1", 1)
        feature_list = df.columns.tolist()

    if isinstance(df, list):
        df = np.array(df).reshape(1,-1)
        df = pd.DataFrame(df, columns=required_features)
        df.insert(0, "1", 1)

    feature_list = df.columns.tolist()

    feature_combinations = []
    sorted_features = sorted(feature_list)
    feature_combinations = list(combinations_with_replacement(sorted_features, degree))
    feature_combinations = [list(item) for item in feature_combinations]
    #print(len(feature_combinations))

    df_new = pd.DataFrame()
    for i, item in enumerate(feature_combinations):
        col_name = "".join(item)
        df_new[col_name] = df[item].prod(axis=1)
       
    return df_new


if __name__ == "__main__":
    degree = 1

    X = pd.DataFrame(
        {
            "A": [1, 5, 3, 4, 2],
            "B": [3, 2, 4, 3, 4],
            "C": [2, 2, 7, 3, 4],
            "D": [4, 3, 6, 12, 7],
        }
    )


    df = polynomial_features(X, degree)
    print(df.shape)
    print(df)


    one_sample = [4.1739,	40.0,	4.510638,	1.089362,	667.0,	2.838298,	37.79,	-122.20]

    x = polynomial_features(one_sample, degree)
    print(x)