import numpy as np
import pandas as pd
import random


def monte_carlo_split(n_splits: int, df: pd.DataFrame, y: np.array, percentage: float = 0.2):
    """
    TODO
    """

    split_dict = {}
    counter = 1

    while counter <= n_splits:

        test_size = int(len(df) * percentage)
        train_size = int(len(df) * (1 - percentage))

        all_index = list(df.index)
        test_data_index = random.sample(all_index, test_size)

        x_train = df.drop(test_data_index)  # .reset_index(drop=True)
        x_test = df.loc[test_data_index]  # .reset_index(drop=True)

        y_train = y[x_train.index.values]
        y_test = y[test_data_index]

        split_dict.update({

            counter :{
                'x_train':x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test
            }
        })
        counter +=1

    return split_dict



if __name__=='__main__':

    df = pd.DataFrame(np.random.randint(0,10, size=(10,5)), columns = list('abcde'))
    y = np.random.randint(0,10, 10)
    print(df)
    print(y)
    n_splits = 2
    percentage = 0.2

    split_dict = monte_carlo_split(n_splits, df, y, percentage)

    for item in split_dict:
        print(f"Split number = {item}")
        x_train = split_dict[item]['x_train']
        y_train = split_dict[item]['y_train']
        x_test = split_dict[item]['x_test']
        y_test = split_dict[item]['y_test']

        print(f"{x_train} \n {y_train} \n {x_test} \n {y_test}")
        print('-'*50)

    # print(df)
    # print(y)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

