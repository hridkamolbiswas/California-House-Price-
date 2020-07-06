import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from helpers import rmse, dataset


class SFFS:
    def __init__(self, n_features, classifier, x_train, y_train, x_test, y_test):
        self.n_features = n_features
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_list = x_train.columns.tolist()
        self.data_row = x_train.shape[0]
        self.data_col = x_train.shape[1]

        self.train_input_data = x_train[[]]
        self.test_input_data = x_test[[]]
        self.train_output = y_train
        self.test_output = y_test
        self.selected_features_index_list = []
        self.ignoring_cols = []
        self.cadidates = []

    def inclusion_step(self, ignore_col=None):

        new_x_train = self.x_train.drop(self.x_train.columns[self.selected_features_index_list], axis=1)
        new_x_test = self.x_test.drop(self.x_test.columns[self.selected_features_index_list], axis=1)
        feature_list = new_x_train.columns.tolist()  # features of remaining df
        feature_list = [feature for feature in feature_list if feature not in self.ignoring_cols]

        if not feature_list:
            return self.selected_features_index_list, None

        for i, feature in enumerate(feature_list):
            current_index = self.x_train.columns.get_loc(feature)
            train_input = pd.concat([self.train_input_data, new_x_train[[feature]]], axis=1)
            test_input = pd.concat([self.test_input_data, new_x_test[[feature]]], axis=1)
            train_output = self.train_output
            test_output = self.test_output
            # print(f"train col size = {train_input.shape[1]} and test cols size ={test_input.shape[1]} and features = {train_input.columns.tolist()}")

            model_train = self.classifier.fit(train_input, train_output)
            predictions = self.classifier.predict(test_input)
            current_error = rmse(test_output, predictions)

            if i == 0:
                selected_feature_error = current_error
                selected_feature_index = current_index
                selected_feature_title = feature
            else:
                if current_error < selected_feature_error:
                    selected_feature_error = current_error
                    selected_feature_index = current_index
                    selected_feature_title = feature
                else:
                    pass

        # print(f"selected feature index = {selected_feature_index} and error = {selected_feature_error}")

        self.train_input_data = self.train_input_data.join(self.x_train[selected_feature_title])
        self.test_input_data = self.test_input_data.join(self.x_test[selected_feature_title])

        return selected_feature_index, selected_feature_error

    def exclusion_step(self, error):
        """
        TODO
        """
        if len(self.selected_features_index_list) < 2:
            pass
        else:
        # find less significatnt feature, index here for the subset, not for the original df
            (
                less_significant_feature_index,
                feature_title,
            ) = self.return_less_significant_feature()
            

            # remove less significant from the collected features
            rest_features_train = self.train_input_data.drop(
                self.train_input_data.columns[less_significant_feature_index], axis=1
            )
            rest_features_test = self.test_input_data.drop(
                self.test_input_data.columns[less_significant_feature_index], axis=1
            )

            model_train = self.classifier.fit(rest_features_train, self.train_output)
            predictions = self.classifier.predict(rest_features_test)
            current_error = rmse(predictions, self.test_output)
            if error is None:
                return 
            if (current_error < error) and (len(self.selected_features_index_list) > 1):
                #print(f"-->ERROR REDUCED for using {feature_title}")

                # remove from training_subset
                self.train_input_data = self.train_input_data.drop(
                    self.train_input_data.columns[less_significant_feature_index],
                    axis=1,
                )
                self.test_input_data = self.test_input_data.drop(
                    self.test_input_data.columns[less_significant_feature_index], axis=1
                )

                # need to know the feature name or index for original x_tain
                #get the index of less signi. feature in original x_train, because we need to remove it from there
                original_index = self.feature_list.index(feature_title)
                #remove the exact one instead of randomly first one
                self.selected_features_index_list.remove(original_index)
                self.ignoring_cols.append(feature_title)

                # call the exclusion function
                if len(self.selected_features_index_list) + len(self.ignoring_cols) < len(self.feature_list):
                    self.exclusion_step(error)


            else:
                pass
            
    def return_less_significant_feature(self):

        for i, feature in enumerate(self.train_input_data.columns.tolist()):
            current_index = self.train_input_data.columns.get_loc(feature)
            model_train = self.classifier.fit(
                self.train_input_data[[feature]], self.train_output
            )
            predictions = self.classifier.predict(self.test_input_data[[feature]])
            current_error = rmse(self.test_output, predictions)

            if i == 0:
                selected_feature_error = current_error
                selected_feature_index = current_index
                selected_feature_title = feature
            else:
                if current_error > selected_feature_error:
                    selected_feature_error = current_error
                    selected_feature_index = current_index
                    selected_feature_title = feature
                else:
                    pass
            #print(f"using index {current_index}  {feature}  error = {selected_feature_error}")
        return selected_feature_index, selected_feature_title

    def find_best_features(self):

        while len(self.selected_features_index_list) + len(self.ignoring_cols) < self.n_features:
            # inclusion step
            selected_feature_index, error = self.inclusion_step()
            self.selected_features_index_list.append(selected_feature_index)
            if self.n_features < len(self.feature_list):
                self.exclusion_step(error)
            

        return self.selected_features_index_list

    def get_selected_features_index(self):
        return self.selected_features_index_list

    def transform(self, df: pd.DataFrame):

        transformed_df = df.iloc[:, self.selected_features_index_list]
        return transformed_df


if __name__ == "__main__":

    X, y = dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=0
    )

    n_features = 8
    classifier = LinearRegression()

    sffs = SFFS(n_features, classifier, x_train, y_train, x_test, y_test)
    # sffs.find_best_features()
    selected_index = sffs.find_best_features()
    print(selected_index)

    x_train_transformed = sffs.transform(x_train)
    x_test_transformed = sffs.transform(x_test)
    print(x_train_transformed)
    print(sffs.get_selected_features_index())
