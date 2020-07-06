from sffs_class import SFFS
from monte_carlo_split import monte_carlo_split

def get_best_features_index(n_best_features, METHOD, N_SPLITS, X, y, PERCENTAGE):
    split_dict = monte_carlo_split(N_SPLITS, X, y, PERCENTAGE)

    best_feature_combi_per_split = []
    for item in split_dict:
        #print(f"Split number = {item}")
        x_train = split_dict[item]['x_train']
        y_train = split_dict[item]['y_train']
        x_test = split_dict[item]['x_test']
        y_test = split_dict[item]['y_test']

        SFFS_object = SFFS(n_best_features, METHOD, x_train, y_train, x_test, y_test)
        selected_index = SFFS_object.find_best_features() #column index
        best_feature_combi_per_split.append(selected_index)

    #for different splits, we can get different best_features
    #there can be many ways to handle this issue
    #for the simplicity; we take the feature combination, that comes maximum times for N_splits (e.g., 20)
    best_combi = max(best_feature_combi_per_split,key=best_feature_combi_per_split.count)
    #print(f"\n using best {n_best_features} the combi = {best_combi} \n ")
    return best_combi