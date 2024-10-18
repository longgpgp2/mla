# TODO: complete this file.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__), '..')))
import numpy as np
from utils import * 
from knn import knn_impute_by_item, knn_impute_by_user

def bootstrap_sample(matrix, valid_data):
    """ Generate a bootstrapped sample from the dataset. """
    n_samples = matrix.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrapped_matrix = matrix[indices, :]

    bootstrapped_valid_data = {
        "user_id": [valid_data["user_id"][i] for i in indices],
        "question_id": [valid_data["question_id"][i] for i in indices],
        "is_correct": [valid_data["is_correct"][i] for i in indices]
    }

    return bootstrapped_matrix, bootstrapped_valid_data

def bagging_knn_impute(matrix, valid_data, k, method="user"):
    """ Apply bagging to KNN imputation by generating multiple bootstrapped datasets and averaging the results. """
    if method == "user":
        acc = knn_impute_by_user(matrix, valid_data, k)
    else:
        acc = knn_impute_by_item(matrix, valid_data, k)

    return acc

def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")

    k_values = [1, 6, 11, 16, 21, 26]
    n_bootstraps = 3  # Number of bootstrapped datasets

    user_based_accuracies = []
    item_based_accuracies = []

    # Create 3 main bootstrapped datasets
    bootstrapped_matrices = [bootstrap_sample(sparse_matrix, val_data)[0] for _ in range(n_bootstraps)]
    bootstrapped_valid_data = [bootstrap_sample(sparse_matrix, val_data)[1] for _ in range(n_bootstraps)]

    # Bagging for user-based KNN
    for i in range(n_bootstraps):
        bootstrap_accuracies = []
        for k in k_values:
            print(f"Running bagging for User-based KNN with k={k} on bootstrap {i+1}")
            acc = bagging_knn_impute(bootstrapped_matrices[i], bootstrapped_valid_data[i], k, method="user")
            bootstrap_accuracies.append(acc)
            print(f"Bagging Validation Accuracy on Bootstrap {i+1} (User-based, k={k}): {acc}")
            print()

        avg_bootstrap_accuracy = np.mean(bootstrap_accuracies)
        print(f"Average Accuracy for Bootstrap {i+1} (User-based): {avg_bootstrap_accuracy}")
        print()
        user_based_accuracies.append(avg_bootstrap_accuracy)

    # Bagging for item-based KNN
    for i in range(n_bootstraps):
        bootstrap_accuracies = []
        for k in k_values:
            print(f"Running bagging for Item-based KNN with k={k} on bootstrap {i+1}")
            acc = bagging_knn_impute(bootstrapped_matrices[i], bootstrapped_valid_data[i], k, method="item")
            bootstrap_accuracies.append(acc)
            print(f"Bagging Validation Accuracy on Bootstrap {i+1} (Item-based, k={k}): {acc}")
            print()

        avg_bootstrap_accuracy = np.mean(bootstrap_accuracies)
        print(f"Average Accuracy for Bootstrap {i+1} (Item-based): {avg_bootstrap_accuracy}")
        print()
        item_based_accuracies.append(avg_bootstrap_accuracy)

    # Calculate overall average accuracy
    avg_user_based_accuracy = np.mean(user_based_accuracies)
    avg_item_based_accuracy = np.mean(item_based_accuracies)

    print(f"Overall Average Accuracy (User-based): {avg_user_based_accuracy}")
    print(f"Overall Average Accuracy (Item-based): {avg_item_based_accuracy}")

if __name__ == "__main__":
    main()