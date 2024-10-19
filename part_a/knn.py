import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
     # Impute missing values using KNN based on user similarity (rows)
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)

  # Evaluate accuracy on validation data using your provided function
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc




def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
  # Transpose the matrix to find similarities between questions (columns)
    transposed_matrix = matrix.T

    # Create a KNN imputer
    nbrs = KNNImputer(n_neighbors=k)

    # Fit the imputer to the transposed matrix
    nbrs.fit(transposed_matrix)

    # Transform the transposed matrix to fill missing values
    imputed_transposed_matrix = nbrs.transform(transposed_matrix)

    # Transpose back to original orientation
    imputed_matrix = imputed_transposed_matrix.T

    # Evaluate accuracy on validation data
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def predict(matrix, k, user_based=True):
    """ Predict the probability of correct answers in the matrix using k-Nearest Neighbors.

    :param matrix: 2D sparse matrix
    :param k: int
    :param user_based: bool, if True use user-based KNN, else use item-based KNN
    :return: 2D numpy array with predicted probabilities
    """
    nbrs = KNNImputer(n_neighbors=k)
    
    if user_based:
        imputed_matrix = nbrs.fit_transform(matrix)
    else:
        imputed_matrix = nbrs.fit_transform(matrix.T).T

    return imputed_matrix




def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    
    # List of k values to try
    k_values = [1, 6, 11, 16, 21, 26]
    user_based_accuracies = []
    item_based_accuracies = []

    # User-based KNN
    for k in k_values:
      acc = knn_impute_by_user(sparse_matrix.copy(), val_data.copy(), k)
      user_based_accuracies.append(acc)   
      print("Validation Accuracy (User-based, k={}): {}".format(k, acc))
    print()
  # Item-based KNN
    for k in k_values:
      acc = knn_impute_by_item(sparse_matrix.copy(), val_data.copy(), k)
      item_based_accuracies.append(acc)
      print("Validation Accuracy (Item-based, k={}): {}".format(k, acc))
    print()
  # Plot and compare accuracies (implementation omitted for brevity)
  # You can use libraries like matplotlib or seaborn to create the plot

  # Choose k with best validation accuracy for user-based and item-based
     # Determine the best k and its accuracy for user-based and item-based approaches
    best_k_user = k_values[user_based_accuracies.index(max(user_based_accuracies))]
    best_acc_user = max(user_based_accuracies)
    print("Best validation accuracy for user-based KNN with k={}: {}".format(best_k_user, best_acc_user))

    best_k_item = k_values[item_based_accuracies.index(max(item_based_accuracies))]
    best_acc_item = max(item_based_accuracies)
    print("Best validation accuracy for item-based KNN with k={}: {}".format(best_k_item, best_acc_item))

    # Test accuracy with the best k values
    test_acc_user = knn_impute_by_user(sparse_matrix.copy(), test_data.copy(), best_k_user)
    print("Test accuracy for user-based KNN with k={}: {}".format(best_k_user, test_acc_user))

    test_acc_item = knn_impute_by_item(sparse_matrix.copy(), test_data.copy(), best_k_item)
    print("Test accuracy for item-based KNN with k={}: {}".format(best_k_item, test_acc_item))
    
    # Create the plot
    plt.scatter(k_values, user_based_accuracies, label='User-based KNN', marker='o')  
    plt.scatter(k_values, item_based_accuracies, label='Item-based KNN', marker='o')


    # Add labels and title
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy and k')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

