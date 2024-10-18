import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.impute import KNNImputer
from utils import *


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
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
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
    # Transpose the matrix to consider question similarity
    mat = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(mat)
    # Transpose back to original shape
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
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

    # Compute the validation accuracy for each k
    k_values = [1, 6, 11, 16, 21, 26]
    user_based_accuracies = []
    item_based_accuracies = []
    for k in k_values:
        user_based_accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        item_based_accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        user_based_accuracies.append(user_based_accuracy)
        item_based_accuracies.append(item_based_accuracy)

    # Plot the validation accuracy for each k
    import matplotlib.pyplot as plt
    plt.plot(k_values, user_based_accuracies, label='User -based')
    plt.plot(k_values, item_based_accuracies, label='Item-based')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('KNN Validation Accuracy')
    plt.legend()
    plt.show()

    # Pick k* with the best performance and report the test accuracy with the chosen k*
    best_k_user_based = k_values[np.argmax(user_based_accuracies)]
    best_k_item_based = k_values[np.argmax(item_based_accuracies)]
    print("Best k for user-based: {}".format(best_k_user_based))
    print("Best k for item-based: {}".format(best_k_item_based))
    test_accuracy_user_based = sparse_matrix_evaluate(test_data, knn_impute_by_user(sparse_matrix, test_data, best_k_user_based))
    test_accuracy_item_based = sparse_matrix_evaluate(test_data, knn_impute_by_item(sparse_matrix, test_data, best_k_item_based))
    print("Test Accuracy for user-based: {}".format(test_accuracy_user_based))
    print("Test Accuracy for item-based: {}".format(test_accuracy_item_based))


if __name__ == "__main__":
    main()