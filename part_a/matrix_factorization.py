import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from scipy.linalg import sqrtm
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    #compute the error
    error = c - np.dot(u[n], z[q])

    #update u and z using the gradient of the squared error loss
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]

    return u, z
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: (u, z, training_loss, validation_loss)
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    training_loss = []
    validation_loss = []

    for iteration in range(num_iteration):
        # Randomly select a pair (user_id, question_id)
        i = np.random.choice(len(train_data["question_id"]), 1)[0]
        c = train_data["is_correct"][i]
        n = train_data["user_id"][i]
        q = train_data["question_id"][i]

        # Compute the error
        error = c - np.dot(u[n], z[q])

        # Update u and z using the gradient of the squared error loss
        u[n] += lr * error * z[q]
        z[q] += lr * error * u[n]

        # Calculate and store the training and validation loss
        train_loss = squared_error_loss(train_data, u, z)
        val_loss = squared_error_loss(val_data, u, z)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)

        print(f"Iteration {iteration}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

    return u, z, training_loss, validation_loss


def calculate_accuracy(data, reconstructed_matrix):
    """ Calculate the accuracy of the reconstructed matrix on the given data.
    
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param reconstructed_matrix: 2D matrix
    :return: float
    """
    correct_predictions = 0
    total_predictions = len(data["is_correct"])
    
    for i in range(total_predictions):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        actual = data["is_correct"][i]
        predicted = reconstructed_matrix[user_id, question_id] >= 0.5  # Assuming threshold of 0.5
        
        if predicted == actual:
            correct_predictions += 1
    
    return correct_predictions / total_predictions

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # SVD: Try out at least 5 different k and select the best k using the validation set.
    best_k_svd = None
    best_val_accuracy_svd = 0
    k_values = [5, 10, 20, 50, 100]
    for k in k_values:
        reconstructed_matrix = svd_reconstruct(train_matrix, k)
        val_accuracy = evaluate_accuracy(reconstructed_matrix, val_data)
        if val_accuracy > best_val_accuracy_svd:
            best_val_accuracy_svd = val_accuracy
            best_k_svd = k

    print(f"Best k for SVD: {best_k_svd} with validation accuracy: {best_val_accuracy_svd}")
    test_accuracy_svd = evaluate_accuracy(svd_reconstruct(train_matrix, best_k_svd), test_data)
    print(f"Test accuracy for SVD with best k: {test_accuracy_svd}")

    # ALS: Try out at least 5 different k and select the best k using the validation set.
    best_k_als = None
    best_val_loss_als = float('inf')
    for k in k_values:
        u, z, train_loss, val_loss = als(train_data, val_data, k, lr=0.01, num_iteration=100)
        if val_loss[-1] < best_val_loss_als:
            best_val_loss_als = val_loss[-1]
            best_k_als = k

    print(f"Best k for ALS: {best_k_als} with validation loss: {best_val_loss_als}")

    # Compare SVD and ALS results
    print("Comparison of SVD and ALS:")
    print(f"SVD Best k: {best_k_svd}, Validation Accuracy: {best_val_accuracy_svd}, Test Accuracy: {test_accuracy_svd}")
    print(f"ALS Best k: {best_k_als}, Validation Loss: {best_val_loss_als}")

    # Discuss limitations
    print("Limitations of Matrix Factorization:")
    print("1. Handling of missing data: Both SVD and ALS require filling in missing values, which can introduce bias.")
    print("2. Linear assumptions: SVD assumes linear relationships, which may not capture complex interactions.")
    print("3. Hyperparameter tuning: ALS requires careful tuning of learning rate and number of iterations.")
    print("4. Scalability: Large datasets can be computationally expensive to factorize.")
    print("5. Cold start problem: New users or items with no interactions are difficult to handle.")

def evaluate_accuracy(reconstructed_matrix, data):
    correct_predictions = 0
    total_predictions = len(data["is_correct"])
    for i, q in enumerate(data["question_id"]):
        prediction = reconstructed_matrix[data["user_id"][i], q] >= 0.5
        if (prediction == data["is_correct"][i]):
            correct_predictions += 1
    return correct_predictions / total_predictions

if __name__ == "__main__":
    main()