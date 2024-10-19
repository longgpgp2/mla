import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
import knn
import neural_network
from item_response import *
from sklearn.utils import resample
import matrix_factorization
import matplotlib.pyplot as plt
import predict_neural
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix

def bootstrap_neural_data(data, n_samples):
    """ Generate a bootstrapped sample from the dataset for neural network training.
    
    :param data: Original dataset
    :param n_samples: Number of samples to draw
    :return: Bootstrapped dataset
    """
    indices = np.random.choice(len(data["is_correct"]), size=n_samples, replace=True)
    bootstrapped_data = {
        "user_id": [data["user_id"][i] for i in indices],
        "question_id": [data["question_id"][i] for i in indices],
        "is_correct": [data["is_correct"][i] for i in indices]
    }
    # Convert bootstrapped data to PyTorch tensors for neural network compatibility
    bootstrapped_data["user_id"] = torch.tensor(np.array(bootstrapped_data["user_id"]))
    bootstrapped_data["question_id"] = torch.tensor(np.array(bootstrapped_data["question_id"]))
    bootstrapped_data["is_correct"] = torch.tensor(np.array(bootstrapped_data["is_correct"]), dtype=torch.float32)
    
    return bootstrapped_data

def bootstrap_training_data(data, n_samples):
    """ Generate a bootstrapped sample from the dataset.
    
    :param data: Original dataset
    :param n_samples: Number of samples to draw
    :return: Bootstrapped dataset
    """
    indices = np.random.choice(len(data["is_correct"]), size=n_samples, replace=True)
    bootstrapped_data = {
        "user_id": [data["user_id"][i] for i in indices],
        "question_id": [data["question_id"][i] for i in indices],
        "is_correct": [data["is_correct"][i] for i in indices]
    }
    return bootstrapped_data

import numpy as np

def bootstrap_sparse_matrix(sparse_matrix, n_samples=None):
    """ Create a bootstrapped dataset from the sparse matrix.

    :param sparse_matrix: scipy.sparse.csr_matrix, the input sparse matrix
    :param n_samples: int, number of samples to draw; if None, use the size of sparse_matrix
    :return: scipy.sparse.csr_matrix representing the bootstrapped dataset
    """
    if n_samples is None:
        n_samples = sparse_matrix.shape[0]
    bootstrapped_indices = np.random.choice(sparse_matrix.shape[0], size=n_samples, replace=True)
    bootstrapped_data = sparse_matrix[bootstrapped_indices]

    return bootstrapped_data

def train_irt(train_data, val_data, test_data, lr, iterations, ):
    print("training the IRT model:")
    n_bootstraps = 3
    all_theta = []
    all_beta = []
    irt_accuracies =[]
    irt_test_accuracies = []
    irt_test_accuracy = 0
    for i in range(n_bootstraps):
        bootstrapped_data = bootstrap_training_data(train_data, len(train_data["is_correct"]))
        theta, beta, bs_val_acc_lst = irt(bootstrapped_data, val_data, lr, iterations)
        all_theta.append(theta)
        all_beta.append(beta)
        irt_accuracies.append(bs_val_acc_lst[-1])
        bs_test_accuracy = evaluate(test_data, theta, beta)
        irt_test_accuracies.append(bs_test_accuracy)
        print(bs_val_acc_lst[-1])
        print(bs_test_accuracy)
    irt_theta=np.mean(all_theta, axis=0)
    irt_beta=np.mean(all_beta, axis=0)
    print("Finished training the IRT model.")
    irt_valid_acc = np.mean(irt_accuracies)
    print("IRT Validation Accuracy:", irt_valid_acc)
    irt_test_acc=np.mean(irt_test_accuracies)
    print("IRT Test Accuracy:", irt_test_acc)
    
    return irt_theta, irt_beta, irt_valid_acc, irt_test_acc

def train_knn(sparse_matrix, val_data, test_data):
    print("Training the KNN model:")
    n_bootstraps = 3
    k = 21
    user_based_valid_accuracies = []
    user_based_test_accuracies = []
    for i in range(n_bootstraps):
        bootstrapped_data = bootstrap_sparse_matrix(sparse_matrix)
        user_based_valid_accuracy = knn.knn_impute_by_user(bootstrapped_data, val_data, k)
        user_based_test_accuracy = knn.knn_impute_by_user(bootstrapped_data, test_data, k)
        user_based_valid_accuracies.append(user_based_valid_accuracy)
        user_based_test_accuracies.append(user_based_test_accuracy)
    knn_valid_acc = np.mean(user_based_valid_accuracies)
    knn_test_acc =  np.mean(user_based_test_accuracies)
    print("Finished training KNN model.")
    print("KNN Validation Accuracy:", knn_valid_acc, "KNN Test Accuracy:", knn_test_acc)
    knn_predicted_probabilities = knn.predict(sparse_matrix, k, user_based=True)
    return knn_predicted_probabilities, knn_valid_acc, knn_test_acc

def train_neural_network():
    print("Training the Neural Network model:")
    zero_train_matrix, train_matrix, valid_data, test_data = neural_network.load_data()
    # Set model hyperparameters.
    k = 50  # Example value for the latent dimension
    lr = 0.005  # Learning rate
    num_epoch = 4  # Number of iterations
    n_bootstraps = 3  # Number of bootstraps
    all_val_acc_lst = []
    all_test_accuracy = []
    models =[]

    for i in range(n_bootstraps):
        # Generate a bootstrapped sample from the training data
        bootstrapped_data = bootstrap_neural_data({
            "user_id": zero_train_matrix.numpy(),
            "question_id": train_matrix.numpy(),
            "is_correct": train_matrix.numpy()  # Assuming is_correct is in train_matrix
        }, len(train_matrix))

        # Initialize the model
        model = neural_network.AutoEncoder(num_question=train_matrix.shape[1], k=k)
        models.append(model)
        # Train the model on the bootstrapped data
        final_valid_acc = neural_network.train(model, lr, lamb=0.01, train_data=train_matrix, 
                                zero_train_data=zero_train_matrix, valid_data=valid_data, 
                                num_epoch=num_epoch)

        # Evaluate the model on the test data
        test_acc = neural_network.evaluate(model, zero_train_matrix, test_data)
        all_val_acc_lst.append(final_valid_acc)
        all_test_accuracy.append(test_acc)
        
        neural_valid_acc = np.mean(all_val_acc_lst)
        neural_test_acc = np.mean(all_test_accuracy)
        print(f"Bootstrap {i + 1}: Validation Accuracy: {final_valid_acc}, Test Accuracy: {test_acc}")
    print("Finished training Neural Network model.")
    return models, zero_train_matrix, neural_valid_acc, neural_test_acc
    # print("Average Validation Accuracy:", avg_val_acc, "Average Test Accuracy:", avg_test_acc)

    # Return the average accuracies
    # return avg_val_acc, avg_test_acc
def predict_ensemble(u, q, knn_predicted_probabilities, theta, beta, models, zero_train_matrix):
    knn_prediction = knn_predicted_probabilities[u,q]
    irt_prediction = predict(u, q, theta, beta)
    neural_prediction = predict_neural(models, zero_train_matrix, 1, 1046)
    ensemble_prediction = (knn_prediction+irt_prediction+ neural_prediction)/3
    print(ensemble_prediction)
    return ensemble_prediction

def predict_neural(models, zero_train_matrix, u, q):
    probs=[]
    for model in models:
        prob= neural_network.predict(model, zero_train_matrix, u, q)
        probs.append(prob)
    return np.mean(probs)


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.005
    iterations = 4

    irt_theta, irt_beta, irt_valid_acc, irt_test_acc = train_irt(train_data, val_data, test_data, lr, iterations)
    
    knn_predicted_probabilities, knn_valid_acc, knn_test_acc = train_knn(sparse_matrix, val_data, test_data)
    print(irt_test_acc, irt_valid_acc, knn_valid_acc, knn_test_acc)
    models, zero_train_matrix, neural_valid_acc, neural_test_acc = train_neural_network()
    es_valid_acc = (knn_valid_acc+irt_valid_acc+neural_valid_acc)/3
    es_test_acc = (knn_test_acc+irt_test_acc+neural_test_acc)/3
    print("Ensemble Validation Accuracy:", es_valid_acc, "Ensemble Test Accuracy:", es_test_acc)
    predict_ensemble(1, 1046, knn_predicted_probabilities, irt_theta, irt_beta, models, zero_train_matrix)
    # train_neural_network()
    

if __name__ == "__main__":
    main()