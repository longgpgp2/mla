import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix

# if the code is not running for you, try uncomment the line below (especially if you are using a Mac computer)
# from starter_code.utils import load_train_sparse, load_valid_csv, load_public_test_csv

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

def bootstrap_sample(data, n_samples):
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


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.

        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.k = k

        self.encoder = torch.nn.Sequential(
            self.g,
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            self.h,
            torch.nn.Sigmoid()
        )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        # reference: https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def sigmoid(x):
    return 1 / (1 + torch.exp(-1 * x))


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # reference: scipy.sparse.csc_matrix documentation
    valid_sparse = csc_matrix((valid_data["is_correct"], (valid_data["user_id"], valid_data["question_id"])),
                              shape=train_data.shape, dtype=np.int32).toarray()
    valid_sparse = torch.FloatTensor(valid_sparse)

    # count the number of non-zero entries, will be used to normalizing the squared loss
    N_valid = np.count_nonzero(valid_sparse)
    N_train = np.count_nonzero(zero_train_data)

    valid_acc_list = []
    train_cost_list = []
    valid_cost_list = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)  # Shape: [1, 1774]
            target_train = inputs.clone()
            optimizer.zero_grad()
            output = model(inputs)

            # Create a mask that matches the shape of the output
            nan_mask_train = np.isnan(train_data[user_id].numpy())  # Shape: [1774]
            
            # Ensure the mask is applied correctly
            target_train[0][nan_mask_train] = output[0][nan_mask_train]

            loss = torch.sum(
                (output - target_train) ** 2) + 0.5 * lamb * model.get_weight_norm()  # with L2 regularization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        loss_valid = valid_loss(model, zero_train_data, valid_data)
        valid_cost_list.append(loss_valid)
        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_cost_list.append(train_loss)
        valid_acc_list.append(valid_acc)

        print("Epoch: {epoch}, Training loss: {train_loss}, "
              "Validation Loss: {valid_loss}, Valid Accuracy: {acc}".format(epoch=epoch,
                                                                            train_loss=train_loss,
                                                                            valid_loss=loss_valid, acc=valid_acc))
    epochs = np.arange(0, num_epoch, 1)
    plt.plot(epochs, np.array(valid_cost_list) / N_valid, label="Average Validation Loss")
    plt.plot(epochs, np.array(train_cost_list) / N_train, label="Average Training Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Training and Validation loss")
    plt.legend()
    plt.show()
    plot(epochs, valid_acc_list, "Validation accuracy", "Validation accuracy vs Number of epochs")
    plot(epochs, train_cost_list, "Training loss", "Training loss vs Number of epochs")
    return valid_acc_list[-1]  # returns final validation accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def plot(x, y, ylabel, title, xlabel="Number of Epochs"):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        # print(output)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def valid_loss(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()
    loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item()
        loss += (guess - valid_data["is_correct"][i]) ** 2
    return loss

def calculate_average_accuracy(val_acc_list, test_acc_list):
    """ Calculate and return the average validation and test accuracy.

    :param val_acc_list: List of validation accuracies
    :param test_acc_list: List of test accuracies
    :return: Tuple of average validation accuracy and average test accuracy
    """
    avg_val_acc = sum(val_acc_list) / len(val_acc_list) if val_acc_list else 0
    avg_test_acc = sum(test_acc_list) / len(test_acc_list) if test_acc_list else 0
    return avg_val_acc, avg_test_acc
def predictneural(models, test_data, zero_train_data):
    """ Predict whether the student will answer a question correctly using multiple models.

    :param models: List of trained models
    :param test_data: A dictionary {user_id: list, question_id: list}
    :param zero_train_data: 2D FloatTensor
    :return: Final conclusion ("Correct" or "Incorrect")
    """
    model_outputs = []  # To store predictions from each model

    for model in models:
        model.eval()  # Set the model to evaluation mode
        predictions = []

        for i, u in enumerate(test_data["user_id"]):
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            output = model(inputs)
            # Get the predicted probability for the question
            guess = output[0][test_data["question_id"][i]].item()
            predictions.append(1 if guess >= 0.5 else 0)  # Store binary prediction

        model_outputs.append(predictions)
    
    # Aggregate predictions: Average the probabilities
    all_predictions = []
    num_models = len(models)

    for i in range(len(test_data["user_id"])):
        # Calculate the average prediction
        avg_prediction = sum(model_outputs[j][i] for j in range(num_models)) / num_models
        conclusion = "Correct" if avg_prediction >= 0.5 else "Incorrect"
        all_predictions.append(conclusion)

    # Aggregate predictions: Count "Correct" and "Incorrect"
    correct_count = 0
    incorrect_count = 0
    num_models = len(models)

    for i in range(len(test_data["user_id"])):
        # Calculate the number of correct predictions
        vote = sum(model_outputs[j][i] for j in range(num_models))
        if vote > num_models / 2:
            correct_count += 1
        else:
            incorrect_count += 1

    # Determine the final conclusion
    final_conclusion = "Correct" if correct_count > incorrect_count else "Incorrect"

    return final_conclusion
    # return all_predictions,final_conclusion


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    # Set model hyperparameters.
    k = 50  # Example value for the latent dimension
    lr = 0.005  # Learning rate
    num_epoch = 10  # Number of iterations
    n_bootstraps = 3  # Number of bootstraps
    all_val_acc_lst = []
    all_test_accuracy = []

    for i in range(n_bootstraps):
        # Generate a bootstrapped sample from the training data
        bootstrapped_data = bootstrap_sample({
            "user_id": zero_train_matrix.numpy(),
            "question_id": train_matrix.numpy(),
            "is_correct": train_matrix.numpy()  # Assuming is_correct is in train_matrix
        }, len(train_matrix))

        # Initialize the model
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

        # Train the model on the bootstrapped data
        final_valid_acc = train(model, lr, lamb=0.01, train_data=train_matrix, 
                                zero_train_data=zero_train_matrix, valid_data=valid_data, 
                                num_epoch=num_epoch)

        # Evaluate the model on the test data
        test_acc = evaluate(model, zero_train_matrix, test_data)
        all_val_acc_lst.append(final_valid_acc)
        all_test_accuracy.append(test_acc)

        print(f"Bootstrap {i + 1}: Validation Accuracy: {final_valid_acc}, Test Accuracy: {test_acc}")

    # Calculate and print average accuracies
    avg_val_acc, avg_test_acc = calculate_average_accuracy(all_val_acc_lst, all_test_accuracy)
    print(f"Average Validation Accuracy: {avg_val_acc}, Average Test Accuracy: {avg_test_acc}")

if __name__ == "__main__":
    main()
