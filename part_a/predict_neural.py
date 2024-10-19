import os
import sys
from ensemble_neural import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
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
    num_epoch = 3  # Number of iterations
    n_bootstraps = 3  # Number of bootstraps
    all_val_acc_lst = []
    all_test_accuracy = []

    models = []  # List to store trained models

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

        # Store the trained model
        models.append(model)

        print(f"Bootstrap {i + 1}: Validation Accuracy: {final_valid_acc}, Test Accuracy: {test_acc}")

    # Predict using the trained models
    final_prediction = predictneural(models, test_data, zero_train_matrix)
    print(f"Final Prediction: {final_prediction}")

    # Calculate and print average accuracies
    avg_val_acc, avg_test_acc = calculate_average_accuracy(all_val_acc_lst, all_test_accuracy)
    print(f"Average Validation Accuracy: {avg_val_acc}, Average Test Accuracy: {avg_test_acc}")

if __name__ == "__main__":
    main()