import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        #Compute the difference between the user's ability (theta[u]) and the question's difficulty (beta[q])
        #output is a vector that is then summed to get a double result
        x = (theta[u] - beta[q]).sum()
        #maps the input value to a probability between 0 and 1, higher prob means better answer
        p_a = sigmoid(x)
        if data["is_correct"][i] == 1:
            log_lklihood += np.log(p_a)
        else:
            log_lklihood += np.log(1 - p_a)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # copy the vector beta & theta
    new_theta = np.copy(theta)
    new_beta = np.copy(beta)
    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        is_correct = data["is_correct"][i]
        new_theta[u]+=lr*(is_correct*(1-p_a)-(1-is_correct)*p_a)
        new_beta[q]-=lr*(is_correct*(1-p_a)-(1-is_correct)*p_a)
        # if data["is_correct"][i] == 1:
        #     # ability++, difficulty--
        #     # print(new_theta[u], " and ", new_beta[q])
        #     # print("x", x)
        #     # print("to")
        #     new_theta[u] += lr * (1 - p_a)
        #     new_beta[q] -= lr * (1 - p_a)
        #     # print(new_theta[u], " and ", new_beta[q])
        #     # print("+++++++++++++")
        # else:
        #     # print(new_theta[u], " and ", new_beta[q])
        #     # print("x", x)
        #     # print("dec")
        #     new_theta[u] -= lr * p_a
        #     new_beta[q] += lr * p_a
        #     # print(new_theta[u], " and ", new_beta[q])
        #     # print("-------------")
    return new_theta, new_beta

   
def bootstrap_sample(data, n_samples):
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

def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    # theta[u] represents the ability of user u that influences their performance
    # theta = np.zeros(num_users) # return an array filled of zeros || num_users = maxID+1 of the training data
    # # each element beta[q] represents the difficulty of question q.
    # beta = np.zeros(num_questions)
    theta = np.random.normal(0, 0.1, num_users)
    beta = np.random.normal(0, 0.1, num_questions)
    val_acc_lst = []
    iter=0
    for i in range(iterations):
        new_theta, new_beta = update_theta_beta(data, lr, theta, beta)
        # This checks if the current beta/theta values are close to the updated new_beta/new_theta values
        if np.allclose(theta, new_theta) and np.allclose(beta, new_beta):
            print("Converged at iteration", i)
            break
        iter+=1
        theta = new_theta
        beta = new_beta
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("Iteration",iter ," NLLK: {} \t Score: {}".format(neg_lld, score))
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    # iterate over the data["question_id"] using enumerate to get both index "i" and id "q" of the question
    for i, q in enumerate(data["question_id"]):
        #get the user_id in the same row as the question_id retrieved
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        # append to the predictions if correct (p_a>=0.5)
        pred.append(p_a >= 0.5)
    # the total number of right predictions divided by the total number of correct answers
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def predict(user_id, question_id, theta, beta):
    user_ability = theta[user_id]
    question_difficulty = beta[question_id]
    probability_correct = 1 / (1 + np.exp(-(user_ability - question_difficulty)))
    print("Student",user_id,"ability:",user_ability," Question",question_id,"difficulty:", question_difficulty,"Propability:" ,probability_correct)
    return probability_correct

def plot_probability(theta, beta, question_ids):
    probabilities = []
    sorted_theta = np.copy(theta)
    sorted_theta.sort()
    for question_id in question_ids:
        probability = 1 / (1 + np.exp(-(sorted_theta - beta[question_id])))
        probabilities.append(probability)

    # Plot the probabilities
    plt.figure(figsize=(10, 6))
    for i, probability in enumerate(probabilities):
        plt.plot(sorted_theta, probability, label=f"Question {question_ids[i]}")
    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs. Student Ability")
    plt.legend()
    plt.show()
    
    
def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.005
    iterations = 10
    n_bootstraps = 3
    all_theta = []
    all_beta = []
    all_val_acc_lst =[]
    val_acc_lst = 0
    all_test_accuracy = []
    test_accuracy = 0
    new_theta = []
    new_beta = []
    for i in range(n_bootstraps):
        bootstrapped_data = bootstrap_sample(train_data, len(train_data["is_correct"]))
        theta, beta, bs_val_acc_lst = irt(bootstrapped_data, val_data, lr, iterations)
        all_theta.append(theta)
        all_beta.append(beta)
        all_val_acc_lst.append(bs_val_acc_lst[-1])
        bs_test_accuracy = evaluate(test_data, theta, beta)
        all_test_accuracy.append(bs_test_accuracy)
        print(bs_val_acc_lst[-1])
        print(bs_test_accuracy)
        new_theta=np.mean(all_theta, axis=0)
        new_beta=np.mean(all_beta, axis=0)

    val_acc_lst = np.mean(all_val_acc_lst)
    print("Validation Accuracy:", val_acc_lst)
    test_accuracy=np.mean(all_test_accuracy)
    print("Test Accuracy:", test_accuracy)
    predict(1,1,new_theta,new_beta)
    predict(1,1046,new_theta,new_beta)
    predict(1,1444,new_theta,new_beta)
    # the first, the hardest, and the easiest questions
    question_ids = [1, 1046, 1444]
    print("Total questions:",len(beta),"Total students:",len(theta), "Number of iterations:", iterations)
    plot_probability(theta=theta, beta=beta, question_ids=question_ids)
    


if __name__ == "__main__":
    main()
