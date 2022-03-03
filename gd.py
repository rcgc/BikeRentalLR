import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np


# https://towardsdatascience.com/implementing-gradient-descent-in-python-from-scratch-760a8556c31f
def plot_dataset(X, Y, figure_number):
    plt.figure(figure_number)
    plt.title('Bike Rental Prediction')
    plt.xlabel('Feeling Temperature (°C)')
    plt.ylabel('Rented bikes')

    plt.scatter(X, Y, alpha=0.3)

    plt.show(block=False)


def plot_linear_regression(bias, theta, figure_number):
    x = np.linspace(-3, 3, 100)
    y = theta*x+bias

    plt.figure(figure_number)
    plt.plot(x, y, '-r')
    plt.title('Bike Rental Prediction')
    plt.xlabel('Feeling Temperature (°C)')
    plt.ylabel('Rented bikes')

    plt.show(block=False)


def coefficient_of_regression(df_X_sample, df_Y_original, bias, theta, mean):
    Y_predict = list()
    for i in df_X_sample:
        Y_predict.append(bias + theta * i)

    Y_predict = np.array(Y_predict)

    mse = 0
    for i, j in zip(df_Y_original, Y_predict):
        mse = mse + (i - j) * (i - j)

    ss_res = 0
    for i in df_Y_original:
        ss_res = ss_res + abs(i - mean) * abs(i - mean)

    ss_tot = mse * 730

    R_square = 1 - (ss_res / ss_tot)

    return mse, R_square


def make_predictions(bias, theta, figure_number, mean, stddev):
    # X -> input, Y -> predicted
    X_queries = []
    Y_queries = []

    prediction_Y = 0
    option = 0

    while option == 0:
        print('0) Make paredictions')
        print('1) Exit')

        option = int(input())

        if option == 1:
            break

        n = int(input("How many integers : "))

        # Queries
        for i in range(0, n):
            ele = float(input())
            ele_scaled = (ele - mean)/stddev    # Scaling x inputted for predictions

            prediction_Y = theta * ele_scaled + bias
            print(ele, '->', prediction_Y)
            X_queries.append(ele_scaled)
            Y_queries.append(prediction_Y)

        plt.figure(figure_number)
        plt.scatter(X_queries, Y_queries, c='red', alpha=0.3)
        plt.show(block=False)

        X_queries.clear()
        Y_queries.clear()

def initialize():
    b = 0
    theta = 0
    return b, theta


def predict_Y(b, theta, X):
    return b + np.dot(X, theta)


def get_cost(Y, Y_hat):
    Y_resd = Y - Y_hat
    return np.sum(np.dot(Y_resd.T, Y_resd))/len(Y-Y_resd)


def update_theta(X, Y, Y_hat, b_0, theta_0, learning_rate):
    db = (np.sum(Y_hat-Y)*2)/len(Y)
    dw = (np.dot((Y_hat-Y), X)*2)/len(Y)

    b_1 = b_0 - learning_rate*db
    theta_1 = theta_0-learning_rate*dw

    return b_1, theta_1


def run_gradient_descent(X, Y, alpha, num_iterations):
    b, theta = initialize()
    iter_num = 0
    gd_iterations_df = pd.DataFrame(columns=['iteration', 'cost'])
    result_idx = 0

    for each_iter in range(num_iterations):
        Y_hat = predict_Y(b, theta, X)
        this_cost = get_cost(Y, Y_hat)
        prev_b = b
        prev_theta = theta
        b, theta = update_theta(X, Y, Y_hat, prev_b, prev_theta, alpha)

        gd_iterations_df.loc[result_idx] = [iter_num, this_cost]

        result_idx = result_idx + 1
        iter_num = iter_num + 1
    return gd_iterations_df, b, theta


def start(df_X, df_Y, learning_rate, num_iterations):
    b, theta = initialize()
    print("Bias_0: ", b, " Theta_0: ", theta)

    X = df_X.to_numpy()
    Y = df_Y.to_numpy()

    gd_iterations_df, b, theta = run_gradient_descent(X, Y, learning_rate, num_iterations)
    print(gd_iterations_df)

    plt.figure(2)
    plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'])
    plt.title('Cost Vs.Iterations for different alpha values')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost or MSE')

    plt.show(block=False)

    print("Final Bias: ", b, " Final Theta: ", theta)

    return b, theta, gd_iterations_df

