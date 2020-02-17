import random

import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np


# TODO: (a) Load the data and plot it
def load_plot_data(input_file, output_file):
    with open(input_file) as input_file:
        input_lines = input_file.read().splitlines()
    with open(output_file) as output_file:
        output_lines = output_file.read().splitlines()

    # Clear input data
    input_data_str = list(filter(None, [line.replace('   ', '') for line in input_lines]))
    input_data = [float(data) for data in input_data_str]
    output_data_str = list(filter(None, [line.replace('   ', '') for line in output_lines]))
    output_data = [float(data) for data in output_data_str]

    plt.plot(input_data, output_data, 'o', markersize=1)

    plt.show()


# TODO: (b) Implement the closed form linear regression formula to obtain the weight vector theta.
#  Plot both the linear regression line and the data on the same graph
def calculate_weight_theta_closed_form(input_file, output_file):
    with open(input_file) as input_file:
        input_lines = input_file.read().splitlines()
    with open(output_file) as output_file:
        output_lines = output_file.read().splitlines()

    # Clear input data
    input_data_str = list(filter(None, [line.replace('   ', '') for line in input_lines]))
    input_data = [[1.0, float(data)] for data in input_data_str]
    output_data_str = list(filter(None, [line.replace('   ', '') for line in output_lines]))
    output_data = [float(data) for data in output_data_str]

    n = len(input_data)
    # Calculate matrix A
    matrix_x = np.array(input_data)
    matrix_x_T = matrix_x.transpose()
    matrix_a = np.array(1/n*np.dot(matrix_x_T, matrix_x), dtype='float')
    inv_matrix_a = inv(matrix_a)
    # Calculate b
    matrix_y = np.array(output_data)
    matrix_b = 1/n*np.dot(matrix_x_T, matrix_y)
    # Calculate theta
    theta = np.dot(inv_matrix_a, matrix_b)
    print("theta(closed form): ", theta)
    print("dimension A: ", matrix_a.shape)
    print("dimension y: ", matrix_y.shape)
    print("dimension b: ", matrix_b.shape)
    print("dimension x: ", matrix_x.shape)


    matrix_predicted_y = np.dot(matrix_x, theta)

    plt.plot(matrix_x[:, 1], matrix_y, 'o', markersize=1)
    plt.plot(matrix_x[:, 1], matrix_predicted_y, 'g')
    plt.show()
    return matrix_x, matrix_y, matrix_predicted_y


# TODO: (c) Evaluate the training error in terms of empirical risk of the resulting fit in 2(b)
#  and report the error.
def calculate_training_error(matrix_x, matrix_y, matrix_predicted_y):
    n = len(matrix_x)
    empirical_risk = 1/n * sum([((matrix_y[i]-matrix_predicted_y[i])**2)/2 for i in range(0, n)])
    print("empirical_risk(closed_form): ", empirical_risk)
    return empirical_risk


# TODO: (d) Calculate weight vector theta using gradient descent with eta=0.01, threshold=100
def calculate_weight_theta_gradient_descent(matrix_x, matrix_y, eta, threshold):
    theta = np.array([0.0, 0.0])
    n = len(matrix_x)
    risk_result = 500

    for iter_no in range(0, threshold):
        for i in range(0, n):
            y = matrix_y[i]
            predicted_y = np.dot(matrix_x[i], theta)
            theta += eta*(y-predicted_y)*matrix_x[i]
            empirical_risk = 1 / n * sum([((matrix_y[i] - np.dot(matrix_x, theta)[i]) ** 2) / 2 for i in range(0, n)])
            if empirical_risk < risk_result:
                risk_result = empirical_risk
                theta_result = list(theta)

    print("theta(gradient_descent): ", theta_result)
    print("empirical_risk(gradient_descent): ", risk_result)
    return theta


# TODO: (e) Calculate weight vector theta using stochastic gradient descent with eta=0.01, threshold=50
def calculate_weight_theta_stochastic_gradient(matrix_x, matrix_y, eta, threshold):
    theta = np.array([0.0, 0.0])
    n = len(matrix_x)
    risk_result = 500
    print((matrix_x))

    for iter_no in range(0, threshold):
        i = random.randint(0, n-1)
        y = matrix_y[i]
        predicted_y = np.dot(matrix_x[i], theta)
        # print(y, matrix_x[i], theta)
        theta += eta*(y-predicted_y)*matrix_x[i]
        empirical_risk = 1 / n * sum([((matrix_y[i] - np.dot(matrix_x, theta)[i]) ** 2) / 2 for i in range(0, n)])

        if empirical_risk < risk_result:
            risk_result = empirical_risk
            theta_result = theta

    print("theta(stochastic_gradient): ", theta_result)
    print("empirical_risk(stochastic_gradient): ", risk_result)
    return theta


# TODO: (f) Add the features x^2, x^3, ...x^d to the inputs and
#  performs polynomial regression using closed form solution
def PolyRegress(x, y, d):
    with open(x) as input_file:
        input_lines = input_file.read().splitlines()
    with open(y) as output_file:
        output_lines = output_file.read().splitlines()

    # Clear input data
    input_data_str = list(filter(None, [line.replace('   ', '') for line in input_lines]))
    input_data = [[float(data)] for data in input_data_str]
    initial_x = np.array(input_data)
    temp = np.ones(shape=initial_x.shape)
    matrix_x = np.concatenate((temp.reshape(-1, 1), initial_x.reshape(-1, 1)), 1)
    if d >= 2:
        for i in range(2, d+1):
            matrix_x = np.concatenate((matrix_x, (initial_x**i).reshape(-1, 1)), 1)

    output_data_str = list(filter(None, [line.replace('   ', '') for line in output_lines]))
    output_data = [float(data) for data in output_data_str]

    n = len(input_data)
    # Calculate matrix A
    matrix_x_T = matrix_x.transpose()
    matrix_a = np.array(1/n*np.dot(matrix_x_T, matrix_x), dtype='float')
    inv_matrix_a = inv(matrix_a)
    # Calculate b
    matrix_y = np.array(output_data)
    matrix_b = 1/n*np.dot(matrix_x_T, matrix_y)
    # Calculate theta
    theta = np.dot(inv_matrix_a, matrix_b)
    print("theta(poly regression for d = {}): ".format(d), theta)
    matrix_predicted_y = np.dot(matrix_x, theta)

    return matrix_x, matrix_y, theta, matrix_predicted_y


# TODO: (g) Get a quadratic fit of the data. Plot the data and the fit. Report the training error.
#  Repeat the same for 3rd order fit to 9th order fit.
def training_error_for_poly_regression(input_file, output_file):
    training_error_dic = {}
    # for quadratic fit
    matrix_x, matrix_y, theta, matrix_predicted_y = PolyRegress(input_file, output_file, d=2)
    n = len(matrix_x)
    empirical_risk = 1 / n * sum([((matrix_y[i] - np.dot(matrix_x, theta)[i]) ** 2) / 2 for i in range(0, n)])

    training_error_dic["quadratic fit"] = empirical_risk
    print("training error for a quadratic fit: ", empirical_risk)
    plt.plot(matrix_x[:, 1], matrix_y, 'o', markersize=1)
    plt.plot(matrix_x[:, 1], matrix_predicted_y, 'g')
    plt.show()
    for i in range(3, 10):
        matrix_x, matrix_y, theta, matrix_predicted_y = PolyRegress(input_file, output_file, d=i)
        empirical_risk = 1 / n * sum([((matrix_y[i] - np.dot(matrix_x, theta)[i]) ** 2) / 2 for i in range(0, n)])
        training_error_dic["{}th order fit".format(i)] = empirical_risk
    print("training error from 3th order fit to 9th order fit: ", training_error_dic)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', dest='filein', help='input file')
    # parser.add_argument('-t', dest='threshold', help='threshold')
    #
    # args = parser.parse_args()
    # filein = args.filein
    # threshold = args.threshold
    input_file = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/2/hw1x.dat"
    output_file = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/2/hw1y.dat"

    # load_plot_data(input_file, output_file)
    matrix_x, matrix_y, matrix_predicted_y = calculate_weight_theta_closed_form(input_file, output_file)
    # calculate_training_error(matrix_x, matrix_y, matrix_predicted_y)
    # calculate_weight_theta_gradient_descent(matrix_x, matrix_y, eta=0.01, threshold=100)
    # calculate_weight_theta_stochastic_gradient(matrix_x, matrix_y, eta=0.01, threshold=50)
    # PolyRegress(input_file, output_file, d=5)
    # training_error_for_poly_regression(input_file, output_file)
