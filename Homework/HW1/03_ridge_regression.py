import operator

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# TODO: (a) Calculate the exact solution theta
def ridge_regression(tX, tY, l):
    n = len(tX)
    matrix_I = np.identity(tX.shape[1], dtype='float')
    tX_transpose = tX.transpose()
    term_1_1 = n*l*matrix_I
    term_1_2 = np.dot(tX_transpose, tX)
    term_1 = inv(term_1_1 + term_1_2)
    theta = np.dot(np.dot(term_1, tX_transpose), tY)
    print("when lambda={}, theta: \n{}".format(l, theta))
    return theta


# TODO: (b) Plot graphs of the validation loss and training loss as lambda =10^-5 to 10^0,
#  return the lambda that minimizes the validation loss
def calculate_loss_and_plot(vX, vY, tX, tY):
    tn = tX.shape[0]
    vn = vX.shape[0]
    tloss = []
    vloss = []
    loss_dic = {}

    index = -np.arange(0, 5, 0.1)
    for i in index:
        w = ridge_regression(tX, tY, 10**i)
        tloss = tloss+[np.sum((np.dot(tX, w)-tY)**2/tn/2)]
        vloss = vloss+[np.sum((np.dot(vX, w)-vY)**2/vn/2)]
        loss_dic[np.sum((np.dot(vX, w)-vY)**2/vn/2)] = 10**i

    plt.plot(index, np.log(tloss), 'r')
    plt.plot(index, np.log(vloss), 'b')
    plt.show()
    min_loss_value = min(loss_dic.keys())
    print("the minimum validation loss: ", min_loss_value)
    print("the value of lambda with min validation loss: ", loss_dic[min_loss_value])



if __name__ == "__main__":
    # Clear input data
    input_file = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/3/hw1_ridge_x.dat"
    output_file = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/3/hw1_ridge_y.dat"
    # input_lines = np.loadtxt(input_file, dtype=np.float64)
    with open(input_file) as input_file:
        input_lines = input_file.read().splitlines()
    with open(output_file) as output_file:
        output_lines = output_file.read().splitlines()
    matrix_x = np.array([line.split(',') for line in input_lines], dtype='float')
    matrix_y = np.array([[line] for line in output_lines], dtype='float')
    vX = matrix_x[0:10]
    tX = matrix_x[-40:]
    vY = matrix_y[0:10]
    tY = matrix_y[-40:]

    ridge_regression(tX, tY, l=0.15)
    calculate_loss_and_plot(vX, vY, tX, tY)
