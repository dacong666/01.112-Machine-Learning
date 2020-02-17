import argparse


def train(train_file, threshold):
    with open(train_file) as train_file:
        lines = train_file.read().splitlines()
    train_date = [line.split(',') for line in lines]

    symmetry_value = [float(data[0]) for data in train_date]
    intensity_value = [float(data[1]) for data in train_date]
    label_value = [float(data[2]) for data in train_date]

    # initiate theta value
    theta = [0, 0]
    iter_no = 0
    while iter_no < threshold:
        for i in range(0, len(train_date)):
            if label_value[i]*(theta[0]*symmetry_value[i] + theta[1]*intensity_value[i]) <= 0:
                theta[0] += label_value[i]*symmetry_value[i]
                theta[1] += label_value[i]*intensity_value[i]
            else:
                pass
        iter_no += 1
    print("when iteration={}, theta is {}".format(threshold, theta))
    return theta


def predict(test_file, theta):
    with open(test_file) as test_file:
        lines = test_file.read().splitlines()
    test_date = [line.split(',') for line in lines]

    symmetry_value = [float(data[0]) for data in test_date]
    intensity_value = [float(data[1]) for data in test_date]
    label_value = [float(data[2]) for data in test_date]

    predicted_label = []
    for i in range(0, len(test_date)):
        predicted_value = theta[0]*symmetry_value[i] + theta[1]*intensity_value[i]
        if predicted_value >= 0:
            predicted_label.append(1.0)
        else:
            predicted_label.append(-1.0)
    correct_num = 0
    if len(label_value) == len(predicted_label):
        for i in range(0, len(label_value)):
            correct_num += 1 if label_value[i] == predicted_label[i] else 0
    else:
        print("something went wrong.")

    accuracy = correct_num/len(label_value)
    print("when theta = {}, the accuracy: {}".format(theta, accuracy))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', dest='filein', help='input file')
    # parser.add_argument('-t', dest='threshold', help='threshold')
    #
    # args = parser.parse_args()
    # filein = args.filein
    # threshold = args.threshold
    train_1 = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/1/train_1_5.csv"
    test_1 = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/HW1/data/1/test_1_5.csv"

    # set the threshold as how many iterations you want to run perceptron algorithm
    theta = train(train_1, threshold=5)
    predict(test_1, theta)
    theta_2 = train(train_1, threshold=10)
    predict(test_1, theta_2)