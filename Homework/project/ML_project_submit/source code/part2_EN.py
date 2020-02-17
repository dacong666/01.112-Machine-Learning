def modify_train_data(train_x, train_y, modified_train_path):
    print("-------- Modifying the train data (comment this part if you have modified file, for saving time.) ---------")
    # with open(modified_train_path, 'w+') as modified_train_file:
    #     # modified training data
    #     for i in range(len(train_x)):
    #         if train_x.count(train_x[i]) < 3:
    #             train_x[i] = '#UNK#'
    #         modified_train_file.write('{} {} \n'.format(train_x[i], train_y[i]))
    print("-------- Train data is modified. (comment this part if you have modified file, for saving time.) ---------")

    lines = list(filter(None, open(modified_train_path).read().splitlines()))
    train_data = [line.split() for line in lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    return train_data, train_x, train_y


def calculate_emission_parameter(train_data, train_x, train_y):
    iter_num = 0  # can be deleted later
    print('----------------------------- Estimating the emission parameter ------------------------')
    emission_parameter = {}
    for word in train_x:
        emission_parameter[word] = [[], []]  # the 1st sub_list stores tags, the 2nd sub_list stores score

    for pair in train_data:
        if pair[1] not in emission_parameter[pair[0]][0]:
            emission_parameter[pair[0]][0].append(pair[1])
            emission_parameter[pair[0]][1].append(train_data.count(pair) / train_y.count(pair[1]))
        iter_num += 1
    print('----------------------------- Emission parameter is calculated -------------------------')
    return emission_parameter


def predict_tag(input_data, emission_para, output_file):
    print("----------------------------- Predicting tag ---------------------------------")
    with open(output_file, 'w+') as file:
        word_list = list(emission_para.keys())
        for word in input_data:
            if word == '':
                file.write("\n")
            elif word in word_list:
                max_idx = emission_para[word][1].index(max(emission_para[word][1]))
                file.write("{} {}\n".format(word, emission_para[word][0][max_idx]))
            else:
                max_idx = emission_para["#UNK#"][1].index(max(emission_para["#UNK#"][1]))
                file.write("{} {}\n".format(word, emission_para["#UNK#"][0][max_idx]))
    pass


if __name__ == '__main__':
    path_en_train = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/EN/train"
    path_en_in = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/EN/dev.in"
    path_en_out = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/EN/dev.out"
    modified_train_path = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/EN/en_train_modified"
    predicted_out_path = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/EN/dev.p2.out"
    # train_lines = open(path_en_train).read().splitlines()

    train_lines = list(filter(None, open(path_en_train).read().splitlines()))
    # in_lines = list(filter(None, open(path_en_in).read().splitlines()))
    in_lines = open(path_en_in, encoding='utf-8').read().splitlines()

    train_en_data = [line.split() for line in train_lines]
    train_en_x = [line[0] for line in train_en_data if line]
    train_en_y = [line[1] for line in train_en_data if line]

    # train_data, train_x, train_y = modified_train_path(path_en_train, modified_train_path)
    train_data, train_x, train_y = modify_train_data(train_en_x, train_en_y, modified_train_path)
    emission_para = calculate_emission_parameter(train_data, train_x, train_y)
    predict_tag(in_lines, emission_para, predicted_out_path)


