def modify_train_data(train_x, train_y):
    x_list = []
    y_list = []
    print("-------- Modifying the train data ---------")
    for i in range(len(train_x)):
        if train_x.count(train_x[i]) < 3:
            train_x[i] = '#UNK#'
        if train_x[i] not in x_list:
            x_list.append(train_x[i])
        if train_y[i] not in y_list:
            y_list.append(train_y[i])
    print("-------- Train data is modified. ---------")

    return x_list, y_list


def init_parameters(x_list, y_list):
    init_emission = {}
    init_transition = {}
    for y in y_list:
        init_emission[y] = {}
        for x in x_list:
            init_emission[y][x] = [0.0, 0.0]

    y_list = ['STARTTTT'] + y_list + ['STOPPPP']
    for y_1 in y_list[:-1]:
        init_transition[y_1] = {}
        for y_2 in y_list[1:]:
            init_transition[y_1][y_2] = [0, 0]

    print("Parameter are initiated.")
    return init_emission, init_transition


def calculate_parameters(train_path, train_x, emission_parameter, transition_parameters, threshold):
    n = 1  # total no of items in train
    for iter_no in range(threshold):
        print("-------- {}th iteration on calculating the parameters. ---------------------".format(iter_no))
        words = []
        tags = []
        for line in open(train_path, encoding='utf-8').read().splitlines():
            if line:
                line = line.split()
                words.append(line[0].lower())
                tags.append(line[1])
            else:
                # train for one sentence
                predicted_tags = viterbi(words, train_x, emission_parameter, transition_parameters)
                emission_parameter, transition_parameter = update_parameters(words, tags, predicted_tags, train_x, emission_parameter, transition_parameters)
                n += 1
                words = []
                tags = []

    # for tag in list(emission_parameter):
    #     for word in emission_parameter[tag]:
    #         emission_parameter[tag][word][0] /= (n)
    #
    # for tag_1 in list(transition_parameters):
    #     for tag_2 in transition_parameters[tag_1]:
    #         transition_parameters[tag_1][tag_2][0] /= (n)
    return emission_parameter, transition_parameters


def viterbi(words, train_x, emission_para, transition_para):
    tags = list(emission_para.keys())
    pi = [{tag: [-100, ''] for tag in tags} for word in words]  # no. of layers = no. of words
    # base case
    for tag in tags:
        score = 0

        # score + transition_para
        score += transition_para['STARTTTT'][tag][0]
        # score + transition_para + emission_para
        if words[0] in train_x:
            score += emission_para[tag][words[0]][0]
        else:  #word is #UNK#
            score += emission_para[tag]['#UNK#'][0]

        pi[0][tag] = [score, 'STARTTTT']

    # moving forward recursively
    for k in range(1, len(words)):
        for tag_1 in tags:
            for tag_2 in tags:
                score = pi[k-1][tag_2][0]
                # score + transition_para
                score += transition_para[tag_2][tag_1][0]

                if score >= pi[k][tag_1][0]:  # take the max
                    pi[k][tag_1] = [score, tag_2]

            if words[k] in train_x:
                pi[k][tag_1][0] += emission_para[tag_1][words[k]][0]
            else:  # word is #UNK#
                pi[k][tag_1][0] += emission_para[tag_1]['#UNK#'][0]

    # final state
    result = [-100, '']
    for p_tag in tags:
        score = pi[-1][p_tag][0] + transition_para[p_tag]['STOPPPP'][0]
        if score >= result[0]:
            result = [score, p_tag]

    # Backtracking
    prediction = [result[1]]
    for k in reversed(range(len(words))):
        if k != 0:
            prediction.insert(0, pi[k][prediction[0]][1])

    return prediction


def update_parameters(words, tags, predicted_tags, train_x, emission_para, transition_para):
    tags = ['STARTTTT'] + tags + ['STOPPPP']
    predicted_tags = ['STARTTTT'] + predicted_tags + ['STOPPPP']
    words = [''] + words + ['']

    for i in range(len(tags)):
        if tags[i] != predicted_tags[i]:
            word = words[i] if words[i] in train_x else '#UNK#'
            # update emission parameters
            emission_para[tags[i]][word][0] += 1
            emission_para[predicted_tags[i]][word][0] -= 1

            # update transition parameters
            transition_para[tags[i-1]][tags[i]][0] += 1
            transition_para[tags[i]][tags[i+1]][0] += 1

            transition_para[predicted_tags[i-1]][predicted_tags[i]][0] -= 1
            transition_para[predicted_tags[i]][predicted_tags[i+1]][0] -= 1

    return emission_para, transition_para


def predict_tags(input_file, train_x, emission_para, transition_para, output_file):
    print("-------- Start to predict the output labels. --------------------")

    words = []
    modified_words = []
    with open(output_file, 'w+', encoding='utf-8') as file:
        for line in open(input_file, encoding='utf-8').read().splitlines():
            line = line.split()
            if line:
                words.append(line[0])
                modified_words.append(line[0].lower())
            else:
                predicted_tags = viterbi(modified_words, train_x, emission_para, transition_para)
                for i in range(len(words)):
                    file.write("{} {}\n".format(words[i], predicted_tags[i]))
                file.write('\n')

                words = []
                modified_words = []

    print("-------- Labels are predicted. Please check. --------------------")


if __name__ == '__main__':
    path_al_train = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/AL/train"
    path_al_in = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/AL/dev.in"
    path_al_out = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/AL/dev.out"
    predicted_out_path = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/AL/dev.p5.out"
    path_test_in = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/Test/AL/test.in"
    path_test_out = "C:/Users/87173/Desktop/term6/Machine Learning/Homework/project/AL/test.p5.out"

    train_lines = list(filter(None, open(path_al_train, encoding='utf-8').read().splitlines()))
    in_lines = open(path_al_in, encoding='utf-8').read().splitlines()

    train_al_data = [line.split() for line in train_lines]
    train_al_x = [line[0].lower() for line in train_al_data if line]
    train_al_y = [line[1] for line in train_al_data if line]

    train_x, train_y = modify_train_data(train_al_x, train_al_y)
    init_emission, init_transition = init_parameters(train_x, train_y)
    emission_parameter, transition_parameter = calculate_parameters(path_al_train, train_x, init_emission, init_transition, threshold=8)  # k=8 max
    predict_tags(path_al_in, train_x, emission_parameter, transition_parameter, predicted_out_path)
    predict_tags(path_test_in, train_x, emission_parameter, transition_parameter, path_test_out)
