import numpy as np
def minvaluelistmaker(n):
    listofminvalues = [-1e99] * n
    return listofminvalues
def modify_train_data_em(train_lines,modified_train_path):
    print("-------- Modifying the train data (comment this part if you have modified file, for saving time.) ---------")
    # train_al_data = [line.split() for line in train_lines if line]
    # train_al_x = [line[0] for line in train_al_data if line]
    # train_al_y = [line[-1] for line in train_al_data if line]

    # with open(modified_train_path, 'w+', encoding='utf8') as modified_train_file:
    #     # modified training data
    #     for i in range(len(train_al_x)):
    #         if train_al_x.count(train_al_x[i]) < 3:
    #             train_al_x[i] = '#UNK#'
    #         modified_train_file.write('{} {} \n'.format(train_al_x[i], train_al_y[i]))
    print("-------- Train data is modified. (comment this part if you have modified file, for saving time.) ---------")

    lines = open(modified_train_path, encoding='utf8').read().splitlines()
    train_data = [line.split() for line in lines if line]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]
    return train_data, train_x, train_y
def modify_train_data_tr(train_lines, modified_train_path_tr):
    print("-------- Modifying the train data (comment this part if you have modified file, for saving time.) ---------")
    # train_al_data = [line.split() if line else ['STOPKKKK','STOPKKKK'] for line in train_lines]
    # train_al_x = [line[1] if line else 'STOPKKKK' for line in train_al_data]
    # train_al_y = [line[1] if line else 'STOPKKKK' for line in train_al_data]
    # with open(modified_train_path_tr, 'w+', encoding='utf8') as modified_train_file:
    #     # modified training data
    #     for i in range(len(train_al_x)):
    #         if train_al_x.count(train_al_x[i]) < 3:
    #             train_al_x[i] = '#UNK#'
    #         modified_train_file.write('{} {} \n'.format(train_al_x[i], train_al_y[i]))
    print("-------- Train data is modified. (comment this part if you have modified file, for saving time.) ---------")

    lines = open(modified_train_path_tr, encoding='utf8').read().splitlines()
    train_data = [line.split() for line in lines]
    train_y = [line[1] if line else 'STOPKKKK' for line in train_data]
    # print(type(train_y))
    train_y = ['STOPKKKK']+train_y
    # train_y = train_y.(-1,'STOPKKKK')


    return train_y


def calculate_emission_parameter(train_data, train_x, train_y,tag_list,word_list):
    iter_num = 0  # can be deleted later
    print('----------------------------- Estimating the emission parameter ------------------------')
    emission_parameter = {}
    for word in word_list:
        emission_parameter[word] = [tag_list,minvaluelistmaker(len(tag_list))]  # the 1st sub_list stores tags, the 2nd sub_list stores score
    print('----------------------------- halfway here ------------------------')
    tag_dict = {}
    for idx, tag in enumerate(tag_list):
        tag_dict[tag]=idx
    pair_list = []
    for pair in train_data:
        if pair:
            if pair not in pair_list:
                pair_list.append(pair)
                T_id = tag_dict[pair[1]]
                emission_parameter[pair[0]][1][T_id]=float(np.log(train_data.count(pair) / train_y.count(pair[1])))
        # iter_num += 1
    print('----------------------------- Emission parameter is calculated -------------------------')
    return emission_parameter
def calculate_transition_parameter(train_y,tag_list,tag_dict):
    iter_num = 0
    print('----------------------------- Estimating the transition parameter ------------------------')
    transition_parameter = {}
    train_y_set = []
    for i in tag_list:
        transition_parameter[i] = [tag_list,minvaluelistmaker(len(tag_list))]
    print('--------------------- 1/3 way --------------------------')
    for i in range(len(train_y)-1):
        train_y_set = train_y_set+[[train_y[i],train_y[i+1]]]
    print('--------------------- 2/3 way --------------------------')
    tagpair_list=[]
    for i in range(len(train_y)-1):
        if [train_y[i],train_y[i+1]] not in tagpair_list:
            tagpair_list.append([train_y[i],train_y[i+1]])
            T_id = tag_dict[train_y[i+1]]
            transition_parameter[train_y[i]][1][T_id]=float(np.log(train_y_set.count([train_y[i],train_y[i+1]]) / train_y.count(train_y[i])))
    print('----------------------------- Transition parameter is calculated -------------------------')


    return transition_parameter

def viterbi(sentence,emission_para,transition_para,tag_dict,tag_list):

    T = len(tag_list)
    le = len(sentence)
    dp = np.array([[-1e99] *T] * (le))  # 节点最大概率对数
    path = np.zeros((le, T), dtype=int)  # 节点转移记录

    for j in range(T):
        T_id = tag_dict[tag_list[j]]
        # dp[0][j] = start_p[T_id] + emission_para[sentence[0]][1][j]
        dp[0][j] = (transition_para['STOPKKKK'][1][T_id])+(emission_para[sentence[0]][1][j])
        path[0][j] = -1

    for i in range(1,le):
        for j in range(T):
            T_id = tag_dict[tag_list[j]]
            dp[i][j], path[i][j]= max(((dp[i-1][k]) + (transition_para[tag_list[k]][1][T_id]) + (emission_para[sentence[i]][1][j]),k)
                                for k in range(T))

    for i in range(T):
        dp[-1][i]= dp[-1][i]+(transition_para[tag_list[i]][1][tag_dict['STOPKKKK']])

    states = [np.argmax(dp[-1])]
    for i in range(le-2, -1, -1):
        states.insert(0, path[i + 1][states[0]])

    return states

def predict_tag_viterbi(input_data, transition_para,emission_para,tag_list,word_list,tag_dict,output_file):

    print("----------------------------- Predicting tag ---------------------------------")
    with open(output_file, 'w+', encoding='utf8') as file:
    # Generate array of arrays for sentences in document
        unlabelled_in = []
        temp_data = []
        for line in input_data:
            if line == "":
                if temp_data:  # catch any multiple line breaks
                    unlabelled_in.append(temp_data)
                temp_data = []
            else:
                temp_data.append(line)
        unlabelled_in.append(temp_data)


        results = []
        unlabeled_sentence = []


            # execute viterbi for each sentence
        for sentence in unlabelled_in:
            if sentence:
                parsed_sentence = []
                # parse and replace unknowns with #UNK#
                for i in range(len(sentence)):
                    if sentence[i] in word_list:
                        parsed_sentence.append(sentence[i])
                    else:
                        parsed_sentence.append('#UNK#')
                unlabeled_sentence.append(sentence)
                result = viterbi(parsed_sentence,emission_para,transition_para,tag_dict,tag_list)
                results.append(result)
        print(len(unlabeled_sentence[1]))
        print(len(results[1]))
        print(len(unlabeled_sentence))
        print(len(results))

        print("----------------------------- Writing to file ---------------------------------")
        for i in range(len(results)):
            for word,tid in zip(unlabeled_sentence[i],results[i]):
                file.write("{} {}\n".format(word, tag_list[tid]))
            file.write('\n')
    pass


if __name__ == '__main__':
    path_cn_train = "/Users/hehe/Desktop/ML/project/CN/train"
    path_cn_in = "/Users/hehe/Desktop/ML/project/CN/dev.in"
    path_cn_out = "/Users/hehe/Desktop/ML/project/CN/dev.out"
    modified_train_path = "/Users/hehe/Desktop/ML/project/cn_train_modified"
    modified_train_path_tr = "/Users/hehe/Desktop/ML/project/cn_train_modified_tr"
    predicted_out_path = "/Users/hehe/Desktop/ML/project/CN/dev.p3.out"
    #

    train_lines = open(path_cn_train, encoding='utf8').read().splitlines()
    in_lines = open(path_cn_in, encoding='utf8').read().splitlines()


    train_data_em, train_x, train_y_em = modify_train_data_em(train_lines, modified_train_path)
    train_y_tr = modify_train_data_tr(train_lines, modified_train_path_tr)
    tag_list_tr = list(dict.fromkeys(train_y_tr))
    tag_list_em = list(dict.fromkeys(train_y_em))
    word_list = list(dict.fromkeys(train_x))
    tag_dict = {}
    for idx, tag in enumerate(tag_list_tr):
        tag_dict[tag]=idx
        print(idx)
    print(len(tag_list_tr))
    print(len(tag_list_em))
    emission_para = calculate_emission_parameter(train_data_em, train_x, train_y_em, tag_list_em, word_list)
    transition_para = calculate_transition_parameter(train_y_tr,tag_list_tr,tag_dict)

    predict_tag_viterbi(in_lines, transition_para,emission_para,tag_list_em,word_list,tag_dict,predicted_out_path)

