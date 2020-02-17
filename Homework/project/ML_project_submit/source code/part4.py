import numpy as np
import json
def zerolistmaker(n):
    listofzeros = [-1e99] * n
    return listofzeros
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
        emission_parameter[word] = [tag_list,zerolistmaker(len(tag_list))]  # the 1st sub_list stores tags, the 2nd sub_list stores score
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
    # print(emission_parameter)
    return emission_parameter
def calculate_transition_parameter(train_y,tag_list,tag_dict):
    iter_num = 0
    print('----------------------------- Estimating the transition parameter ------------------------')
    transition_parameter = {}
    train_y_set = []
    for i in tag_list:
        transition_parameter[i] = [tag_list,zerolistmaker(len(tag_list))]
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
    print (transition_parameter)

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


def viterbi_kth(sentence,emission_para,transition_para,tag_dict,tag_list,k_value = 1):

    # shape of dp: (45,17) T -- 17 le -- 45
    T = len(tag_list)
    le = len(sentence)
    dp = np.array([[-1e99] *(T*k_value)] * (le+1))  # 节点最大概率对数
    path = np.zeros((le+1, T*k_value), dtype=int)  # 节点转移记录

    for index in range(T*k_value):
        if index % k_value == 0:
            j = index//k_value
            T_id = tag_dict[tag_list[j]]
            dp[0][index] = (transition_para['STOPKKKK'][1][T_id])+(emission_para[sentence[0]][1][j])
            path[0][index] = -1
        else:
            dp[0][index] =  -1.00000000e+10000
            path[0][index] = -1


    for i in range(1,le):
        for j in range(T):
            T_id = tag_dict[tag_list[j]]
            top_k_pairs = sorted(((dp[i-1][k]) + (transition_para[tag_list[k//k_value]][1][T_id]) + (emission_para[sentence[i]][1][j]),k)
                                for k in range(T*k_value))[-k_value:]
            dp[i][j*k_value:(j+1)*k_value] = np.array(list(tup[0] for tup in top_k_pairs))
            path[i][j*k_value:(j+1)*k_value] = np.array(list(tup[1] for tup in top_k_pairs))

    for j in range(T):
        top_k_pairs = sorted(((dp[-2][k]) + (transition_para[tag_list[k // k_value]][1][tag_dict['STOPKKKK']]),k)
                              for k in range(T * k_value))[-k_value:]

    dp[-1][0 * k_value:(0 + 1) * k_value] = np.array(list(tup[0] for tup in top_k_pairs))
    path[-1][0 * k_value:(0 + 1) * k_value] = np.array(list(tup[1] for tup in top_k_pairs))

    for j in range(1,T):
        dp[-1][j * k_value:(j + 1) * k_value] =  -1.00000000e+10000
        path[-1][j * k_value:(j + 1) * k_value] = -1
    # Backward decoding by checking the parent stored
    top_k_final_scores = sorted(dp[-1])[-k_value:][::-1] # large to small
    states = []


    for i in range(len(top_k_final_scores)):
        score = top_k_final_scores[i]
        start_ending = path[-1][list(dp[-1]).index(score)]
        states.append([start_ending//k_value])
        next = start_ending
        for x in range(le-2,-1,-1):
            next = path[x+1][next]
            states[i].append(next//k_value)
        states[i] = states[i][::-1]

    print("Top {}:\n".format(str(k_value)))
    for state in states:
        print(state)

    return states[-1]




def predict_tag_viterbi(input_data, transition_para,emission_para,tag_list,word_list,tag_dict,output_file,K=1):

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
                if K == 1:
                    result = viterbi(parsed_sentence,emission_para,transition_para,tag_dict,tag_list)
                else:
                    result = viterbi_kth(parsed_sentence,emission_para,transition_para,tag_dict,tag_list,K)
                results.append(result)
        # count=0
        # for states in results[0]:
        #     count+=1
        #     print('--------------here is sequence',count,'--------------------')
        #     for word,tid in zip(unlabeled_sentence[0],states):
        #         print(word, tag_list[tid])

        #write into file
        for i in range(len(results)):
            for word,tid in zip(unlabeled_sentence[i],results[i]):
                # print(word, tag_list[tid])
                file.write("{} {}\n".format(word, tag_list[tid]))
            file.write('\n')
    pass

def predict_tag(input_data, emission_para, output_file):
    print("----------------------------- Predicting tag ---------------------------------")
    with open(output_file, 'w+', encoding='utf8') as file:
        word_list = list(emission_para.keys())
        for word in input_data:
            if word in word_list:
                max_idx = emission_para[word][1].index(max(emission_para[word][1]))
                file.write("{} {}\n".format(word, emission_para[word][0][max_idx]))
            else:
                max_idx = emission_para["#UNK#"][1].index(max(emission_para["#UNK#"][1]))
                file.write("{} {}\n".format(word, emission_para["#UNK#"][0][max_idx]))
    pass

def dict_to_json_file(dic,file):
    j = json.dumps(dic)
    with open("{}.json".format(file),"w") as f:
        f.write(j)

def json_file_to_dict(json_file):
    with open(json_file,"r") as f:
        j = f.read()
    dict = json.loads(j)
    return dict

if __name__ == '__main__':
    path_cn_train = "./EN/train"
    path_cn_in = "./EN/dev.in"
    path_cn_out = "./EN/dev.out"
    modified_train_path = "en_train_modified"
    modified_train_path_tr = "en_train_modified_tr"
    predicted_out_path = "dev.p4_EN.out"
    #

    train_lines = open(path_cn_train, encoding='utf8').read().splitlines()
    in_lines = open(path_cn_in, encoding='utf8').read().splitlines()


    train_data_em, train_x, train_y_em = modify_train_data_em(train_lines, modified_train_path) #对数据做初步处理 划分train set
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
    transition_para = calculate_transition_parameter(train_y_tr,tag_list_tr,tag_dict) # 两个parameter返回的是dict形式
    dict_to_json_file(emission_para,"emission_para_{}".format("EN"))
    dict_to_json_file(transition_para,"transition_para_{}".format("EN"))

    # emission_para = json_file_to_dict("emission_para_{}.json".format("EN"))
    # transition_para = json_file_to_dict("transition_para_{}.json".format("EN"))

    print("result")
    # When the last parameter (k_value) is larger than 1, the code in part4 will be called
    predict_tag_viterbi(in_lines, transition_para,emission_para, tag_list_em,word_list,tag_dict,predicted_out_path,7)
    # predict_tag_viterbi(in_lines, transition_para,emission_para,tag_list_em,word_list,tag_dict,predicted_out_path,1)
    # print(transition_para)
    # predict_tag(in_lines, emission_para, predicted_out_path)

