from svmutil import *
# Read data in LIBSVM format
# y, x = svm_read_problem('../heart_scale')
train_y, train_x = svm_read_problem("../../HW3_data/1/promoters/training.txt")
test_y, test_x = svm_read_problem("../../HW3_data/1/promoters/test.txt")

# isKernel=True must be set for precomputed kernel
prob0 = svm_problem(train_y, train_x, isKernel=True)
param0 = svm_parameter('-t 0')
m0 = svm_train(prob0, param0)
p_label0, p_acc0, p_val0 = svm_predict(test_y, test_x, m0)

prob1 = svm_problem(train_y, train_x, isKernel=True)
param1 = svm_parameter('-t 1')
m1 = svm_train(prob1, param1)
p_label1, p_acc1, p_val1 = svm_predict(test_y, test_x, m1)

prob2 = svm_problem(train_y, train_x, isKernel=True)
param2 = svm_parameter('-t 2')
m2 = svm_train(prob1, param1)
p_label2, p_acc2, p_val2 = svm_predict(test_y, test_x, m2)

prob3 = svm_problem(train_y, train_x, isKernel=True)
param3 = svm_parameter('-t 3')
m3 = svm_train(prob3, param3)
p_label3, p_acc3, p_val3 = svm_predict(test_y, test_x, m3)




# # Other utility functions
# svm_save_model('heart_scale.model', m)
# m = svm_load_model('heart_scale.model')
# p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
# ACC, MSE, SCC = evaluations(y, p_label)
#
# # Getting online help
#
# from svm import *
# prob = svm_problem([1,-1], [{1:1, 3:1}, {1:-1,3:-1}])
# param = svm_parameter('-c 4')
# m = libsvm.svm_train(prob, param) # m is a ctype pointer to an svm_model
# # Convert a Python-format instance to svm_nodearray, a ctypes structure
# x0, max_idx = gen_svm_nodearray({1:1, 3:1})
# label = libsvm.svm_predict(m, x0)
