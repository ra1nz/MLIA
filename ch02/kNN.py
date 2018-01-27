# -- coding: utf-8 --
from numpy import *
import os
import operator


def create_data_set():
    group = array([[1., 1.1], [1., 1.], [0, 0], [0, .1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


# kNN算法
def classify0(in_x, data_set, labels, k):
    # 距离计算
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    # 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 将文件转为矩阵
def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines();
    number_of_lines = len(array_of_lines);  # 取文件行
    re_mat = zeros((number_of_lines, 3))  # number_of_lines*3矩阵
    class_label_vector = []
    index = 0;
    for line in array_of_lines:  # 解析数据
        line = line.strip();
        list_from_line = line.split('\t');
        re_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return re_mat, class_label_vector


# 归一化特征值
def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals;
    zeros(shape(data_set))
    m = data_set.shape[0];
    norm_data_set = data_set - tile(ranges, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    horatio = 0.1
    dating_data_mat, dating_labels = file2matrix("datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * horatio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ["not at all", "in small doses", "in large doses"]
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix("datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("you will probably like this person:", result_list[classifier_result - 1])


def img2vector(filename):
    re_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline();
        for j in range(32):
            re_vect[0, 32 * i + j] = int(line[j])
    return re_vect


def handwriting_class_test():
    hw_labels = []
    training_file_list = os.listdir("digits/trainingDigits")
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i];
        file_str = file_name_str.split(".")[0]
        class_num_str = int(file_str.split("_")[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector("digits/trainingDigits/%s" % file_name_str)
    test_file_list = os.listdir("digits/testDigits")
    errorCount = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split(".")[0]
        class_num_str = int(file_str.split("_")[0])
        vector_under_test = img2vector("digits/testDigits/%s" % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with:%d,the real answer is:%d" % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m_test)))


handwriting_class_test()
