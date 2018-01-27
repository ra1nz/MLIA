from numpy import *
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
