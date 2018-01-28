# -- coding: utf-8 --
from math import log
import operator


def create_data_set():
    data_set = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return data_set, labels


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    # 为所有分类创建字典
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)  # 以2为底计算对数
    return shannon_ent


# 划分数据集
def split_data_set(data_set, axis, value):
    ret_data_set = []  # 创建新的list对象
    for feat_vec in data_set:  # 抽取数据
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set=data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 创建唯一的分类标签列表
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0
        # 计算每种划分方式的信息熵
        for value in unique_values:
            sub_data_set = split_data_set(data_set=data_set, axis=i, value=value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        # 找到最好的信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    return sorted(class_count.items(), key=operator.itemgetter()[1], reverse=True)[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    # 类别完全相同则停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    print(labels)
    print(best_feat)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    # 得到列表包含的所有属性值
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]  # 复制全部的标签，使树里不会漏掉存在的标签
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set=data_set, axis=best_feat, value=value),
                                                      sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)  # 将标签字符串转为索引
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, "w")
    pickle.dump(input_tree)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
