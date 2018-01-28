# --coding:utf-8--
from numpy import *
import feedparser


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言论
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set()
    for doc in data_set:
        vocab_set = vocab_set | set(doc)  # 合并两个集合
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)  # 创建一个所有元素都为0的向量
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] = 1
        else:
            print("the word:%s is not in my vocabulary!" % word)
    return ret_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def bag_of_words2vec_mn(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] += 1
    return ret_vec


def testing_nb():
    list_of_posts, list_classes = load_data_set();
    vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for post_in_docs in list_of_posts:
        train_mat.append(set_of_words2vec(vocab_list, post_in_docs))
    p0v, p1v, pab = train_nb0(train_mat, list_classes)
    test_entry = ["love", "my", "dalmation"]
    this_doc = array(set_of_words2vec(vocab_list, test_entry))
    print(test_entry, "classified as:", classify_nb(this_doc, p0v, p1v, pab))
    test_entry = ["stupid", "garbage"]
    this_doc = array(set_of_words2vec(vocab_list, test_entry))
    print(test_entry, "classified as:", classify_nb(this_doc, p0v, p1v, pab))


def text_parse(string):
    import re
    list_of_tokens = re.split(r"\W*", string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open("email/spam/%d.txt" % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open("email/ham/%d.txt" % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)  # 创建词表
    training_set = range(50)
    test_set = []  # 创建测试集
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:  # 训练分类器
        train_mat.append(bag_of_words2vec_mn(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, pspam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:  # 对剩下的项进行分类
        word_vector = bag_of_words2vec_mn(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), p0v, p1v, pspam) != class_list[doc_index]:
            error_count += 1
            print("classification error", doc_list[doc_index])
    print("the error rate is:", float(error_count) / len(test_set))


def calc_most_freq(vocab_list, fulltext):
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = fulltext.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    fulltext = []
    min_len = min(len(feed1["entries"]), len(feed0["entries"]))
    for i in range(min_len):
        word_list = text_parse(feed1["entries"][i]["summary"])
        doc_list.append(word_list)
        fulltext.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0["entries"][i]["summary"])
        doc_list.append(word_list)
        fulltext.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)  # 创建词汇表
    top_30_words = calc_most_freq(vocab_list, fulltext)
    for w_pair in top_30_words:
        if w_pair[0] in vocab_list:
            vocab_list.remove(w_pair[0])
    training_set = range(2 * min_len)
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:  # 训练分类器
        train_mat.append(bag_of_words2vec_mn(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, pspam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_words2vec_mn(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), p0v, p1v, pspam) != class_list[doc_index]:
            error_count += 1
    print("the error rate is: ", float(error_count) / len(test_set))
    return vocab_list, p0v, p1v


def get_top_words(ny, sf):
    vocab_list, p0v, p1v = local_words(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0: top_sf.append((vocab_list[i], p0v[i]))
        if p1v[i] > -6.0: top_ny.append((vocab_list[i], p1v[i]))
    sorted_sf = sorted(top_sf, lambda pair: pair[1], reverse=True)
    print("****************SF****************")
    for item in sorted_sf:
        print(item[0])
    sorted_ny = sorted(top_ny, lambda pair: pair[1], reverse=True)
    print("****************NY****************")
    for item in sorted_ny:
        print(item[0])


ny = feedparser.parse("http://newyork.craigslist.org/res/index.rss")
sf = feedparser.parse("http://sfbay.craigslist.org/apa/index.rss")
get_top_words(ny, sf)
