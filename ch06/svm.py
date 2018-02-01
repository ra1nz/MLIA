from numpy import *
from time import sleep


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split("\t")
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i;
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def smo_simple(data_mat_in, class_labels, c, error_t, max_iter):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose();
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    it = 0
    while it < max_iter:
        alpha_paris_changed = 0
        for i in range(m):
            fx_i = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            e_i = fx_i - float(label_mat[i])
            # 检查一个样例是否违反KKT
            if (label_mat[i] * e_i < -error_t and alphas[i] < c) or (label_mat[i] * e_i > error_t and alphas[i] > 0):
                j = select_j_rand(i, m)
                fx_j = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = fx_j - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                if l == h:
                    print("l==h")
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= label_mat[j] * (e_i - e_j) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                b1 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_paris_changed += 1
                print("iter:%d i:%d,pairs changed %d" % (it, i, alpha_paris_changed))
        if alpha_paris_changed == 0:
            it += 1
        else:
            it = 0
        print("iteration number:%d" % it)
    return b, alphas


data_arr, label_arr = load_data_set("testSet.txt")
b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
print(b)