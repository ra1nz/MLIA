# -- coding: utf-8 --
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(tree):
    num_leafs = 0
    for key in tree.keys():
        second_dict = tree[key]
        for key in second_dict.keys():
            # 判断数据类型是否为字典类型
            if type(second_dict[key]) == dict:
                num_leafs += get_num_leafs(second_dict[key])
            else:
                num_leafs += 1
    return num_leafs


def get_tree_depth(tree):
    max_depth = 0
    for key in tree.keys():
        second_dict = tree[key]
        for key in second_dict.keys():
            if type(second_dict[key]) == dict:
                this_depth = 1 + get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
    return max_depth


# 绘制带箭头的注解
def plot_node(node_text, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_text, xy=parent_pt, xycoords="axes fraction", xytext=center_pt,
                             textcoords="axes fraction", va="center", ha="center", bbox=node_type,
                             arrowprops=arrow_args)


# 在父子节点间填充文本信息
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


# 计算宽与高
def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = get_num_leafs(myTree)  # this determines the x width of this tree
    depth = get_tree_depth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)  # 标记子节点属性值
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree[firstStr]
    # 减少Y轴偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leaf_node)
            plot_mid_text((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(get_num_leafs(in_tree))
    plotTree.totalD = float(get_tree_depth(in_tree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(in_tree, (0.5, 1.0), '')
    plt.show()


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]
