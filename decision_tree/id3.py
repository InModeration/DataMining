from math import log
import operator
from decision_tree import tree_plotter


def create_data_set():
    """"
    创建样本数据
    :return 数据集及特征集
    """
    data_set = [[1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [1, 1, 'yes'], [0, 1, 'no']]
    labels = ['不浮出水面生存', '有脚蹼']
    return data_set, labels


def cal_shannon_ent(data_set):
    """"
    计算数据集的信息熵
    :param  data_set: 数据集
    :return 熵值
    """
    # 获取数据集的样本数量
    num = len(data_set)
    # 为所有分类类目创建字典
    label_counts = {}
    for feat_vec in data_set:
        # 获取样本最后一列的数据，记录每个类别出现的次数
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 计算熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num
        shannon_ent = shannon_ent - prob * log(prob, 2)
        return shannon_ent


def split_data_set(data_set, axis, value):
    """
    返回特征值等于value的子数据集，且该数据集不包含特定特征
    :param data_set: 等待划分的数据集
    :param axis:     已选择的特征的索引
    :param value:    分类值
    :return:         划分的子数据集
    """
    # 划分的子数据集
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 移除该特征
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    根据最大信息增益划分数据
    :param data_set: 样本数据集
    :return:    信息增益最大的特征的下标
    """
    # 定义最大的信息增益
    best_info_gain = 0
    # 定义最大信息增益对应的特征下标
    best_feature_idx = -1
    # 特征个数
    num_feature = len(data_set[0]) - 1
    # 数据集的信息熵
    base_entropy = cal_shannon_ent(data_set)
    for feature_idx in range(num_feature):
        # 获取某一特征所有的值
        feature_val_list = [row[feature_idx] for row in data_set]
        # 获取无重复的属性特征值
        unique_feature_val_list = set(feature_val_list)

        new_entropy = 0
        for feature_val in unique_feature_val_list:
            # 获取根据该特征值划分的子树
            sub_data_set = split_data_set(data_set, feature_idx, feature_val)
            # 求各个子树的熵并求和
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * cal_shannon_ent(sub_data_set)
        # 计算该特征的信息增益
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_idx = feature_idx
    return best_feature_idx


def majority_cnt(class_list):
    """
    统计每个类别出现的次数，从大到小排序，返回次数最大的类别标签
    :param class_list:  类数组
    :return:    次数出现最多的类别
    """
    # 统计类别个数的字典
    class_count = {}
    for kind in class_list:
        if kind not in class_count.keys():
            class_count[kind] = 0
        class_count[kind] += 1
    # 降序排列
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reversed=True)
    # print(sorted_class_count[0][0])
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    构建决策树
    :param data_set: 建树的数据集
    :param labels:   特征集
    :return:         根据传入的数据集和特征集生成的决策树
    """
    # 获取类别列表
    class_list = [data[-1] for data in data_set]
    # 如果所有的类别都相同的话，停止划分，即单节点
    # 对应的step1
    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[-1]
    # 如果类别的长度为1的话，则返回次数出现最多的类别
    # 对应的step2
    if len(class_list[0]) == 1:
        return majority_cnt(class_list)

    # 不符合上述情况的时候，选取信息增益最大的特征进行划分
    # 对应的step3及以后
    # 获取最大信息增益的特征下标
    best_feature_idx = choose_best_feature_to_split(data_set)
    # 该特征的名字
    best_feature_label = labels[best_feature_idx]
    # 构建树的字典
    tree = {best_feature_label: {}}
    # 从标签集合中删除选中的这个分类,因为即将用这个分类建子树
    del (labels[best_feature_idx])
    # 获取该特征属性所有的值
    best_feature_values = [feature_values[best_feature_idx] for feature_values in data_set]
    # 去重复
    unique_best_feature_values = set(best_feature_values)
    for best_feature_value in unique_best_feature_values:
        # 去除该分类后的分类集合
        sub_labels = labels[:]
        # 获取划分后的子数据集
        sub_data_set = split_data_set(data_set, best_feature_idx, best_feature_value)
        # 调用create_tree函数递归建树
        tree[best_feature_label][best_feature_value] = create_tree(sub_data_set, sub_labels)
    return tree


def classify(input_tree, feat_labels, test_vec):
    """
    决策树分类
    :param input_tree:  决策树
    :param feat_labels: 特征标签
    :param test_vec:    测试的数据
    :return:
    """
    # 获取树的第一层
    first_str = list(input_tree.keys())[0]
    # 第一层的子树
    second_dict = input_tree[first_str]
    # 获取决策树的第一层特征在特征数组中的位置
    feat_index = feat_labels.index(first_str)
    # 第二层开始递归调用该函数
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label


# test
data_set, labels = create_data_set()
decision_tree = create_tree(data_set, labels)
print("决策树", decision_tree)
data_set, labels = create_data_set()
print("不浮出水面可以生存，有脚蹼：", classify(decision_tree, labels, [1, 1]))
print("不浮出水面不可以生存，有脚蹼：", classify(decision_tree, labels, [0, 1]))
print("不浮出水面不可以生存，无脚蹼：", classify(decision_tree, labels, [0, 0]))
tree_plotter.create_plot(decision_tree)
