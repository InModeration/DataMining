import math
import jieba.analyse
import random
import os

"""
多项式朴素贝叶斯的实现，即最终的公式为

    P(wi | c) = count(wi, c) + 1 / ∑count(w, c) + |V|
    --> max( P(wi | c) * P(c) ) = log P(c) + ∑log P(wi | c)

"""

# 文档类型数目
TYPES = 14

# 平滑值
SMOOTH = 1

# 停用词的路径
stop_words_path = "stopwords.txt"

# 训练集每类文档的大小
train_num = 50
# 测试集每类文档的大小
test_num = 10


class Bayes(object):
    def __init__(self, dataset, types, typees):
        # 数据字典
        self.dataset = dataset
        # 每个文档类型的数目
        self.types = types
        # 记载训练集数据
        self.V, self.all_data, self.type_datas, self.type_words = self.load()
        self.all = 0
        for type_word in self.type_words:
            self.all += type_word
        self.typees = typees

    # 获取整个词库大小（无重复）、整个数据集的所有词list 、每个文档类型的所有词list以及它的数目
    def load(self):
        v_data = []
        all_data = []
        # 文档类型的所有单词
        type_datas = []
        # 文档类型的单词数
        type_words = []
        for data in self.dataset.values():
            temp = []
            for type_data in data:
                all_data += type_data
                temp += type_data
            type_datas.append(temp)
        v_data = set(all_data)
        for i in range(TYPES):
            type_words.append(len(type_datas[i]))
        return len(v_data), all_data, type_datas, type_words

    def get_type(self, input_data):
        p_values = []
        for i in range(TYPES):
            # p_value = 1
            temp = []
            for word_i in input_data:
                numerator = 0
                for word in self.all_data:
                    if word_i == word:
                        numerator += 1
                numerator += SMOOTH
                denominator = self.V + self.type_words[i]
                p_value = math.log(numerator / denominator)
                temp.append(p_value)
            p_value = 0
            for p_value_i in temp:
                p_value += p_value_i
            p_value += math.log(self.type_words[i] / self.all)
            p_values.append(p_value)
        index = p_values.index(max(p_values))
        # return self.typees[index]
        return index


# 获取停用词集合
def get_stop():
    stop_f = open(stop_words_path, 'r', encoding='utf-8')
    stop_words = []
    for stop_line in stop_f.readlines():
        stop_line = stop_line.strip()
        if not len(stop_line):
            continue
        stop_words.append(stop_line)
    return stop_words


# 获取关键字数据集
def load_dataset(filename):
    stop_words = get_stop()
    with open(filename, 'r', encoding='utf-8') as file:
        data_lines = file.readlines()
        dataset = ''
        for lines in data_lines:
            lines = lines.strip()
            dataset += lines
        dataset.replace(u'\xa0', u' ').replace(u'\u3000', u' ')
        # dataset = jieba.analyse.extract_tags(dataset)
        dataset = jieba.cut(dataset, cut_all=False)
        dataset = list(dataset)
        out_dataset = [data for data in dataset if
                       data not in stop_words and data != '\u3000' and data != '\xa0' and data != ' ']
        return out_dataset


def random_types(small, big):
    randoms = []
    for i in range(TYPES):
        randoms.append(random.randint(small, big))
    return randoms


# 获取某一类文档的文本集
def make_dataset(num, typee, filepath):
    type_dataset = []
    filepath = filepath + typee + "\\"

    for file in os.listdir(filepath):
        filename = filepath + file
        dataset = load_dataset(filename)
        type_dataset.append(dataset)
        num -= 1
        if num == 0:
            break
    print(len(type_dataset))
    return type_dataset


def main():
    data = {}
    i = 0
    filepath = "E:\\嗯冲丶\\大三上\\数据挖掘\\实验二\\THUCNews\\"

    # 读取任意数量的不同类型文本
    randoms = random_types(train_num, train_num)
    # 文档类型列表
    typees = os.listdir(filepath)
    # 总的文本集合，用字典存储，键为标签，值为文本集合
    for typee in typees:
        print(typee)
        data[typee] = make_dataset(randoms[i], typee, filepath)
        i += 1
    # # 目录下所有的文件
    # all_file = os.listdir(filepath)
    # all_samples = set(random.sample(all_file, 150))
    # train_samples = set(random.sample(all_samples, 100))
    # test_samples = set(random.sample(all_samples, 50))

    # 进行测试
    bayes_test = Bayes(data, randoms, typees)
    # filepath = "E:\\嗯冲丶\\大三上\\数据挖掘\\实验二\\THUCNews\\游戏\\406435.txt"
    # f = open(filepath, 'r', encoding='utf-8')
    # file = f.read()
    # test_data = load_dataset(filepath)
    # # test_data = load_dataset(filepath)
    # print("电磁脉冲已就绪")
    # print(test_data)
    # print(bayes_test.get_type(test_data))

    print("开始测试")

    # test_randoms = random_types(test_num, test_num)
    # for t in typees:
    #     filepath = "E:\\嗯冲丶\\大三上\\数据挖掘\\实验二\\THUCNews\\" + t + "\\"
    #     all_files =

    up = 0
    down = 0
    i = 0
    gap = 200
    test_randoms = random_types(test_num, test_num)
    for typee in typees:
        print(typee)
        filepath = "E:\\嗯冲丶\\大三上\\数据挖掘\\实验二\\THUCNews\\" + typee + "\\"
        count = 0
        all_file = os.listdir(filepath)
        for file in all_file:
            gap -= 1
            if gap < 0:
                if count < test_randoms[i]:
                    print("开始读取" + typee + file + str(count))
                    filename = filepath + file
                    test_data = load_dataset(filename)
                    if bayes_test.get_type(test_data) == i:
                        up += 1
                    down += 1
                    count += 1
                else:
                    print(typee + "读取完成，退出")
                    break
        i += 1
    print(up / down) / 10


if __name__ == '__main__':
    main()
    # path = "E:\\嗯冲丶\\大三上\\数据挖掘\\实验二\\THUCNews\\股票\\644606.txt"
    # file = open(path, 'r', encoding='utf-8')
    # data = file.read()
    # before_data = list(jieba.cut(data))
    # print(before_data)
    # after_data = load_dataset(path)
    # print(after_data)
