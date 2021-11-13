import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import json

'''超参数'''
FILENAME = "git_zhihu.txt"
WINDOW = 5
SLICE = 100000
NEGATIVE_SAMPLE_M = 1e8
'''全局变量'''
len_w_list = []
'''读取文件'''

# def file_to_str():
#     return open(FILENAME).read()


# print(file_to_str())

'''先做文本处理，将语料库粗加工，根据输入取不同的部分'''

# def txt_process(mode, txt_content):
# txt_part = ''
# if mode == 'train':
#     txt_part = txt_content.split('<train>')[0]  # 训练集
# else:
#     txt_part = txt_content.split('<train>')[1]  # 非训练集
# filename = 'data_' + mode + '.txt'
# with open(filename, 'w') as file_object:
#     file_object.write(txt_content)
#     file_object.flush()
#     file_object.close()
# print(sentence)
# return txt_part


# print(txt_process('train',file_to_str()))
'''制作词汇表和数据字典,只运行一次'''


def make_word_dic(txt_content):
    txt_reshape = txt_content.replace('\n', ' ')
    reshape_content = txt_reshape.strip().split(" ")
    word_table = set(reshape_content)
    word_table = list(word_table)
    '''这里的set是无序的，导致word对应的索引每次训练时候都不一样'''
    word_dic = {word: i for i, word in enumerate(word_table)}
    json.dump(word_table, open('word_table.json', 'w'))
    json.dump(word_dic, open('word_dic.json', 'w'))
    # return word_table, word_dic


# make_word_dic(file_to_str())
'''读取数据字典'''


def get_word_dic(select):
    if select == 'dic':
        f = open('word_dic.json', 'r')
        f = json.load(f)
        return f
    elif select == 'table':
        f = open('word_dic.json', 'r')
        f = json.load(f)
        return f
    else:
        f1 = open('word_dic.json', 'r')
        f1 = json.load(f1)
        f2 = open('word_dic.json', 'r')
        f2 = json.load(f2)
        return f1, f2


# get_word_dic('dic')

'''再做数据处理，生成输入x和标准值y'''


def data_process_():
    # 此时数据是由space和enter分割的，按行读取，并且分割space，变成以句子为单元的word pair,因为句子有完整正确的语法和语义
    # 词汇表一般情况是以整体数据集为准，不按着训练集、测试集划分
    word_pair = []
    word_dic = get_word_dic('dic')
    txt_part = open(FILENAME)
    while True:
        sentence = txt_part.readline()
        if sentence is None:
            break
        sentence = sentence.replace("\n", ' ')
        if sentence.strip() == "":
            break
        words = sentence.strip().split(" ")  # 句子分成word list
        for i in range(0, len(words)):
            center_word = word_dic[words[i]]
            for j in range(1, WINDOW + 1):
                front = i - j
                behind = i + j
                if front < 0 or behind >= len(words):
                    continue
                else:
                    # word_pair_front = [center_word, word_dic[words[front]]]
                    # word_pair_behind = [center_word, word_dic[words[behind]]]
                    word_pair.append([center_word, word_dic[words[front]]])
                    word_pair.append([center_word, word_dic[words[behind]]])
    txt_part.flush()
    txt_part.close()
    data_x = []
    data_y = []
    for k in range(len(word_pair)):
        data_x.append(word_pair[k][0])
        data_y.append(word_pair[k][1])
        '''现在数据集太大，不能一次性转成Tensor，有序集合怎么有序切分，然后能整体放进loader'''
    data_pair = (data_x, data_y)

    return data_pair


# data_process_('train')
'''统计每个词组在文本中出现的频率'''


def neg_sample_init():
    data_set_io = open(FILENAME)
    data_set = data_set_io.read().replace("\n", " ").split(" ")
    data_set_io.flush()
    data_set_io.close()
    count_u = len(data_set)
    word_freq_list = {}
    word_dic = get_word_dic('dic')
    for word in data_set:
        if word in word_freq_list.keys():
            word_freq_list[word] += 1
        else:
            word_freq_list[word] = 1
    for word_frq in word_freq_list.items():
        len_w_list.extend(
            [word_dic[word_frq[0]]] * int(np.round(word_frq[1] ** 0.75 / count_u ** 0.75 * NEGATIVE_SAMPLE_M)))
    # 至此，每个word的长度已经求出
    '''如何将每个词频率的长度与单位长度相绑定 :将1长度均分为M个，按每组词的出现频率将M个单位分堆，并且每堆标号'''


# neg_sample_init()


def get_neg_sample(k):
    # neg = np.random.choice(np.array(len_w_list),size=1,replace=False) '''这个算法会使用过多内存，大部分时间都会报错sigkill'''
    '''负采样采到正案例怎么办?'''
    neg = random.sample(len_w_list, k * 2)
    return neg
