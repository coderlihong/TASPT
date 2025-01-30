import random
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer
import pandas as pd
from nltk import sent_tokenize


class Mydataset(Dataset):
    def __init__(self, config, data, label2idx):
        super().__init__()
        self.config = config
        self.data = data
        self.special_tokens_mask = None
        self.mlm_probability = 0.15
        self.prob_replace_mask = 0.8
        self.prob_replace_rand = 0.1
        self.prob_keep_ori = 0.1
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_tokenizer)


        self.label2idx = label2idx

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):
        input_a, label_a, input_b, label_b = self.data[item]
        relative_distance = self.label2idx[str(label_b)] - self.label2idx[str(label_a)]

        sentence_len = int(self.config.max_len/2)
        input_ids_a = self.tokenizer.encode_plus(str(input_a), max_length=int(sentence_len), padding='max_length',
                                                 add_special_tokens=True,
                                                 truncation=True)['input_ids']
        input_ids_b = self.tokenizer.encode_plus(str(input_b), max_length=int(sentence_len), padding='max_length',
                                            add_special_tokens=False,
                                            truncation=True)['input_ids']

        masked_input_ids_b, masked_labels_b = self.mask_tokens(torch.tensor(input_ids_b))

        inputs = input_ids_a+masked_input_ids_b.tolist()
        inputs = torch.tensor(inputs)
        mlm_labels = [-100] * len(input_ids_a) + masked_labels_b.tolist()

        attention_masks = inputs != 0
        mlm_labels = torch.tensor(mlm_labels)
        relative_distance, label_a, label_b = (torch.tensor(relative_distance, dtype=torch.float16),
                                               torch.tensor(self.label2idx[str(label_a)]),
                                               torch.tensor(self.label2idx[str(label_b)]))
        labels = torch.stack([label_a, label_b])

        return inputs, mlm_labels, attention_masks, relative_distance, labels

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if self.special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        else:
            special_tokens_mask = self.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.prob_replace_rand / (1 - self.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        mlm_labels = labels
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, mlm_labels


import zhon


class DataObj():
    def __init__(self, sentence, label):
        super().__init__()
        self.sentence = sentence
        self.label = label

    def __str__(self):
        return f'{self.sentence} {self.label}'


def read_data(path):
    df = pd.read_excel(path)
    dfdata = df.values
    data_list = []
    # sample_num = 50000
    for i in dfdata:

        sentence_split = i[0]
        data_list.append(DataObj(sentence_split, i[1]))
    return data_list


def sample_data(data_list):
    data_sampled = []
    epochs = 1
    random.shuffle(data_list)
    for epoch in range(epochs):
        for i in data_list:
            input_a, label_a = i.sentence, i.label
            while 1:
                rand_num = random.randint(0, len(data_list) - 1)
                sample = data_list[rand_num]
                if sample.label != i.label:
                    data_sampled.append((input_a, label_a, sample.sentence, sample.label))
                    break


    return data_sampled


def get_label2idx():
    f = open('../data/zh/label.txt', encoding='utf-8').readlines()
    f = {i.strip(): iter for iter, i in enumerate(f)}
    return f


def get_loader(data, config, label2idx):
    myset = Mydataset(config, data, label2idx)
    loader = DataLoader(myset, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=3)

    return loader, label2idx


from config import *


def combine_all(config):
    data_list = read_data('../data/zh/trunc_train.xlsx')
    data_sampled = sample_data(data_list)
    label2idx = get_label2idx()
    print(label2idx)
    loader, label2idx = get_loader(data_sampled, config, label2idx)
    return loader, label2idx


class MyCLSdataset(Dataset):
    def __init__(self, tokenizer, corpus, labels, label2dict):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.corpus = corpus
        self.labels = labels
        self.label2dict = label2dict

    def __getitem__(self, idx):
        corpus = self.corpus[idx]

        label = self.label2dict[str(self.labels[idx])]

        corpus = self.tokenizer.encode_plus(str(corpus), add_special_tokens=True, padding='max_length', truncation=True,
                                            max_length=512)
        input_ids = corpus['input_ids']
        attention_mask = corpus['attention_mask']

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label)

    def __len__(self):
        return len(self.corpus)

class MyLLMdataset(Dataset):
    def __init__(self, config, corpus, labels, label2dict):
        super().__init__()
        self.config = config
        self.bert_tokenizer =AutoTokenizer.from_pretrained(config.bert_tokenizer,trust_remote_code=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_tokenizer,trust_remote_code=True)
        self.corpus = corpus
        self.labels = labels
        self.label2dict = label2dict

    def __getitem__(self, idx):
        corpus = self.corpus[idx]

        bert_corpus= self.bert_tokenizer.encode_plus(corpus, add_special_tokens=True, padding='max_length', truncation=True,
                                            max_length=512)
        bert_input_ids = bert_corpus['input_ids']
        bert_masks=  bert_corpus['attention_mask']

        corpus = self.llm_tokenizer.encode_plus(corpus, add_special_tokens=True, padding='max_length', truncation=True,
                                            max_length=512)
        tgt_str = f"这段文本所属的年代是：{self.labels[idx]}。"
        tgt =self.llm_tokenizer.encode_plus(tgt_str, add_special_tokens=True, truncation=True,
                                            max_length=16)
        llm_input_ids = corpus['input_ids']
        llm_attention_mask = corpus['attention_mask']
        llm_tgt = tgt['input_ids'] +[-100]*(16-len(tgt['input_ids']))

        label = self.label2dict[self.labels[idx]]

        return torch.tensor(bert_input_ids),torch.tensor(bert_masks),torch.tensor(llm_input_ids), torch.tensor(llm_attention_mask), torch.tensor(llm_tgt)

    def __len__(self):
        return len(self.corpus)

def getdata(path, mode):
    if path.find('zh')!=-1:
        df = pd.read_excel(f'{path}/trunc_{mode}.xlsx')
        corpus = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        df = pd.read_excel(f'{path}/{mode}.xlsx')
        corpus = df['txt'].tolist()
        labels = df['year'].tolist()
    return corpus, labels


def load_dataloader(config, label2idx):
    idx2label = {label2idx[item]: item for idx, item in enumerate(label2idx)}

    train_corpus, train_labels = getdata('../data/zh', 'train')
    valid_corpus, valid_labels = getdata('../data/zh', 'valid')
    test_corpus, test_labels = getdata('../data/zh', 'test')
    time_lines = ['西汉', '东汉', '西晋', '南朝宋', '南朝梁', '北朝齐', '唐', '后晋', '宋', '元', '明', '清']
    dis2idx = {item: idx for idx, item in enumerate(time_lines)}
    train_dataset = MyCLSdataset(config.bert_tokenizer, train_corpus, train_labels, label2idx)
    valid_dataset = MyCLSdataset(config.bert_tokenizer, valid_corpus, valid_labels, label2idx)
    test_dataset = MyCLSdataset(config.bert_tokenizer, test_corpus, test_labels, label2idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_raw_data(config, label2idx):
    idx2label = {label2idx[item]: item for idx, item in enumerate(label2idx)}

    train_corpus, train_labels = getdata('../data/zh', 'train')
    valid_corpus, valid_labels = getdata('../data/zh', 'valid')
    test_corpus, test_labels = getdata('../data/zh', 'test')



    return train_corpus, train_labels, valid_corpus,valid_labels,test_corpus,test_labels
def load_llm_dataloader(config, label2idx):
    idx2label = {label2idx[item]: item for idx, item in enumerate(label2idx)}

    train_corpus, train_labels = getdata('../data', 'train')
    valid_corpus, valid_labels = getdata('../data', 'val')
    test_corpus, test_labels = getdata('../data', 'test')
    time_lines = ['西汉', '东汉', '西晋', '南朝宋', '南朝梁', '北朝齐', '唐', '后晋', '宋', '元', '明', '清']
    dis2idx = {item: idx for idx, item in enumerate(time_lines)}
    train_dataset = MyLLMdataset(config, train_corpus, train_labels, label2idx)
    valid_dataset = MyLLMdataset(config, valid_corpus, valid_labels, label2idx)
    test_dataset = MyLLMdataset(config, test_corpus, test_labels, label2idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=2, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False)

    return train_loader, valid_loader, test_loader

import pkuseg

# 示例中文文本

import re
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
from nltk import sent_tokenize
def dataset_cal():
    train_corpus, train_labels = getdata('../data/en', 'train')
    valid_corpus, valid_labels = getdata('../data/en', 'valid')
    test_corpus, test_labels = getdata('../data/en', 'test')
    corpus = train_corpus+valid_corpus+test_corpus
    labels = train_labels+valid_labels+test_labels

    year_set = list(set(labels))
    yearset_dict = {item :{
        'all_tokens':0,
        'list_len':0,
        'senten_num':0
    } for item in year_set}
    for i,j in zip(corpus,labels):

        yearset_dict[j]['all_tokens'] +=len(list(str(i)))

        yearset_dict[j]['list_len'] +=1
        yearset_dict[j]['senten_num'] += len( sent_tokenize(str(i)))
        yearset_dict[j]['averge'] = yearset_dict[j]['all_tokens']/yearset_dict[j]['list_len']

    for i in yearset_dict:
        print(i)
        print(yearset_dict[i])
        print()



if __name__ == '__main__':
    dataset_cal()
