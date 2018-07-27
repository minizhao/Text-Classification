import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle as pkl
import os
import sys


# 语料库类，处理数据,先期只计算词级别的
class cropus(object):
    """docstring for ."""
    def __init__(self, train_file="train_set.csv",test_file="test_set.csv"):
        super(cropus, self).__init__()
        # 我们队长文本变成短文本
        self.max_len=800
        self.min_count=2
        self.batch_size=16
        self.n_class=19

        #最后取得的词库,默认有一个填充字段
        self.words=[]
        self.token2idx={}

        self.train_sents=[]
        self.test_sents=[]

        self.train_labels=[]
        self.test_ids=[]

        # 训练数据和测试数据
        self.train_data=[]
        self.test_data=[]
        self.dev_data=[]

        self.load_file()
        self.make_data()
        # 取其中800个作为开发集
        self.train_data,self.dev_data=self.train_data[:-2000],self.train_data[-2000:]
        self.nums_batch=len(self.train_data)//self.batch_size

    def load_file(self):
        with open("data/train_set.csv") as f:
            for line in tqdm(f.readlines()[1:]):
                line=line.split(',')
                if len(line)!=4 :
                    continue
                # 拆分每行的数据
                id=line[0]
                characters=line[1].split(' ')
                # 去掉太长的文本
                words=line[2].split(' ')
                if len(words)>12000:#太大的句子我们会不要
                    continue

                lable=line[3]

                # 把每个长句子拆分成若干个短句子
                for idx in range((len(words)//self.max_len)+1):
                    self.train_sents.append(words[idx:idx+self.max_len])
                    self.train_labels.append(int(lable)-1)

                # label 是从1开始的，所以我们在原来的基础上减去一个1
                self.words.extend(words)
                self.n_class=len(set(self.train_labels))


        with open("data/test_set.csv") as f:
            for line in tqdm(f.readlines()[1:]):
                line=line.split(',')
                # 拆分每行的数据
                id=line[0]
                characters=line[1].split(' ')
                # 取得最大长度之前的长度
                words=line[2].split(' ')[:self.max_len]

                for idx in range((len(words)//self.max_len)+1):
                    self.test_sents.append(words[idx:idx+self.max_len])
                    self.test_ids.append(int(id))

                # 注意在做测试集的时候没有统计出现的词，以训练集中的数据为准
                # self.words.extend(words)

    def make_data(self):
        if os.path.isfile("save/token2idx.pkl"):
            self.token2idx=pkl.load(open('save/token2idx.pkl','rb'))
            print("words length:",len(self.token2idx))
        else:
            # 更新语料库词典，词频大于２０的
            c=Counter(self.words)
            self.words=[w for w,c in c.items() if c>self.min_count]
            self.words.insert(0,"<pad>")
            self.words.insert(0,"<unk>")
            self.token2idx=dict(zip(self.words,range(len(self.words))))
            pkl.dump(self.token2idx,open('save/token2idx.pkl','wb'))

        for sent,label in tqdm(list(zip(self.train_sents,self.train_labels))):
            ids=[self.token2idx.get(w,1) for w in sent ]+[self.token2idx["<pad>"]]*(self.max_len-len(sent))
            pos=[p+1 for p, w in enumerate(sent)]+[0]*(self.max_len-len(sent))
            self.train_data.append([ids,pos,label])
            # pkl.dump(self.train_data,open('save/train_data.pkl','wb'))
        # 对测试集我们需要补齐2个样本

        for sent,id in tqdm(list(zip(self.test_sents,self.test_ids))):
            ids=[self.token2idx.get(w,1) for w in sent]+[self.token2idx["<pad>"]]*(self.max_len-len(sent))
            pos=[p+1 for p, w in enumerate(sent)]+[0]*(self.max_len-len(sent))
            self.test_data.append([ids,pos,id])

            #为了能够是测试数据预测完，用最后的几个样本数据填充到正＝整个batch

        nums_offset=self.batch_size-len(self.test_data)%self.batch_size
        for _ in range(nums_offset):
            self.test_data.append([ids,pos,id])
        print("nums of offset samples:{}".format(nums_offset))
        # pkl.dump(self.test_data,open('save/test_data.pkl','wb'))
        # 断定测试数据集可以被batch_size整除
        assert (len(self.test_data)%self.batch_size==0)

        np.random.shuffle(self.train_data)
        # 释放数据
        del self.train_labels
        del self.test_ids
        del self.train_sents
        del self.test_sents

    def batch_iterator(self,data):
        for idx in range(0,len(data),self.batch_size):
            batch_data=data[idx:idx+self.batch_size]
            if len(batch_data)<self.batch_size:
                continue
            yield batch_data


if __name__ == '__main__':
    mycropus=cropus()
    print(len(mycropus.train_data))
    print(len(mycropus.train_data[100]))
