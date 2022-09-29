'''
AG News (AG’s News Corpus) is a subdataset of AG's corpus of news articles constructed 
by assembling titles and description fields of articles from the 4 largest classes 
(“World”, “Sports”, “Business”, “Sci/Tech”) of AG’s Corpus. The AG News contains 30,000 
training and 1,900 test samples per class.
'''

import os
import json
import pickle
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
import torch
from torch.utils.data import Dataset
with open("data_config.json") as f:
    data_config = json.load(f)


class MyDataset():
    '''
    attributes:
        data_source_name: 数据源,如20news,wiki103,...
        docs: 文档集
        vecs: 与词典序号对应的词向量（BERT） 
        bows: 与文档对应bag-of-word表示
    __getitem__:
        return doc, bow
    '''
    def __init__(self, data_source_name) -> None:
        '''
        param: data_source_name: ["20news", "wiki103", "ag_news"]
        '''
        super().__init__()
        self.data_source_name = data_source_name
        config = data_config[data_source_name]

        with open(config["dict_path"], 'rb') as f:
            self.dictionary = pickle.load(f)
        self.vocab_size = len(self.dictionary)            
        self.token2id = self.dictionary.token2id
        self.id2token = {v:k for k,v in self.token2id.items()}
        with open(config["vecs_path"], 'rb') as f:
            self.vecs = pickle.load(f)

        # train
        self.train_docs = open(config["docs_path"]["train"], encoding='utf-8').read().splitlines()
        self.train_bows = MmCorpus(config["bows_path"]["train"])
        self.train_load_dataset = LoadDataset(self.train_bows, self.vocab_size)
        self.train_doc_mtx = np.zeros((len(self.train_bows), self.vocab_size), dtype=float)
        for doc_idx, doc in enumerate(self.train_bows):
            for word_idx, count in doc:
                self.train_doc_mtx[doc_idx, word_idx] += count   

        # test
        self.test_docs = open(config["docs_path"]["test"], encoding='utf-8').read().splitlines()
        self.test_bows = MmCorpus(config["bows_path"]["test"])
        self.test_load_dataset = LoadDataset(self.test_bows, self.vocab_size)
        self.test_doc_mtx = np.zeros((len(self.test_bows), self.vocab_size), dtype=float)
        for doc_idx, doc in enumerate(self.test_bows):
            for word_idx, count in doc:
                self.test_doc_mtx[doc_idx, word_idx] += count   
    
        


class LoadDataset(Dataset):
    '''
    For DataLoader
    '''
    def __init__(self, bows, vocab_size) -> None:
        super().__init__()
        self.bows = bows
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocab_size)
        src_bow = self.bows[idx] # [[tokenid, count], ...]
        item = list(zip(*src_bow))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        # bow = torch.where(bow>0, 1.0, 0.0)
        return bow

    def __len__(self):
        return len(self.bows)

if __name__ == "__main__":
    data_source_name = "wiki103"
    dataset_wiki103 = MyDataset(data_source_name)