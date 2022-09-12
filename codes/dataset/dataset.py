import os
import pickle
from collections import Counter
from xml.etree import ElementTree as et
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
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
        super().__init__()
        self.data_source_name = data_source_name

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocab_size)
        src_bow = self.bows[idx] # [[tokenid, count], ...]
        item = list(zip(*src_bow))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        # bow = torch.where(bow>0, 1.0, 0.0)
        doc = self.docs[idx]
        return doc, bow

    def __len__(self):
        return len(self.bows)


