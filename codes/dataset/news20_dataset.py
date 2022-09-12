import os
import re
import pickle
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import nltk
nltk.data.path.append("D:\program_files\\nltk_data")
from nltk.tokenize import word_tokenize

from dataset.dataset import MyDataset
from dataset.vectorize import PretrainEmbeddingModel

class NewsDataset(MyDataset):
    def __init__(self):
        data_source_name = "20news"
        super().__init__(data_source_name)

        data_path = {
            "dict":"../data/20news/20news_no_below-100_dictionary.pkl",
            "docs":"../data/20news/20news_no_below-100_docs.txt",
            "bows":"../data/20news/20news_no_below-100_bows.mm",
            "vecs": "../data/20news/20news_no_below-100_vocab_embeds.pkl"
        }
        with open(data_path["dict"], 'rb') as f:
            self.dictionary = pickle.load(f)
        self.vocab_size = len(self.dictionary)            
        self.token2id = self.dictionary.token2id
        self.id2token = {v:k for k,v in self.token2id.items()}

        self.docs = open(data_path["docs"], encoding='utf-8').read().splitlines()
        self.bows = MmCorpus(data_path["bows"])
        
        # doc matrix
        self.doc_mtx = np.zeros((len(self.bows), self.vocab_size), dtype=np.float)
        for doc_idx, doc in enumerate(self.bows):
            for word_idx, count in doc:
                self.doc_mtx[doc_idx, word_idx] += count       
        
        with open(data_path["vecs"], 'rb') as f:
            self.vecs = pickle.load(f)


def tokenize(text, stopwords):
    words = []
    for word in word_tokenize(text.lower()):
        if len(word) < 2:
            continue
        if word not in stopwords and not re.search(r'=|\'|-|`|\.|[0-9]|_|-|~|\^|\*|\\|\||\+', word):
            words.append(word)
    return words


def main():
    '''
    生成中间数据文件
    '''
    save_dir = "../data/20news"
    raw_docs = []
    no_below = 100
    source_name = "20news"

    for root, dirs, files in os.walk("../data/20news/raw/20news-bydate"):
        for file in files:
            if "DS" in file: continue
            try:
                doc = open(os.path.join(root, file), encoding='utf-8').read()
            except:
                doc = open(os.path.join(root, file), encoding='cp1252').read()
            lines = doc.splitlines()
            new_lines = []
            for i in range(lines.index(""), len(lines)):
                if "@" in lines[i] and "write" in lines[i]:
                    continue
                if "in article" in lines[i].lower():
                    continue
                new_lines.append(lines[i])
            doc = '\n'.join(new_lines)
            raw_docs.append(doc)

    stopwords = open("../data/stopwords.txt").read().splitlines()
    # 初始化文档 字典 词袋 tfidf模型
    print("总文档数:", len(raw_docs))
    print("preparing docs...")
    docs = []
    for doc in raw_docs:
        doc = tokenize(doc, stopwords)
        if len(doc) > 10:
            docs.append(doc)

    print("preparing dictionary...")
    dictionary = Dictionary(docs)
    print("字典大小为：", len(dictionary))
    dictionary.filter_extremes(no_below=no_below)
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}
    print("过滤后字典大小为：", len(dictionary))

    print("preparing bows...")
    bows = [dictionary.doc2bow(doc) for doc in docs]

    print("preparing tfidf model...")
    tfidf_model = TfidfModel(bows)
    tfidfs = [tfidf_model[bow] for bow in bows]
    # for i in range(10, 20):
    #     print(len(bows[i]))
    #     data = [(self.dictionary.id2token[p[0]], p[1]) for p in self.bows[i]]
    #     print(sorted(data, key=lambda p:p[1], reverse=True))

    # 基于BERT模型转换词向量
    print("转换词向量...")
    embedding_model = PretrainEmbeddingModel("bert")
    vecs = []
    for word in dictionary.token2id.keys():
        vecs.append(embedding_model.get_embedding(word))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 字典
    save_path = os.path.join(save_dir, "{}_no_below-{}_dictionary.pkl".format(source_name, no_below))
    with open(save_path, 'wb') as f:
        pickle.dump(dictionary, f)
    print("已保存字典至:", save_path)
    # 文档
    save_path = os.path.join(save_dir, "{}_no_below-{}_docs.txt".format(source_name, no_below))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([' '.join(doc) for doc in docs]))
    print("已保存文档至:", save_path)
    # bow
    save_path = os.path.join(save_dir, "{}_no_below-{}_bows.mm".format(source_name, no_below))
    MmCorpus.serialize(save_path, bows)
    print("已保存bows至:", save_path)
    # tfidf
    save_path = os.path.join(save_dir, "{}_no_below-{}_tfidfs.mm".format(source_name, no_below))
    MmCorpus.serialize(save_path, tfidfs)
    print("已保存tfidfs至:", save_path)        
    # 词向量
    save_path = os.path.join(save_dir, "{}_no_below-{}_vocab_embeds.pkl".format(source_name, no_below))
    with open(save_path, 'wb') as f:
        pickle.dump(vecs, f)
    print("已保存词向量至:", save_path)


if __name__ == '__main__':
    main()
