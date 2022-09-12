import os
import re
import numpy as np
import pickle
from xml.etree import ElementTree as et
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.corpora.mmcorpus import MmCorpus
from dataset.dataset import MyDataset
# from dataset import MyDataset
from dataset.vectorize import PretrainEmbeddingModel
# from vectorize import PretrainEmbeddingModel
import nltk
nltk.data.path.append("D:/program_files/nltk_data")
from nltk.tokenize import word_tokenize

class RCV1Dataset(MyDataset):
    def __init__(self):
        data_source_name = "rcv1"
        super().__init__(data_source_name)

        data_path = {
            "dict":"../data/rcv1/rcv1_no_below-100_dictionary.pkl",
            "docs":"../data/rcv1/rcv1_no_below-100_docs.txt",
            "bows":"../data/rcv1/rcv1_no_below-100_bows.mm",
            # "bows":"../data/rcv1/rcv1_no_below-100_tfidfs.mm",
            "vecs": "../data/rcv1/rcv1_no_below-100_vocab_embeds.pkl"
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
    base_dir = "../data/rcv1"
    raw_data_dir = os.path.join(base_dir, "raw/rcv1")
    NO_BELOW = 100 # 词频过滤
    SOURCE_NAME = "rcv1"

    ids = []
    raw_docs = []
    topic_classes = []
    # 70万数据太难顶了，先拿前5万看看
    count = 0
    max_count = 50000
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if "xml" not in file: continue
            filepath = os.path.join(root, file)
            content = open(filepath).read()
            root_ele = et.XML(content)
            ids.append(root_ele.attrib["itemid"])
            for child in root_ele:
                # if child.tag == "title":
                #     print("title:", child.text)
                if child.tag == "text":
                    text = ""
                    for node in child:
                        text += '\n' + node.text
                    raw_docs.append(text)
                if child.tag == "metadata":
                    topics = []
                    for node in child:
                        if "class" not in node.attrib: continue
                        if "topic" in node.attrib["class"]:
                            for code_node in node:
                                topics.append(code_node.attrib["code"])
                    topic_classes.append(topics)
            count += 1
            if count >= max_count: break
        if count >= max_count: break


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
    dictionary.filter_extremes(no_below=NO_BELOW)
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}
    print("过滤后字典大小为：", len(dictionary))

    print("preparing bows...")
    bows = [dictionary.doc2bow(doc) for doc in docs]

    print("preparing tfidf model...")
    tfidf_model = TfidfModel(bows)
    tfidfs = [tfidf_model[bow] for bow in bows]  

    # 基于BERT模型转换词向量
    print("转换词向量...")
    embedding_model = PretrainEmbeddingModel("bert")
    vecs = []
    for word in dictionary.token2id.keys():
        vecs.append(embedding_model.get_embedding(word))      


    # 字典
    save_path = os.path.join(base_dir, "{}_no_below-{}_dictionary.pkl".format(SOURCE_NAME, NO_BELOW))
    with open(save_path, 'wb') as f:
        pickle.dump(dictionary, f)
    print("已保存字典至:", save_path)
    # 文档
    save_path = os.path.join(base_dir, "{}_no_below-{}_docs.txt".format(SOURCE_NAME, NO_BELOW))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([' '.join(doc) for doc in docs]))
    print("已保存文档至:", save_path)
    # bow
    save_path = os.path.join(base_dir, "{}_no_below-{}_bows.mm".format(SOURCE_NAME, NO_BELOW))
    MmCorpus.serialize(save_path, bows)
    print("已保存bows至:", save_path)
    # tfidf
    save_path = os.path.join(base_dir, "{}_no_below-{}_tfidfs.mm".format(SOURCE_NAME, NO_BELOW))
    MmCorpus.serialize(save_path, tfidfs)
    print("已保存tfidfs至:", save_path)        
    # 词向量
    save_path = os.path.join(base_dir, "{}_no_below-{}_vocab_embeds.pkl".format(SOURCE_NAME, NO_BELOW))
    with open(save_path, 'wb') as f:
        pickle.dump(vecs, f)
    print("已保存词向量至:", save_path)



if __name__ == "__main__":
    main()