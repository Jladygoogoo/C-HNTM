'''
参考代码：https://github.com/awslabs/w-lda
'''
import os
import pickle
import re
import nltk
import scipy.sparse as sparse
import time
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from tqdm import tqdm
nltk.data.path.append("D:/program_files/nltk_data")
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import MyDataset
from dataset.vectorize import PretrainEmbeddingModel



class Wikitext103Dataset(MyDataset):
    def __init__(self):
        data_source_name = "wiki103"
        super().__init__(data_source_name)

        data_path = {
            "dict":"../data/wiki103/wiki103_no_below-100_dictionary.pkl",
            "docs":"../data/wiki103/wiki103_no_below-100_docs.txt",
            "bows":"../data/wiki103/wiki103_no_below-100_bows.mm",
            "vecs": "../data/wiki103/wiki103_no_below-100_vocab_embeds.pkl"
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



class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return [self.wnl.lemmatize(t) for t in doc.split() if len(t) >= 2 and re.match("[a-z].*", t)
                and re.match(token_pattern, t)]

def is_document_start(line):
    if len(line) < 4:
        return False
    if line[0] == '=' and line[-1] == '=':
        if line[2] != '=':
            return True
        else:
            return False
    else:
        return False



def token_list_per_doc(input_dir, token_file):
    lines_list = []
    line_prev = ''
    prev_line_start_doc = False
    with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip()
            if prev_line_start_doc and line:
                # the previous line should not have been start of a document!
                lines_list.pop()
                lines_list[-1] = lines_list[-1] + ' ' + line_prev

            if line:
                if is_document_start(line) and not line_prev:
                    lines_list.append(line)
                    prev_line_start_doc = True
                else:
                    lines_list[-1] = lines_list[-1] + ' ' + line
                    prev_line_start_doc = False
            else:
                prev_line_start_doc = False
            line_prev = line

    print("{} documents parsed!".format(len(lines_list)))
    return lines_list    



def main():
    base_dir = "../data/wiki103"
    input_dir = os.path.join(base_dir, "raw/wikitext-103")
    NO_BELOW = 100 # 词频过滤
    SOURCE_NAME = "wiki103"    
    train_file = 'wiki.train.tokens'
    # val_file = 'wiki.valid.tokens'
    # test_file = 'wiki.test.tokens'
    train_doc_list = token_list_per_doc(input_dir, train_file)
    # val_doc_list = token_list_per_doc(input_dir, val_file)
    # test_doc_list = token_list_per_doc(input_dir, test_file)

    # 初始化文档 字典 词袋 tfidf模型
    print("总文档数:", len(train_doc_list))
    print("preparing docs...")
    tokenize = LemmaTokenizer()
    docs = []
    for doc in train_doc_list:
        doc = tokenize(doc)
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


    # print('Lemmatizing and counting, this may take a few minutes...')
    # start_time = time.time()
    # vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
    #                              tokenizer=LemmaTokenizer(), max_features=20000)

    # train_vectors = vectorizer.fit_transform(train_doc_list)
    # val_vectors = vectorizer.transform(val_doc_list)
    # test_vectors = vectorizer.transform(test_doc_list)

    # print(train_vectors.shape)

    # vocab_list = vectorizer.get_feature_names()
    # vocab_size = len(vocab_list)
    # print('vocab size:', vocab_size)
    # print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))    

    # with open(os.path.join(base_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
    #     for item in vocab_list:
    #         f.write(item+'\n')

    # sparse.save_npz(os.path.join(base_dir, 'wikitext-103_tra.csr.npz'), train_vectors)
    # sparse.save_npz(os.path.join(base_dir, 'wikitext-103_val.csr.npz'), val_vectors)
    # sparse.save_npz(os.path.join(base_dir, 'wikitext-103_test.csr.npz'), test_vectors)            


if __name__ == '__main__':
    main()

