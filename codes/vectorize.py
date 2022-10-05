import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors, fasttext
from transformers import BertTokenizer, BertModel



class BERTVectorizer:
    def __init__(self, model_path=None, n_layers=4, mode="mean") -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.layers = list(range(-n_layers, 0))
        self.mode = mode
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

    def get_embedding(self, word, return_tensor=False):
        '''
        获取BERT词向量
        param: layers: 指定参与计算词向量的隐藏层，-1表示最后一层
        param: mode: 隐藏层合并策略
        return: torch.Tensor, size=[768]
        '''
        output = self.model(**self.tokenizer(word, return_tensors='pt'))
        specified_hidden_states = [output.hidden_states[i] for i in self.layers]
        specified_embeddings = torch.stack(specified_hidden_states, dim=0)
        # layers to one strategy
        if self.mode == "sum":
            token_embeddings = torch.squeeze(torch.sum(specified_embeddings, dim=0))
        elif self.mode == "mean":
            token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))        
        # tokens to one strategy
        word_embedding = torch.mean(token_embeddings, dim=0)
        if not return_tensor:
            word_embedding = word_embedding.detach().numpy()
        return word_embedding
    



class W2vVectorizer:
    def __init__(self):
        start_time = time.time()
        print("Word2Vec: start loading.")
        self.model = KeyedVectors.load_word2vec_format('../models/embedding_model/GoogleNews-vectors-negative300.bin', binary=True) 
        print("Word2Vec: loaded. use {:.2f}s".format(time.time()-start_time))
    
    def has_word(self, word):
        return self.model.has_index_for(word)

    def get_embedding(self, word):
        return self.model[word]



class GloveVectorizer:
    def __init__(self):
        print("Glove: start loading...")
        start_time = time.time()
        # self.model = api.load('glove-twitter-200')
        self.model = KeyedVectors.load_word2vec_format('../models/embedding_model/glove-twitter-200.gz') 
        print("Glove: loaded. use {:.2f}s".format(time.time()-start_time))
    
    def has_word(self, word):
        return self.model.has_index_for(word)

    def get_embedding(self, word):
        return self.model[word]



class FastTextVectorizer:
    def __init__(self) -> None:
        print("FastText: start loading...")
        start_time = time.time()
        self.model = fasttext.load_facebook_model("")
        print("FastText: loaded. use {:.2f}s".format(time.time()-start_time))

    def has_word(self, word):
        return self.model.has_index_for(word)

    def get_embedding(self, word):
        return self.model[word]



def vectorize(dict_path, embed_model):
    '''
    miss count record:
    + 20news
        - w2v: 10
        - glove: 7
        - fasttext
    + wiki103
        - w2v: 975
        - glove: 369
        - fasttext
    + ag_news
        - w2v: 722
        - glove: 139
        - fasttext
    '''
    dict_dir = os.path.dirname(dict_path)
    dict_name = os.path.basename(dict_path)
    vecs_path = os.path.join(dict_dir, "{}_{}_vecs.pkl".format('_'.join(dict_name.split('.')[0].split('_')[:-1]), embed_model))
    with open(dict_path, 'rb') as f:
        dictionary = pickle.load(f)
    vocab = [word for word in dictionary.token2id]

    if embed_model == "w2v":
        vectorizer = W2vVectorizer()
        embed_dim = 300
    if embed_model == "glove":
        vectorizer = GloveVectorizer()
        embed_dim = 200

    miss_count = 0
    vecs = []
    for word in tqdm(vocab):
        if vectorizer.has_word(word) == False:
            # print(word)
            miss_count += 1
            vecs.append(np.random.rand(embed_dim))
        else:
            vecs.append(vectorizer.get_embedding(word))
    print("miss count:", miss_count)

    with open(vecs_path, 'wb') as f:
        pickle.dump(vecs, f)



if __name__ == "__main__":
    embed_model = "glove"
    dict_path = "../data/ag_news/ag_news_keep-10000_dictionary.pkl"
    vectorize(dict_path, embed_model)