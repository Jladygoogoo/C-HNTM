'''
将数据集转换为nTSNTM和HNTM可接受的输入形式。
得到以下文件：
+ xxx.vocab: 词典文件，每一行为“词 频数”
+ xxx.feat: bow文件，每一行为一个文档，包含多个“词:频数”，用空格隔开
'''
import os
import pickle
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
import torch

def get_vocab(dict_path, save_file):
    with open(dict_path, 'rb') as f:
        dictionary = pickle.load(f)
    id2token = dictionary.id2token
    idx_df = sorted(dictionary.dfs.items(), key=lambda p:p[0])
    with open(save_file, 'w', encoding="utf8") as f:
        for word_id, count in idx_df:
            # print(id2token[word_id])
            f.write("{} {}\n".format(id2token[word_id], count))

def get_bow_feat(bow_path, save_file):
    bows = MmCorpus(bow_path)
    with open(save_file, 'w') as f:
        for bow in bows:
            f.write(' '.join(["{}:{}".format(p[0],int(p[1])) for p in bow]))
            f.write('\n')

def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
      while True:
        line = fin.readline()
        if not line:
          break
        id_freqs = line.split()
        # id_freqs = id_freqs[1:-1]
        doc = {}
        count = 0
        for id_freq in id_freqs:
          items = id_freq.split(':')
          doc[int(items[0])] = int(items[1])
          count += int(items[1])
        if count > 0:
          data_list.append(doc)
          word_count.append(count)

    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.float)
    for doc_idx, doc in enumerate(data_list):
      for word_idx, count in doc.items():
        data_mat[doc_idx, word_idx] += count
    return data_list, data_mat, word_count

def load_vocab(filepath):
    vocabulary = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            word = l.split(' ')[0]
            vocabulary.append(word)
    # print(vocabulary, "\n", len(vocabulary))
    return vocabulary



def cal_cos_distance(m1, m2):
    device = torch.device("cuda:1")
    m1 = torch.Tensor(m1).to(device)
    m2 = torch.Tensor(m2).to(device)
    m1 = m1 / m1.norm(dim=1)[:, None]
    m2 = m2 / m2.norm(dim=1)[:, None]
    res = torch.mm(m1, m2.transpose(0,1))    
    return res


def contruct_graph(train_url, vocabulary, output_path, device):
    size = len(vocabulary)
    train_set, train_data_mat, train_count = data_set(train_url, size)

    n = train_data_mat.shape[0]
    print("n = ", n)
    batch_size = 20000
    top_idx_mtx = []
    for i in range(0, n, batch_size):
        train_data_mat_batch = train_data_mat[i:i+batch_size]
        cos_distance_mtx_batch = cal_cos_distance(train_data_mat_batch, train_data_mat)
        _, top_idx_mtx_batch = torch.topk(cos_distance_mtx_batch, dim=1, k=11)
        del cos_distance_mtx_batch
        torch.cuda.empty_cache()
        top_idx_mtx.append(top_idx_mtx_batch.detach().cpu().numpy())
    top_idx_mtx = np.concatenate(top_idx_mtx)
    top_idx_mtx = top_idx_mtx[:,1:]
    print(top_idx_mtx.shape)

    # distance_index = []
    # for i, d1 in enumerate(train_data_mat):
    #     distance = []
    #     for j, d2 in enumerate(train_data_mat):
    #         distance.append(cal_cos_distance(d1, d2))
    #     distance_index.append(np.argsort(distance)[-10:-2])
    # distance_index = np.array(distance_index)
    # print(distance_index)
    #distance_index = top_idx_mtx.detach().cpu().numpy()
    distance_index = top_idx_mtx
    with open(output_path, 'wb') as f:
        pickle.dump(distance_index, f)


def create_docs_for_rCRP(data_source_name):
    '''
    create docs file for rCRP training.
    '''
    if data_source_name == "20news":
        bow_path = "../data/20news/20news_keep-2000_bows_train.mm"
    elif data_source_name == "wiki103":
        bow_path = "../data/wiki103/wiki103_keep-20000_bows_train.mm"
    elif data_source_name == "ag_news":
        bow_path = "../data/ag_news/ag_news_keep-10000_bows_train.mm"
    output_path = "../data/others/r_crp/r_crp_{}_docs.txt".format(data_source_name)

    bows = MmCorpus(bow_path)
    output = []
    for bow in bows:
        doc = []
        for idx, count in bow:
            doc.extend([str(idx)] * int(count))
        output.append(" ".join(doc))
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))
    print("Save {} docs for rCRP in {}.".format(len(output), output_path))




if __name__ == "__main__":
    # dict_path = "../data/wiki103/wiki103_keep-10000_dictionary.pkl"
    # get_vocab(dict_path, vocab_save_path)
    # train_bow_path = "../data/wiki103/wiki103_keep-10000_bows_train.mm"
    # test_bow_path = "../data/wiki103/wiki103_keep-10000_bows_test.mm"
    train_feat_save_path = "../data/others/ag_news_train.feat"
    # test_feat_save_path = "../data/others/wiki103_test.feat"
    # get_bow_feat(train_bow_path, train_feat_save_path)
    # get_bow_feat(test_bow_path, test_feat_save_path)

    # dataset = "wiki103"
    # data_dir = "../data/others"
    # vocabulary_path = os.path.join(data_dir, "{}.vocab".format(dataset))
    # vocabulary = load_vocab(vocab_save_path)
    # output_path = os.path.join(data_dir, "{}_train_neighbors.pickle".format(dataset))    
    # contruct_graph(train_feat_save_path, vocabulary, output_path)

    # contruct_graph()

    data = "20news"
    create_docs_for_rCRP(data)
