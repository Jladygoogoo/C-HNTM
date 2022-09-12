'''
将数据集转换为nTSNTM和HNTM可接受的输入形式。
得到以下文件：
+ xxx.vocab: 词典文件，每一行为“词 频数”
+ xxx.feat: bow文件，每一行为一个文档，包含多个“词:频数”，用空格隔开
'''

import pickle
from gensim.corpora.mmcorpus import MmCorpus

def get_vocab(dict_path, save_file):
    with open(dict_path, 'rb') as f:
        dictionary = pickle.load(f)
    id2token = dictionary.id2token
    with open(save_file, 'w', encoding="utf8") as f:
        for word_id, count in dictionary.dfs.items():
            # print(id2token[word_id])
            f.write("{} {}\n".format(id2token[word_id], count))

def get_bow_feat(bow_path, save_file):
    bows = MmCorpus(bow_path)
    with open(save_file, 'w') as f:
        for bow in bows:
            f.write(' '.join(["{}:{}".format(p[0],int(p[1])) for p in bow]))
            f.write('\n')




if __name__ == "__main__":
    dict_path = "../../data/wiki103/wiki103_no_below-100_dictionary.pkl"
    vocab_save_path = "../../data/others/wiki103.vocab"
    get_vocab(dict_path, vocab_save_path)
    bow_path = "../../data/wiki103/wiki103_no_below-100_bows.mm"
    feat_save_path = "../../data/others/wiki103.feat"
    get_bow_feat(bow_path, feat_save_path)
