import pickle
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
dict_path = "../data/20news/20news_keep-2000_dictionary.pkl"
with open(dict_path, 'rb') as f:
    dictionary = pickle.load(f)
# idx_df = sorted(dictionary.dfs.items(), key=lambda p:p[0])

# for word_id, count in idx_df[:20]:
#     print(word_id)

print(dictionary.id2token[0])