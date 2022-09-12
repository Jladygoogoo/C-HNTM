from math import floor
import pickle
import matplotlib.pyplot as plt
import nltk

dict_path = "../data/20news/20news_no_below-100_dictionary.pkl"
with open(dict_path, 'rb') as f:
    dictionary = pickle.load(f)

# sorted_dfs = list(sorted(dictionary.dfs.items(), key=lambda p:p[1]))
sorted_dfs = list(sorted(dictionary.dfs.items(), key=lambda p:p[1], reverse=True))

dfs = [p[1] for p in sorted_dfs]

print(dfs[floor(0.05*len(dfs))])

# plt.hist(dfs, bins=20)
# plt.show()

for i in range(floor(0.05*len(dfs))):
    p = sorted_dfs[i]
    w = dictionary.id2token[p[0]]
    print("{} {} {}".format(w, nltk.pos_tag([w])[0][1], p[1]))