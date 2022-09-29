import numpy as np
import pickle
import torch
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.mixture import GaussianMixture


def gmm_fit(dict_path, vecs_path, n_components=20, use_nouns=True, topk=2000):
    '''
    Fit GMM and show results.
    params:
        dict_path: Gensim dictionary filepath
        vecs_path: vectors filepath correspond to dictionary
        n_components: number of clusters
        use_nouns: if use_nouns is True, then only use nouns in vocabularies to fit GMM
        topk: choose topk frequent words to fit GMM
    '''
    with open(dict_path, 'rb') as f:
        dictionary = pickle.load(f)    
    with open(vecs_path, 'rb') as f:
        vecs = pickle.load(f)
    sorted_dfs = sorted(dictionary.dfs.items(), key=lambda p:p[1], reverse=True)[:topk]
    idxs_words = [(p[0], dictionary.id2token[p[0]]) for p in sorted_dfs] 

    chosen_words = []
    chosen_ids = []    
    for idx, w in idxs_words:
        pos = nlp(w)[0].tag_
        if use_nouns == True:
            if pos not in ["NN","NNS","NNP","NNPS"]:
                continue
        chosen_words.append(w)
        chosen_ids.append(idx)            
    print("Chose {} nouns from top-{} frequent words for GMM training.".format(len(chosen_ids), topk))
    chosen_vecs = np.array(vecs)[np.array(chosen_ids)]

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=21, 
        covariance_type="diag", # 注意方差类型需要设置为diag
        reg_covar=1e-5
    ) 
    gmm.fit(chosen_vecs)

    result_path = "../results/gmm/{}_n{}{}_k{}.txt".format(
        dict_path.split('/')[-1].split('.')[0],
        n_components,
        "_nouns" if use_nouns==True else "",
        topk
    )
    f = open(result_path, 'w', encoding='utf8')
    proba_mtx = gmm.predict_proba(chosen_vecs).T
    topk_proba_mtx, words_index_matrix = torch.topk(torch.tensor(proba_mtx), k=20, dim=1)
    words_index_matrix_ori = []
    for indices in words_index_matrix:
        indices_ori = [chosen_ids[i] for i in indices]
        words_index_matrix_ori.append(indices_ori)
    topic_words = []
    for i, words_index in enumerate(words_index_matrix_ori):
        words = [dictionary.id2token[i] for i in words_index]
        topic_words.append(words)
        f.write("topic-{}: {}\n".format(i, words))
    f.close()
    print("Result output:", result_path)        


# def find_best_vecs():
#     '''
#     调整embedding model输出结果的参数，寻找更好的vecs表示
#     '''
#     test_dataset = "wiki103"
#     dict_path = "../data/wiki103/wiki103_no_below-100_dictionary.pkl"
#     with open(dict_path, 'rb') as f:
#         dictionary = pickle.load(f)
#     num_clusters = 20

#     # grid search
#     model_path_list = [None, "../models/bert/wiki103_bert.pt"]
#     n_layers_list = [3, 4]
#     for model_path in model_path_list:
#         for n_layers in n_layers_list:
#             print("model_path={} n_layers={}:".format(model_path, n_layers))
#             print("generating vecs...")
#             embedding_model = PretrainBERT(model_path=model_path, n_layers=n_layers)
#             vecs = []
#             keys = list(dictionary.token2id.keys())
#             # for word in dictionary.token2id.keys():
#             for word in tqdm(keys):
#                 vecs.append(embedding_model.get_embedding(word))

#             # gmm
#             print("training gmm...")
#             gmm = GaussianMixture(
#                     n_components=num_clusters,
#                     random_state=21, 
#                     covariance_type="diag", # 注意方差类型需要设置为diag
#                     reg_covar=1e-5
#             )
#             gmm.fit(vecs)

#             # 将聚类结果输出到文件
#             model_path_flag = 0 if model_path is None else 1
#             result_path = "../data/best_vecs_search/{}_m{}_{}_{}_gmm_results.txt".format(test_dataset, model_path_flag, n_layers, num_clusters)
#             f = open(result_path, 'w')
#             proba_mtx = gmm.predict_proba(vecs).T
#             topk_proba_mtx, words_index_matrix = torch.topk(torch.tensor(proba_mtx), k=20, dim=1)
#             mean_proba = torch.mean(topk_proba_mtx, dim=1)
#             sorted_mean_value, sorted_mean_index = torch.topk(mean_proba, k=mean_proba.shape[0])            
#             topic_words = []
#             for i, words_index in enumerate(words_index_matrix):
#                 words = [dictionary.id2token[i.item()] for i in words_index]
#                 topic_words.append(words)
#             for i in range(len(sorted_mean_index)):
#                 topic_index = sorted_mean_index[i]
#                 mean_value = sorted_mean_value[i]
#                 f.write("topic-{}({}): {}\n".format(topic_index, mean_value.item(), topic_words[topic_index]))
#             f.close()
#             print("已将结果写入：", result_path)
            
#             # save
#             save_path = "../data/best_vecs_search/{}_m{}_{}_vecs.pkl".format(test_dataset, model_path_flag, n_layers)
#             with open(save_path, 'wb') as f:
#                 pickle.dump(vecs, f)
#             print("已保存词向量至:", save_path)

    
if __name__ == "__main__":
    dict_path = "../data/wiki103/wiki103_keep-20000_dictionary.pkl"
    vecs_path = "../data/wiki103/wiki103_keep-20000_vecs.pkl"
    # dict_path = "../data/ag_news/ag_news_keep-10000_dictionary.pkl"
    # vecs_path = "../data/ag_news/ag_news_keep-10000_vecs.pkl"
    # dict_path = "../data/20news/20news_keep-2000_dictionary.pkl"
    # vecs_path = "../data/20news/20news_keep-2000_vecs.pkl"
    n_components = 30
    topk = 3000
    gmm_fit(dict_path, vecs_path, n_components=n_components, topk=topk)