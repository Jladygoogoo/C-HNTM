import os
import wandb
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import torch
import torch.nn.functional as F
# torch.set_printoptions(threshold=np.inf)
# torch.set_printoptions(profile="full")
import pickle
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from sklearn.mixture import GaussianMixture


def get_device(device_id=0):
    if device_id==-2:
        return torch.device("mps")
    elif device_id==-1 or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda:{}".format(device_id))

def get_or_create_path(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return path


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))


def calc_topic_diversity(topic_words):
    '''
    计算话题多样性。
    param: topics_words[List[List[str]]]: [[topic1_w1, topic1_w2, ...], [topic2_w1, ...], ...]
    return[float]: topic_diversity
    '''
    vocab = set()
    for words in topic_words:
        vocab.update(set(words))
    num_words_total = len(topic_words)*len(topic_words[0])
    topic_div = len(vocab)/num_words_total
    return topic_div


def calc_intra_topic_similarity(topic_words_list, w2v_model, mode="avg"):
    '''
    计算话题内部的词相似度。
    '''
    res = 0
    for topic_words in topic_words_list:
        valid_words = []
        for w in topic_words:
            if w2v_model.wv.__contains__(w):
                valid_words.append(w)
        simi_mtx = []
        for i in range(len(valid_words)):
            simi_vec = list(map(lambda w: w2v_model.wv.similarity(valid_words[i], w), valid_words))
            simi_vec[i] = 0
            simi_mtx.append(simi_vec)
        if mode=="avg":
            score_vec = np.mean(np.array(simi_mtx), axis=1)
        elif mode=="max":
            score_vec = np.max(np.array(simi_mtx), axis=1)
        res += np.mean(score_vec)
    return np.mean(res)


def calc_topic_coherence(topic_words, bows, dictionary):
    '''
    计算话题凝聚程度，包括 c_v, c_w2v, c_uci, c_npmi
    params:
        topics_words[List[List[str]]]: [[topic1_w1, topic1_w2, ...], [topic2_w1, ...], ...]
        bows[List[bow]: [bow1, bow2, ...]
        dictionary[gensim.corpora.Dictionary]
    return[dict]:
        {"cv": [cv_score, cv_score_per_topic], "cw2v": ...}
    '''
    # cv_score
    # cv_model = CoherenceModel(topics=topic_words, corpus=bows, dictionary=dictionary, coherence="c_v", processes=1)
    # cv_score = cv_model.get_coherence()

    # # cw2v_score

    # cuci_score
    # cuci_model = CoherenceModel(topics=topic_words, texts=texts, corpus=bows, dictionary=dictionary, coherence="c_uci", processes=1)
    # cuci_score = cuci_model.get_coherence()

    # cnpmi_score
    cnpmi_model = CoherenceModel(topics=topic_words, corpus=bows, dictionary=dictionary, coherence="c_npmi", processes=1)
    cnpmi_score = cnpmi_model.get_coherence()

    score_dict = {
        # "cv": cv_score,
        # "cuci": cuci_score,
        "cnpmi": cnpmi_score
    }

    return score_dict


def get_coherence(beta, doc_mat, N_list=[5,10,15]):
    '''
    coherence: npmi score, npmi = \sum_j\sum_i log[P(wj, wi)/(P(wj)P(wi))] / -logP(wj, wi)
    beta: topic-words 概率矩阵
    doc_mat: doc-words bow矩阵
    N_list: 设置多个topk，结果取平均，如 [5,10,15]
    '''
    topic_size = len(beta)
    doc_size = len(doc_mat)

    average_coherence = 0.0
    for N in N_list:
        # find top words'index of each topic
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(beta[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)

        # compute coherence of each topic
        sum_coherence_score = 0.0
        for i in range(topic_size):
            word_array = topic_list[i]
            sum_score = 0.0
            for n in range(N):
                flag_n = doc_mat[:, word_array[n]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, N):
                    flag_l = doc_mat[:, word_array[l]] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    #if p_n * p_l * p_nl > 0:
                    if p_nl == doc_size:
                        sum_score += 1
                    elif p_n > 0 and p_l>0 and p_nl>0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            sum_coherence_score += sum_score * (2 / (N * N - N))
        sum_coherence_score = sum_coherence_score / topic_size
        average_coherence += sum_coherence_score
    average_coherence /= len(N_list)
    return average_coherence


def get_diversity(beta, N_list=[5,10,15]):
    n_topic, vocab_size = beta.shape
    score = 0
    for N in N_list:
        TU = 0.0
        topic_list = []
        for topic_idx in range(n_topic):
            top_word_idx = np.argpartition(beta[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TU= 0
        cnt =[0 for i in range(vocab_size)]
        for topic in topic_list:
            for word in topic:
                cnt[word]+=1
        for topic in topic_list:
            TU_t = 0
            for word in topic:
                TU_t += 1/cnt[word]
            TU_t /= N
            TU += TU_t
        TU /= n_topic
        score += TU

    score /= len(N_list)    
    return score


def get_CLNPMI_and_OR(root_beta, leaf_beta, root_leaf_mtx, doc_mat, n_children=3, n_words=10):
    '''
    计算CLNPMI和overlap rate(OR)
    root_beta: 顶部主题-词概率矩阵
    leaf_beta: 底部主题-词概率矩阵
    root_leaf_mtx: 顶部主题-底部主题概率矩阵
    doc_mat: 文档-词频矩阵
    return: CLNPMI, OR
    '''
    num_root_topics = len(root_beta)
    doc_size = len(doc_mat)
    N = n_words
    clnmpi_score = 0
    or_score = 0
    for i in range(len(root_beta)):
        clnmpi_score_i = 0
        or_score_i = 0
        parent_words_index = np.argpartition(root_beta[i], -N)[-N:]
        child_topic_index = np.argpartition(root_leaf_mtx[i], -n_children)[-n_children:]
        for j in child_topic_index:
            child_words_index = np.argpartition(leaf_beta[j], -N)[-N:]
            inter = set(parent_words_index).intersection(set(child_words_index))
            or_score_i += len(inter) / N
            parent_words_index_neg = set(parent_words_index) - inter
            child_words_index_neg = set(child_words_index) - inter
            sum_score = 0.0
            for w_n in parent_words_index_neg:
                flag_n = doc_mat[:, w_n] > 0
                p_n = np.sum(flag_n) / doc_size
                for w_l in child_words_index_neg:
                    flag_l = doc_mat[:, w_l] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    #if p_n * p_l * p_nl > 0:
                    if p_nl == doc_size:
                        sum_score += 1
                    elif p_n > 0 and p_l>0 and p_nl>0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            # sum_score *= (2 / (N * N - N))
            sum_score /= (len(parent_words_index_neg) + len(child_words_index_neg))
            clnmpi_score_i += sum_score
        clnmpi_score += clnmpi_score_i / n_children
        or_score += or_score_i / n_children
    clnmpi_score /= num_root_topics
    or_score /= num_root_topics
    return clnmpi_score, or_score

            
def get_topic_specialization(beta, doc_mat):
    tmp = np.sum(doc_mat, axis=0)
    doc_words_mtx = tmp / np.linalg.norm(tmp)
    for i in range(beta.shape[0]):
        beta[i] = beta[i]/np.linalg.norm(beta[i])
    topics_spec = 1 - beta.dot(doc_words_mtx)
    res = np.mean(topics_spec)
    return res


def get_silhouette_coefficient(X, labels):
    '''
    With the ground truth labels not known, a higher Silhouette Coefficient score relates to 
    a model with better defined clusters. 
    param: X: data samples
    param: labels: data sample labels predicted by the clustering model   
    return: score: float
    '''
    score = metrics.silhouette_score(X, labels, metric='euclidean')
    return score

def get_CHI(X, labels):
    '''
    Calinski-Harabasz score, also known as the Variance Ratio Criterion.
    With the ground truth labels not known, a higher Calinski-Harabasz score relates to 
    a model with better defined clusters. 
    param: X: data samples
    param: labels: data sample labels predicted by the clustering model   
    return: score: float
    '''    
    score = metrics.calinski_harabasz_score(X, labels)
    return score


def random_cluster_vec_init(n_clusters=30):
    with open("../models/corpus/dictionary.pkl", 'rb') as f:
        dictionary = pickle.load(f)
    words = list(dictionary.token2id.keys())
    vocab_size = len(words)

    cluster_vec_mtx = np.random.random((vocab_size, n_clusters))
    cluster_vec_mtx = np_softmax(cluster_vec_mtx)    

    word2c_vec = {}
    for i in range(len(words)):
        word2c_vec[words[i]] = cluster_vec_mtx[i]
    with open("../result/word2c_vec.pkl", "wb") as f:
        pickle.dump(word2c_vec, f)
    


def np_softmax(x):
    e = np.exp(x)
    softmax = e / np.sum(e, axis=1).reshape(-1, 1)
    return softmax


def _estimate_log_gaussian_prob(X, means, precisions_chol):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = torch.sum(torch.log(precisions_chol), axis=1)
    # print("log_det.shape", log_det.shape, log_det[0])
    precisions = precisions_chol ** 2
    # print("precisions.shape", precisions.shape, precisions[0])
    log_prob = (torch.sum((means**2 * precisions), 1) -
                2.*torch.mm(X, (means * precisions).T) +
                torch.mm(X**2, precisions.T))
    # print("log_prob.shape", log_prob.shape, log_prob[0])
    return -0.5 * (n_features * np.log(2*np.pi) + log_prob) + log_det


def _estimate_weighted_log_prob(X, means, precisions_chol, weights):
    return _estimate_log_gaussian_prob(X, means, precisions_chol) + torch.log(weights)

def _estimate_log_prob_resp(X, means, precisions_chol, weights):
    weighted_log_prob = _estimate_weighted_log_prob(X, means, precisions_chol, weights)
    log_prob_norm = torch.logsumexp(weighted_log_prob, axis=1)
    # print("weighted_log_prob.shape", weighted_log_prob.shape, weighted_log_prob[0])
    # print("log_prob_norm.shape", log_prob_norm.shape, log_prob_norm[0])
    # with np.errstate(under='ignore'):
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    # print("log_resp.shape", log_resp.shape, log_resp[0])
    return log_prob_norm, log_resp


def _predict_proba_gmm(X, means, precision_chol, weights):
    '''
    计算高斯混合模型下X的概率分布
    默认covariance_type=='diag'
    '''
    _, log_resp = _estimate_log_prob_resp(X, means, precision_chol, weights)
    return torch.exp(log_resp)


def gmm_predict_proba_X(vecs, means, covariances, weights):
    '''
    Evaluate the components' density for each sample(word vector).
    params:
        vecs: data samples, shape = (n_samples, n_features)
        means: the mean of each mixture component in GMM, shape = (n_components, n_features)
        covariances: the covariance of each mixture component in GMM, shape = (n_components, n_features)
        weights: the weights of each mixture components in GMM, shape = (n_components,)
    return: 
        proba_mtx: shape = (n_samples, n_components)
    '''
    precision_chol = 1. / torch.sqrt(covariances)
    _, log_resp = _estimate_log_prob_resp(vecs, means, precision_chol, weights)
    proba_mtx = torch.exp(log_resp)
    return proba_mtx


def gmm_predict_proba_topic(vecs, means, covariances, weights):
    '''
    Evaluate samples distribution over topics for GMM. The inverse of gmm_predict_proba_X.
    params:
        vecs: data samples, shape = (n_samples, n_features)
        means: the mean of each mixture component in GMM, shape = (n_components, n_features)
        covariances: the covariance of each mixture component in GMM, shape = (n_components, n_features)
        weights: the weights of each mixture components in GMM, shape = (n_components,)
    return: 
        proba_mtx: shape = (n_components, n_samples)    
    '''
    X_proba_mtx = gmm_predict_proba_X(vecs, means, covariances, weights)
    topic_proba_mtx = F.softmax(X_proba_mtx.T, dim=1)
    return topic_proba_mtx


def predict_proba_gmm_doc(doc_X, vecs, means, covariances, weights, topic_weights):
    '''
    获得p(t|x)，即在文档的条件下顶层主题的概率，等于文档中所有词的条件下顶层主题概率累积
    params:
        doc_X: 文档词袋表示，size=(batch_size, vocab_size)
        vecs: 词向量集，索引与词袋一致，size=(vocab_size, embed_dim)
        means: gmm模型的means
        covariances: gmm模型的covariances
        weights: gmm模型的weights
    return: 概率矩阵，size=(batch_size, n_topic_root)
    '''
    # precision_chol = 1. / torch.sqrt(covariances)
    # res = []
    # for i in range(len(doc_X)): # 沿batch_size方向遍历
    #     word_vecs = vecs[torch.where(doc_X[i]>0)]
    #     proba_gmm_doc = torch.mean(_predict_proba_gmm(word_vecs, means, precision_chol, weights), axis=0)
    #     res.append(proba_gmm_doc)
    # res = torch.stack(res)
    # return res  
    res = []
    for i in range(len(doc_X)):
        word_vecs = vecs[torch.where(doc_X[i]>0)]
        proba_gmm_words = gmm_predict_proba_X(word_vecs, means, covariances, weights)
        # proba_gmm_doc = 1 - (1-proba_gmm_words-1e-6).prod(dim=0)
        proba_gmm_doc = proba_gmm_words.sum(dim=0) * topic_weights
        proba_gmm_doc = F.softmax(proba_gmm_doc, dim=0).float()
        res.append(proba_gmm_doc)
    return torch.stack(res)


def get_softmax_guassian_multivirate_expectation(mu, logvar, device, eps=0.607):
    '''
    获取经过softmax变换的多元高斯分布概率期望
    param: mu: 多元高斯分布期望; size=(batch_size, latent_dim); type=Torch.tensor
    param: logvar: 多元高斯分布对数方差; size=(batch_size, latent_dim); type=Torch.tensor
    param: eps: 扰动项
    return: expectation; size=(batch_szie, latent_dim); type=Torch.tensor
    '''
    batch_size, latent_dim = mu.shape[0], mu.shape[1]
    transform_mtx_mu = None
    for i in range(latent_dim):
        ones = torch.ones(1, latent_dim-1).to(device)
        diag = -torch.eye(latent_dim-1).to(device)
        tmp_mtx = torch.cat((diag[:i], ones, diag[i:]), 0)
        if transform_mtx_mu == None:
            transform_mtx_mu = tmp_mtx
        else:
            transform_mtx_mu = torch.cat((transform_mtx_mu, tmp_mtx), 1)
    
    transform_mtx_var = None
    for i in range(latent_dim):
        ones = torch.ones(1, latent_dim-1).to(device)
        diag = torch.eye(latent_dim-1).to(device)
        tmp_mtx = torch.cat((diag[:i], ones, diag[i:]), 0)
        if transform_mtx_var == None:
            transform_mtx_var = tmp_mtx
        else:
            transform_mtx_var = torch.cat((transform_mtx_var, tmp_mtx), 1)

    # transform size=(latent_dim, latent_dim*(latent_dim-1))
    sum_mu = torch.matmul(transform_mtx_mu.T, mu.T)
    # print("sum_mu:", torch.sum(torch.where(sum_mu>0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))))
    sum_var = torch.matmul(transform_mtx_var.T, torch.exp(logvar).T)
    re_sigmoid = 1 + torch.exp(- sum_mu / torch.sqrt(1 + eps*eps * sum_var)) # 1/E
    re_sigmoid = torch.sum(re_sigmoid.view(latent_dim, latent_dim-1, batch_size), dim=1).view(latent_dim, batch_size).T
    # print("re_sigmoid:", re_sigmoid)
    
    deno = 2 - latent_dim + re_sigmoid
    deno = torch.where(deno<1e-5, torch.tensor(1.0).to(device), deno)
    expectation = 1.0 / (2 - latent_dim + re_sigmoid)
    # print("expectation:", expectation)
    # print("deno:", 2 - latent_dim + sigmoid)

    # expectation = torch.zeros(batch_size)
    # for i in range(batch_size):
    #     mu_i = mu[i]
    #     var_i = torch.exp(logvar[i])
    #     tmp_sum = 0
    #     for k in range(latent_dim):
    #         for k_ in range(latent_dim):
    #             if k == k_: continue
    #             tmp_sum += 1.0 / 1 + torch.exp(
    #                 -1 * (mu_i[k]-mu_i[k_]) / torch.sqrt(1 + eps*eps * (var_i[k] + var_i[k_]))
    #                 )
    #     expectation[i] = 1.0 / (2 - latent_dim + tmp_sum)
    #     print(expectation[i])
    return expectation



def euclidean_dist(x, y):
    '''
    计算欧氏距离
    参考：https://blog.csdn.net/qq_40816078/article/details/112652548
    '''
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def normalization(mtx):
    '''
    矩阵标准化，将区间变为[0,1]
    '''
    _range = torch.max(mtx, dim=1)[0] - torch.min(mtx, dim=1)[0]
    return (mtx - torch.min(mtx, dim=1)[0].unsqueeze(dim=1)) / _range.unsqueeze(dim=1)



def get_words_cor_mtx(words1, words2, dictionary, vecs):
    idx1 = [dictionary.token2id[w] for w in words1]
    idx2 = [dictionary.token2id[w] for w in words2]
    vecs1 = vecs[torch.Tensor(idx1).long()]
    vecs2 = vecs[torch.Tensor(idx2).long()]
    # 内积得到的结果没有区分性 改用欧氏距离
    cor_mtx = torch.mm(vecs1, vecs2.T)
    cor_mtx = cor_mtx / torch.norm(vecs1, dim=1).unsqueeze(dim=1)
    cor_mtx = cor_mtx / torch.norm(vecs2, dim=1).unsqueeze(dim=0)
    # cor_mtx = euclidean_dist(vecs1, vecs2)
    # print(cor_mtx)
    return cor_mtx


def cal_dependency_score(words_idx1, words_idx2, vecs):
    '''
    Calculate the dependency score of 2 lists of words based on cosine similarity
    param:
        words_idx1: list of int
        words_idx2: list of int
        dictioanry: gensim dictionary
        vecs: Tensor, vectors of all words
    return: score: float
    '''
    vecs1 = vecs[torch.Tensor(words_idx1).long()]
    vecs2 = vecs[torch.Tensor(words_idx2).long()]
    cor_mtx = torch.mm(vecs1, vecs2.T)
    cor_mtx = cor_mtx / torch.norm(vecs1, dim=1).unsqueeze(dim=1)
    cor_mtx = cor_mtx / torch.norm(vecs2, dim=1).unsqueeze(dim=0)
    score = torch.mean(cor_mtx)
    return score



def get_words_by_idx(dictionary, idxs):
    '''
    Get word list by their indices in dictionary
    param: dictionary: gensim dictionary
    param: idxs: list of int
    return: words: list of str
    '''
    words = [dictionary.id2token[i] for i in idxs]
    return words


def min_max_scale(x, feature_range=(0,1)):
    '''
    将x缩放至指定区间
    '''
    scaler = MinMaxScaler(feature_range)
    new_x = scaler.fit_transform(np.array([x]).T).T[0]
    return new_x



def wandb_log(type, log_dict, wandb_flag=True):
    if wandb_flag:
        new_log_dict = {"{}/{}".format(type, k): v for k,v in log_dict.items()}
        wandb.log(new_log_dict)


def print_log(s):
    time_str = datetime.now().strftime("%m-%d %H:%M:%S")
    author = "wnj"
    print("[{}][{}] {}".format(author, time_str, s))


if __name__ == "__main__":
    doc_X = torch.rand((128, 10256))
    doc_X = torch.where(doc_X>0.8, 1, 0)
    vecs = torch.rand((10256, 768))
    means = torch.rand((20, 768))
    covariances = torch.rand((20, 768))
    weights = torch.rand(20)

    print(predict_proba_gmm_doc(doc_X, vecs, means, covariances, weights))