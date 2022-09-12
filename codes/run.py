from collections import Counter
from dataclasses import dataclass
from math import floor
import numpy as np
from tqdm import tqdm
import nltk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from dataset.vectorize import PretrainEmbeddingModel

import utils
from dataset.news20_dataset import NewsDataset
from dataset.wiki103_dataset import Wikitext103Dataset
from dataset.rcv_dataset import RCV1Dataset
from dataset.reuters_dataset import ReutersDataset
from CHNTM import C_HNTM, NVDM_GSM

class C_HNTM_Runner:
    def __init__(self, args, mode="train") -> None:
        '''
        加载参数，模型初始化，预训练聚类模型GMM
        '''
        self.device = utils.get_device(args.device) # device=-1表示cpu，其他表示gpu序号
        print("使用device:", self.device)
        self.save_path = "../models/c_hntm/c_hntm_{}.pkl".format(
                args.data
                # time.strftime("%Y-%m-%d-%H", time.localtime())
                )
        utils.get_or_create_path(self.save_path)
        
        # 加载数据
        self.data_source_name = args.data
        if args.data == "20news":
            self.dataset = NewsDataset()
        if args.data == "wiki103":
            self.dataset = Wikitext103Dataset()
        if args.data == "rcv1":
            self.dataset = RCV1Dataset()
        if args.data == "reuters":
            self.dataset = ReutersDataset()

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.vecs = torch.tensor(np.array(self.dataset.vecs)).to(self.device)
        print("数据集: {}".format(self.data_source_name))
        print("数据集大小: {}, 词典大小: {}".format(len(self.dataset), self.dataset.vocab_size))

        # 加载参数
        self.args = args
        self.encode_dims = [self.dataset.vocab_size, 1024, 512, args.num_topics] # vae模型结构参数
        embed_dim = self.vecs.shape[1]

        self.model = C_HNTM(args.num_clusters, args.num_topics, self.encode_dims, embed_dim, args.hidden_dim, self.device)
        if mode == "train":
            print("pretrain vae ...")
            self.pretrain_vae() # 预训练nvdm模型
            print("pretrain gmm ...")
            self.pretrain_GMM() # 预训练GMM模型
            self.model.to(self.device)
            # self.print_gmm_results()
            print("init dependency matrix ...")
            self.init_depenency_mtx() # 初始化依赖矩阵
            self.model.to(self.device)
            # self.show_hierachical_topic_results()
            print("train model ...")
            # 优化器，设置vae和gmm部分学习率大，dependency学习率小
            dependency_param_ids = list(map(id, self.model.dependency.parameters()))
            base_params = filter(lambda p:id(p) not in dependency_param_ids, self.model.parameters())
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.dependency.parameters(), "lr": 1e-5}, 
                    {"params": base_params, "lr":self.args.lr}
                ]
            )
            

    def pretrain_GMM(self):
        self.gmm = GaussianMixture(
            n_components=self.args.num_clusters,
            random_state=21, 
            covariance_type="diag", # 注意方差类型需要设置为diag
            reg_covar=1e-5
        ) 

        # 筛去高频词汇
        sorted_dfs = sorted(self.dataset.dictionary.dfs.items(), key=lambda p:p[1])
        thres = floor(len(sorted_dfs) * 0.9)
        idx = [p[0] for p in sorted_dfs[:thres]]

        self.gmm.fit(np.array(self.dataset.vecs)[idx])
        # self.gmm.fit(self.dataset.vecs)
        self.model.init_gmm(self.gmm)


    def pretrain_vae(self):
        pretrain_vae_model = NVDM_GSM(self.encode_dims, self.args.hidden_dim)
        pretrain_model_save_path = "../models/pretrain/pretrain_gsm_{}.pkl".format(self.data_source_name)

        if self.args.pretrain == True: # 预训练GSM
            pretrain_vae_model.to(self.device)
            optimizer = torch.optim.Adam(pretrain_vae_model.parameters(), lr=self.args.lr)
            for epoch in range(self.args.pre_num_epochs):
                epoch_loss = []
                for data in tqdm(self.dataloader):
                    optimizer.zero_grad()
                    doc, bows = data
                    x = bows
                    x = x.to(self.device)
                    d_given_theta, mu, logvar = pretrain_vae_model(x)
                    loss = pretrain_vae_model.loss(x, d_given_theta, mu, logvar)

                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item()/len(bows))
                # loss
                avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
                utils.wandb_log("pretrain-loss", {"loss": avg_epoch_loss}, self.args.wandb)
                print("Epoch {} AVG Loss: {:.6f}".format(
                    epoch+1, 
                    avg_epoch_loss))
                # metric
                if (epoch+1) % self.args.metric_log_interval == 0:
                    pretrain_metric_dict = self.evaluate_pretrain(pretrain_vae_model)
                    utils.wandb_log("pretrain-metric", pretrain_metric_dict, self.args.wandb)
                    print("Epoch {} AVG Coherence(NPMI): {:.6f} AVG Diversity: {:.6f}".format(
                        epoch+1, 
                        pretrain_metric_dict["topic_coherence"], 
                        pretrain_metric_dict["topic_diversity"]))
                # topic words
                if (epoch+1) % self.args.topic_log_interval == 0:
                    for i, words in enumerate(self.get_leaf_topic_words(pretrain_vae_model)):
                        print("topic-{}: {}".format(i, words))

            torch.save(pretrain_vae_model.state_dict(), pretrain_model_save_path)
            print("将pretrain model参数保存至", pretrain_model_save_path)

        else: # 直接加载GSM
            print("从pretrain model({})加载参数...".format(pretrain_model_save_path))
            pretrain_vae_model.load_state_dict(torch.load(pretrain_model_save_path))

        # 将pretrain model的参数赋给model
        print("将pretrain model的参数赋给model")
        state_dict = pretrain_vae_model.state_dict()
        self.model.vae.load_state_dict(state_dict, strict=False)


    def init_depenency_mtx(self):
        # 利用初始化的gmm和vae结果对dependency矩阵进行初始化
        root_topic_words = self.get_root_topic_words(self.model)
        leaf_topic_words = self.get_leaf_topic_words(self.model.vae)
        num_root_topics, num_leaf_topics = len(root_topic_words), len(leaf_topic_words)
        dependency_mtx = torch.zeros(num_root_topics, num_leaf_topics)
        for i in range(num_root_topics):
            words1 = root_topic_words[i]
            for j in range(num_leaf_topics):
                words2 = leaf_topic_words[j]
                cor_mtx = utils.get_words_cor_mtx(words1, words2, self.dataset.dictionary, self.vecs)
                dependency_mtx[i][j] = torch.mean(cor_mtx)
        norm_dependency_mtx = utils.normalization(dependency_mtx)
        norm_dependency_mtx = F.softmax(norm_dependency_mtx + 1e-6, dim=1)
        self.model.init_dependency_mtx(norm_dependency_mtx)


    def train(self):
        for epoch in range(self.args.num_epochs):
            epoch_losses = []
            for batch_data in tqdm(self.dataloader):
                batch_size = len(batch_data)
                self.optimizer.zero_grad()
                doc, bow = batch_data # doc-文本序列，bow-文档词袋向量
                x = bow
                x = x.to(self.device)
                logits, mu, logvar = self.model(x)
                loss_dict = self.model.loss(x, logits, mu, logvar, self.vecs)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                epoch_losses.append([v.item() for k, v in loss_dict.items()])
                batch_data_last = bow
             
            # print("=============== dependency ========================")
            # print(self.model.dependency.weight.T)
            # value, index= torch.topk(self.model.dependency.weight.T, k=10, dim=1)
            # print(index)            

            # loss and metric check
            print("EPOCH-{}".format(epoch))
            # loss
            epoch_losses = np.stack(epoch_losses)
            avg_epoch_losses = np.mean(epoch_losses, axis=0)
            avg_epoch_losses_dict = dict(zip(
                            ["l1_loss","l2_loss","l3_loss","l4_loss","l5_loss","dependency_loss","total_loss"], 
                            avg_epoch_losses
                        ))
            utils.wandb_log("loss", avg_epoch_losses_dict, self.args.wandb)
            print("Epoch {} AVG Loss: {:.5f} l1_loss={:.5f} l2_loss={:.5f} l3_loss={:.5f} l4_loss={:.5f} l5_loss={:.5f} dependency_loss={:.5f}".format(
                epoch+1,
                avg_epoch_losses_dict["total_loss"],
                avg_epoch_losses_dict["l1_loss"],
                avg_epoch_losses_dict["l2_loss"],
                avg_epoch_losses_dict["l3_loss"],
                avg_epoch_losses_dict["l4_loss"],
                avg_epoch_losses_dict["l5_loss"],
                avg_epoch_losses_dict["dependency_loss"],
            ))

            self.print_gmm_results()
            self.check_gamma(batch_data_last)

            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate()
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                print("Epoch {} AVG Coherence(NPMI): {:.6f} AVG Diversity: {:.6f}, AVG CLNPMI: {:.6f}, AVG OR: {:.6f}".format(
                    epoch+1, 
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"],
                    metric_dict["clnpmi"],
                    metric_dict["overlap_rate"]))
                print("root topic specialization: {:.6f} | leaf topic specialization: {:.6f}".format(
                    metric_dict["root_topic_specialization"], metric_dict["leaf_topic_specialization"]
                ))
            # topic words           
            if (epoch+1) % self.args.topic_log_interval == 0:
                self.show_hierachical_topic_results()

        torch.save(self.model.state_dict(), self.save_path)
        print("model saved to {}".format(self.save_path))

    
    def load(self, model_path):
        if not torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_path))
        

    def get_leaf_topic_words(self, model):
        '''
        获取底层主题
        '''
        p_matrix_beta = model.decode(torch.eye(self.args.num_topics).to(self.device))
        _, words_index_matrix = torch.topk(p_matrix_beta, k=self.args.topk_words, dim=1)

        topic_words = []
        for words_index in words_index_matrix:
            topic_words.append([self.dataset.dictionary.id2token[i.item()] for i in words_index])        
        return topic_words


    def check_gamma(self, x):
        '''
        待删：检查q(r|x)是否有效，输出文档词和顶层主题分布
        x: 文档的bag-of-word表示
        '''
        print("========== gamma check ===========")
        k = 4
        batch_size = 10
        gamma = utils.predict_proba_gmm_doc(x, self.vecs, self.model.gmm_mu, torch.exp(self.model.gmm_logvar), self.model.weights)
        topk_gamma, topk_gloabl_index = torch.topk(gamma, k=k, dim=1)
        for i in range(batch_size):
            # 输出文档词
            word_index_list = torch.where(x[i]>0)[0]
            print("doc-{} words:".format(i))
            print([self.dataset.dictionary.id2token[i.item()] for i in word_index_list])
            # 输出顶层主题概率分布
            print("global topic distribution:")
            for j in range(k):
                print(topk_gloabl_index[i][j].item(), topk_gamma[i][j].item(), end=" ")
            print("\n")
            


    def print_gmm_results(self):
        '''
        待删：按照词性打印顶层聚类主题
        '''
        proba_mtx = utils.predict_proba_gmm_X(
            self.vecs, 
            self.model.gmm_mu, 
            torch.exp(self.model.gmm_logvar), 
            self.model.weights).T     
        topk_proba_mtx, words_index_matrix = torch.topk(proba_mtx, k=self.args.topk_words, dim=1)
        topk_proba_mtx = torch.softmax(topk_proba_mtx, dim=1)
        # print(topk_proba_mtx)
        topic_words = []
        topic_pos_counter = []
        topic_NN_count = []
        for i, words_index in enumerate(words_index_matrix):
            words = [self.dataset.dictionary.id2token[i.item()] for i in words_index]
            topic_words.append(words)   
            pos_counter = Counter([p[1] for p in nltk.pos_tag(words)])
            topic_pos_counter.append(pos_counter)  
            NN_count = 0
            for pos, count in pos_counter.items():
                if "NN" in pos:
                    NN_count += count
            topic_NN_count.append((i, NN_count))
        
        topic_NN_count = list(sorted(topic_NN_count, key=lambda p:p[1], reverse=True))

        
        print("root topics:")
        for topic_idx, NN_count in topic_NN_count:
            print("topic-{}: {}".format(topic_idx, topic_words[topic_idx]))
            print(topic_pos_counter[topic_idx])
        # for i,words in enumerate(topic_words):
        #     print("topic-{}: {}".format(i, words))
        #     print(pos_counter)


    def get_root_topic_words(self, model):
        '''
        获取聚类主题词（顶层主题）
        '''
        proba_mtx = utils.predict_proba_gmm_X(
            self.vecs, 
            model.gmm_mu, 
            torch.exp(model.gmm_logvar), 
            model.weights)
        proba_mtx = torch.softmax(proba_mtx, dim=0).T # proba_mtx size: (n_component, vocab_size)
        # print(proba_mtx.T)
        topk_proba_mtx, words_index_matrix = torch.topk(proba_mtx, k=self.args.topk_words, dim=1)
        topic_words = []
        for words_index in words_index_matrix:
            topic_words.append([self.dataset.dictionary.id2token[i.item()] for i in words_index])        
        return topic_words        


    
    def show_hierachical_topic_results(self):
        '''
        展示主题模型结果
        '''
        root_topic_words = self.get_root_topic_words(self.model)
        leaf_topic_words = self.get_leaf_topic_words(self.model.vae)
        root_leaf_mtx = F.softmax(self.model.dependency.weight, dim=0).T # size:(n_topic_root,n_topic_leaf)
        _, leaf_index_matrix = torch.topk(root_leaf_mtx, k=5, dim=1)

        for i in range(len(leaf_index_matrix)):
            print("========= root topic-{} =========".format(i))
            print(root_topic_words[i])
            for leaf_index in leaf_index_matrix[i]:
                print("#### leaf topic-{} ####".format(leaf_index))
                print(leaf_topic_words[leaf_index])



    def evaluate_pretrain(self, model):
        '''
        model: vae model(gsm)
        '''
        topic_words = self.get_leaf_topic_words(model)
        coherence_score = utils.get_coherence(model.get_beta(), self.dataset.doc_mtx)
        diversity_score = utils.calc_topic_diversity(topic_words)
        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score
        }

        return metric_dict


    def evaluate(self):
        leaf_topic_words = self.get_leaf_topic_words(self.model.vae)
        X = torch.Tensor(np.array(self.dataset.vecs)).to(self.device)
        root_beta = utils.predict_proba_gmm_X(
            X, 
            self.model.gmm_mu, 
            torch.exp(self.model.gmm_logvar), 
            self.model.weights)
        root_beta = torch.softmax(root_beta, dim=0).T.detach().cpu().numpy()
        leaf_beta = self.model.vae.get_beta()
        root_leaf_mtx = F.softmax(self.model.dependency.weight, dim=0).T.detach().cpu().numpy() # size:(n_topic_root,n_topic_leaf)
        doc_mtx = self.dataset.doc_mtx

        # leaf topic npmi
        coherence_score = utils.get_coherence(leaf_beta, doc_mtx)
        # leaf topic diversity
        diversity_score = utils.calc_topic_diversity(leaf_topic_words)
        # clnpmi, or
        clnpmi_score, or_score = utils.get_CLNPMI_and_OR(root_beta, leaf_beta, root_leaf_mtx, doc_mtx)
        # topic specialization
        root_tc_score = utils.get_topic_specialization(root_beta, doc_mtx)
        leaf_tc_score = utils.get_topic_specialization(leaf_beta, doc_mtx)

        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score,
            "clnpmi": clnpmi_score,
            "overlap_rate": or_score,
            "root_topic_specialization": root_tc_score,
            "leaf_topic_specialization": leaf_tc_score,
        }  

        return metric_dict      