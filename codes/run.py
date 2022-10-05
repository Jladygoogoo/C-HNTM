import os
import time
import pickle
from collections import Counter
from math import floor
import numpy as np
from tqdm import tqdm
from datetime import datetime

import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

import utils
from my_dataset import MyDataset
from CHNTM import C_HNTM, NVDM_GSM

class C_HNTM_Runner:
    def __init__(self, args, mode="train") -> None:
        '''
        加载参数，模型初始化，预训练聚类模型GMM
        '''
        self.args = args
        self.device = utils.get_device(args.device) # device=-1表示cpu，其他表示gpu序号
        utils.print_log("device: {}".format(self.device))
        
        # load data
        self.data_source_name = args.data
        self.dataset = MyDataset(self.data_source_name)
        self.vecs = torch.tensor(np.array(self.dataset.vecs), dtype=torch.float).to(self.device)
        self.train_dataloader = DataLoader(self.dataset.train_load_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.dataset.test_load_dataset, batch_size=args.batch_size, shuffle=True)

        utils.print_log("Loading data from dataset-[{}]".format(self.data_source_name))
        utils.print_log("dictionary size: {} | train size: {} | test size: {}".format(
            self.dataset.vocab_size, len(self.dataset.train_load_dataset), len(self.dataset.test_load_dataset)
        ))

        # model args
        self.encode_dims = [self.dataset.vocab_size, 1024, 512, args.num_topics] # vae模型结构参数
        embed_dim = self.vecs.shape[1]

        self.model = C_HNTM(args.num_clusters, args.num_topics, self.encode_dims, embed_dim, args.hidden_dim, self.device)
        if mode == "train":
            if self.args.resume_train_path is not None:
                self.model.load_state_dict(torch.load(self.args.resume_train_path))
                utils.print_log("Resume training CHNTM from: {}".format(self.args.resume_train_path))
            else:
                utils.print_log("Pretraining vae ...")
                self.pretrain_vae()
                utils.print_log("Pretraining gmm ...")
                self.pretrain_GMM() # 预训练GMM模型
                self.model.to(self.device)
                # self.print_gmm_results()
                # return
                utils.print_log("Initialize dependency matrix ...")
                self.init_depenency_mtx() # 初始化依赖矩阵
            self.model.to(self.device)
            self.show_hierachical_topic_results()
            utils.print_log("Training model ...")
            
            # set different lr for dependency matrix and other parameters
            dependency_param_ids = list(map(id, self.model.dependency.parameters()))
            base_params = filter(lambda p:id(p) not in dependency_param_ids, self.model.parameters())
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.dependency.parameters(), "lr": self.args.d_lr}, 
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

        # use topk frequent words and choose nouns for gmm training
        topk = 2000
        sorted_dfs = sorted(self.dataset.dictionary.dfs.items(), key=lambda p:p[1], reverse=True)[:topk]
        idxs_words = [(p[0], self.dataset.dictionary.id2token[p[0]]) for p in sorted_dfs] 

        chosen_words = []
        chosen_ids = []    
        for idx, w in idxs_words:
            pos = nlp(w)[0].tag_
            if pos in ["NN","NNS","NNP","NNPS"]:
                # print(w, pos)
                chosen_words.append(w)
                chosen_ids.append(idx)            
        utils.print_log("Chosen {} nouns from top-{} frequent words for GMM training.".format(len(chosen_ids), topk))

        vecs = np.array(self.dataset.vecs)[np.array(chosen_ids)]
        self.gmm.fit(vecs)
        utils.print_log("Copying pretrained GMM parameters to CHNTM...")
        self.model.init_gmm(self.gmm)


    def pretrain_vae(self):
        # args check
        if self.args.pretrain == True and self.args.resume_pretrain == True:
            raise ValueError("参数pretrain和resume_pretrain不能同时为True")

        pretrain_vae_model = NVDM_GSM(self.encode_dims, self.args.hidden_dim)
        if self.args.pretrain == True:
            utils.print_log("Pretrain VAE from scratch...")
        elif self.args.resume_pretrain == True:
            if self.args.resume_pretrain_path is None:
                raise ValueError("如果resume_pretrain=True，则参数resume_pretrain_path必须提供")
            pretrain_vae_model.load_state_dict(torch.load(self.args.resume_pretrain_path))
            utils.print_log("Resume pretraining VAE from: {}".format(self.args.resume_pretrain_path))
        else:
            if self.args.pretrain_model_load_path is None:
                raise ValueError("如果不进行预训练VAE，需要提供VAE模型路径参数pretrain_model_load_path")

        # pretrain
        if self.args.pretrain == True or self.args.resume_pretrain == True:
            timestamp = datetime.now().strftime("%y%m%d%H%M")
            pretrain_model_save_path = "../models/pretrain/pretrain_gsm_{}_{}.pt".format(self.data_source_name, timestamp)
            utils.print_log("Pretrained VAE will be saved in {}".format(pretrain_model_save_path))
            checkpoint_dir = "../models/pretrain/.ckpt/"
            utils.get_or_create_path(pretrain_model_save_path)
            utils.get_or_create_path(checkpoint_dir)            

            pretrain_vae_model.to(self.device)
            optimizer = torch.optim.Adam(pretrain_vae_model.parameters(), lr=self.args.pretrain_lr)
            for epoch in range(self.args.pre_num_epochs):
                epoch_loss = []
                for data in self.train_dataloader:
                    optimizer.zero_grad()
                    x = data
                    x = x.to(self.device)
                    d_given_theta, mu, logvar = pretrain_vae_model(x)
                    loss = pretrain_vae_model.loss(x, d_given_theta, mu, logvar)

                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item()/len(x))

                utils.print_log("Epoch-[{}]".format(epoch))
                # loss
                avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
                utils.wandb_log("pretrain/loss", {"loss": avg_epoch_loss}, self.args.wandb)
                utils.print_log("Loss: {:.6f}".format(avg_epoch_loss))

                
                # metric
                if (epoch+1) % self.args.metric_log_interval == 0:
                    # train data
                    pretrain_metric_dict = self.evaluate_pretrain(pretrain_vae_model, self.dataset.train_doc_mtx)
                    utils.wandb_log("pretrain/metric", pretrain_metric_dict, self.args.wandb)
                    utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                        pretrain_metric_dict["topic_coherence"], 
                        pretrain_metric_dict["topic_diversity"]))

                # test
                if (epoch+1) % self.args.test_interval == 0:
                    utils.print_log("======= Test =======")
                    with torch.no_grad():
                        loss_list = []
                        for batch_data in self.test_dataloader:
                            x = batch_data
                            x = x.to(self.device)
                            d_given_theta, mu, logvar = pretrain_vae_model(x)
                            loss = pretrain_vae_model.loss(x, d_given_theta, mu, logvar)
                            loss_list.append(loss.item()/len(x))
                        avg_loss = np.mean(loss_list)
                    utils.wandb_log("pretrain/test/loss", {"loss": avg_loss}, self.args.wandb)
                    utils.print_log("Loss: {:.6f}".format(avg_loss))                
                    # test data
                    pretrain_metric_dict = self.evaluate_pretrain(pretrain_vae_model, self.dataset.test_doc_mtx)
                    utils.wandb_log("pretrain/test/metric", pretrain_metric_dict, self.args.wandb)
                    utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                        pretrain_metric_dict["topic_coherence"], 
                        pretrain_metric_dict["topic_diversity"]))   
                    utils.print_log("====================")                                         

                # topic words
                if (epoch+1) % self.args.topic_log_interval == 0:
                    utils.print_log("Topic results:")
                    for i, idxs in enumerate(self.get_local_topic_words_idx(pretrain_vae_model)):
                        print("topic-{}: {}".format(i, utils.get_words_by_idx(self.dataset.dictionary, idxs)))

                # checkpoint
                if (epoch+1) % self.args.checkpoint_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, "pretrain_gsm_{}_{}_{}.ckpt".format(
                        self.data_source_name, timestamp, epoch+1))
                    torch.save(pretrain_vae_model.state_dict(), ckpt_path)
                    utils.print_log("Checkpoint saved: {}".format(ckpt_path))                             

            torch.save(pretrain_vae_model.state_dict(), pretrain_model_save_path)
            utils.print_log("Pretrained VAE saved: {}".format(pretrain_model_save_path))

        # not pretrain, load existed vae
        else:
            pretrain_vae_model.load_state_dict(torch.load(self.args.pretrain_model_load_path))
            utils.print_log("Pretrained VAE Loaded: {}".format(self.args.pretrain_model_load_path))

        utils.print_log("Copying pretrained VAE parameters to CHNTM...")
        state_dict = pretrain_vae_model.state_dict()
        self.model.vae.load_state_dict(state_dict, strict=False)


    def init_depenency_mtx(self):
        '''
        Initialize the dependency matrix of CHNTM. Since the range of model.dependency is (-∞,+∞), 
        there is no need to use softmax. 
        '''
        global_topic_words_idx = self.get_global_topic_words_idx(self.model)
        local_topic_words_idx = self.get_local_topic_words_idx(self.model.vae)
        num_global_topics, num_local_topics = len(global_topic_words_idx), len(local_topic_words_idx)
        
        dependency_mtx = torch.zeros(num_global_topics, num_local_topics)
        for i in range(num_global_topics):
            words_idx1 = global_topic_words_idx[i]
            for j in range(num_local_topics):
                words_idx2 = local_topic_words_idx[j]
                dependency_score = utils.cal_dependency_score(words_idx1, words_idx2, self.vecs)
                dependency_mtx[i][j] = dependency_score
        # print(dependency_mtx)
        self.model.init_dependency_mtx(dependency_mtx)



    def train(self):
        timestamp = time.strftime("%y%m%d%H%M", time.localtime())
        model_save_path = "../models/c_hntm/c_hntm_{}_{}.pt".format(self.data_source_name, timestamp)
        checkpoint_dir = "../models/c_hntm/.ckpt/"
        utils.get_or_create_path(model_save_path)        
        utils.get_or_create_path(checkpoint_dir)        

        for epoch in range(self.args.num_epochs):
            epoch_losses = []
            for batch_data in self.train_dataloader:
                batch_size = len(batch_data)
                self.optimizer.zero_grad()
                bow = batch_data
                x = bow
                x = x.to(self.device)
                logits, mu, logvar = self.model(x)
                # global_topic_weights = torch.tensor(self.get_global_topic_weights()).to(self.device)
                global_topic_weights = torch.ones(self.args.num_clusters).to(self.device)

                loss_dict = self.model.loss(x, logits, mu, logvar, self.vecs, global_topic_weights)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                epoch_losses.append([v.item() for k, v in loss_dict.items()])
                batch_data_last = bow
                # self.check_gamma(batch_data_last[:1], global_topic_weights)      

            utils.print_log("EPOCH-[{}]".format(epoch))
            # loss
            epoch_losses = np.stack(epoch_losses)
            epoch_losses = np.mean(epoch_losses, axis=0)
            epoch_losses_dict = dict(zip(
                            ["l1_loss","l2_loss","l3_loss","l4_loss","l5_loss","dependency_loss","total_loss"], 
                            epoch_losses
                        ))
            utils.wandb_log("loss", epoch_losses_dict, self.args.wandb)
            utils.print_log("Loss: {:.5f} | l1_loss={:.5f} | l2_loss={:.5f} | l3_loss={:.5f} | l4_loss={:.5f} | l5_loss={:.5f} | dependency_loss={:.5f}".format(
                epoch_losses_dict["total_loss"],
                epoch_losses_dict["l1_loss"],
                epoch_losses_dict["l2_loss"],
                epoch_losses_dict["l3_loss"],
                epoch_losses_dict["l4_loss"],
                epoch_losses_dict["l5_loss"],
                epoch_losses_dict["dependency_loss"],
            ))

            # self.print_gmm_results()
            # self.check_gamma(batch_data_last[:10], global_topic_weights)

            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.dataset.train_doc_mtx)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f} | CLNPMI: {:.6f} | OR: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"],
                    metric_dict["clnpmi"],
                    metric_dict["overlap_rate"]))
                utils.print_log("root topic specialization: {:.6f} | leaf topic specialization: {:.6f}".format(
                    metric_dict["global_topic_specialization"], metric_dict["local_topic_specialization"]
                ))
                utils.print_log("Silhouette Coefficient: {:.6f} | CHI: {:.6f}".format(
                    metric_dict["silhouette_score"], metric_dict["chi_score"]
                ))

            # test
            if (epoch+1) % self.args.test_interval == 0:
                utils.print_log("======= Test =======")
                with torch.no_grad():
                    loss_list = []
                    for batch_data in self.test_dataloader:
                        x = batch_data
                        x = x.to(self.device)
                        logits, mu, logvar = self.model(x)
                        global_topic_weights = torch.ones(self.args.num_clusters).to(self.device)

                        loss_dict = self.model.loss(x, logits, mu, logvar, self.vecs, global_topic_weights)
                        loss = loss_dict["loss"].item()
                        loss_list.append(loss)
                    avg_loss = np.mean(loss_list)
                utils.wandb_log("test/loss", {"loss": avg_loss}, self.args.wandb)
                utils.print_log("Loss: {:.6f}".format(avg_loss))

                metric_dict = self.evaluate(self.dataset.test_doc_mtx)
                utils.wandb_log("test/metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f} | CLNPMI: {:.6f} | OR: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"],
                    metric_dict["clnpmi"],
                    metric_dict["overlap_rate"]))
                utils.print_log("root topic specialization: {:.6f} | leaf topic specialization: {:.6f}".format(
                    metric_dict["global_topic_specialization"], metric_dict["local_topic_specialization"]
                ))  
                utils.print_log("====================")            

            # topic words           
            if (epoch+1) % self.args.topic_log_interval == 0:
                self.show_hierachical_topic_results()

            # checkpoint
            if (epoch+1) % self.args.checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, "chntm_{}_{}_{}.ckpt".format(
                    self.data_source_name, timestamp, epoch+1))
                torch.save(self.model.state_dict(), ckpt_path)
                utils.print_log("Checkpoint saved: {}".format(ckpt_path))                   

        torch.save(self.model.state_dict(), model_save_path)
        utils.print_log("model saved to {}".format(model_save_path))

    
    def load(self, model_path):
        if not torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_path))
        


    def check_gamma(self, x, global_topic_weights):
        '''
        待删：检查q(r|x)是否有效，输出文档词和顶层主题分布
        x: 文档的bag-of-word表示
        '''
        utils.print_log("========== gamma check ===========")
        k = 4
        batch_size = len(x)
        gamma = utils.predict_proba_gmm_doc(x, self.vecs, self.model.gmm_mu, torch.exp(self.model.gmm_logvar), self.model.gmm_weights, global_topic_weights)
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



    
            
    def get_global_topic_tfidf_score(self):
        '''
        获取当前各聚类主题下的平均tfidf值
        '''
        words_index_matrix = self.get_global_topic_words_idx(self.model)
        tfidf_score = []
        for words_index in words_index_matrix:
            topic_avg_tfidfs = np.mean([self.dataset.token_tfidf[idx.item()] for idx in words_index])  
            tfidf_score.append(topic_avg_tfidfs)
        return tfidf_score


    def get_global_topic_weights(self):
        '''
        基于tfidf，获取聚类主题的权重
        '''
        global_topic_tfidf = self.get_global_topic_tfidf_score()
        global_topic_weights = utils.min_max_scale(global_topic_tfidf, feature_range=(0.2, 1))
        return global_topic_weights



    def print_gmm_results(self):
        '''
        待删：按照词性打印顶层聚类主题
        '''

        with open("../data/best_vecs_search/avg_tfidfs.pkl", 'rb') as f:
            avg_tfidfs = pickle.load(f)

        proba_mtx = utils.gmm_predict_proba_topic(
            self.vecs, 
            self.model.gmm_mu, 
            torch.exp(self.model.gmm_logvar), 
            self.model.gmm_weights)
        topk_proba_mtx, words_index_matrix = torch.topk(proba_mtx, k=self.args.topk_words, dim=1)

        topic_words = []
        tfidf_score = []
        for i, words_index in enumerate(words_index_matrix):
            words = [self.dataset.dictionary.id2token[j.item()] for j in words_index]
            topic_avg_tfidfs = np.mean([avg_tfidfs[j.item()] for j in words_index])
            topic_words.append(words)   
            tfidf_score.append((i, topic_avg_tfidfs))

        sorted_tfidf_score = sorted(tfidf_score, key=lambda p:p[1], reverse=True)
        print("root topics:")
        for p in sorted_tfidf_score:
            idx, score = p
            print("topic-{}[{}]:\n {}".format(idx, score, topic_words[idx]))


    def get_global_topic_words_idx(self, model):
        '''
        Get global topic words index matrix
        param: model: CHNTM
        return: words_index_matrix: shape=(n_topic, self.args.topk_words)
        '''
        proba_mtx = utils.gmm_predict_proba_topic(
            self.vecs, 
            model.gmm_mu, 
            torch.exp(model.gmm_logvar), 
            model.gmm_weights)
        topk_proba_mtx, words_index_matrix = torch.topk(proba_mtx, k=self.args.topk_words, dim=1)   
        words_index_matrix = words_index_matrix.detach().cpu().numpy()
        return words_index_matrix      


    def get_local_topic_words_idx(self, model):
        '''
        Get local topic words index matrix
        param: model: VAE
        return: words_index_matrix: shape=(n_topic, self.args.topk_words)
        '''
        p_matrix_beta = model.decode(torch.eye(self.args.num_topics).to(self.device))
        _, words_index_matrix = torch.topk(p_matrix_beta, k=self.args.topk_words, dim=1)
        words_index_matrix = words_index_matrix.detach().cpu().numpy()
     
        return words_index_matrix

    
    def show_hierachical_topic_results(self):
        '''
        Display hierachical topic results
        '''
        global_topic_words_idx = self.get_global_topic_words_idx(self.model)
        local_topic_words_idx = self.get_local_topic_words_idx(self.model.vae)
        dependency_weights = self.model.dependency_weights
        proba_mtx, dependency_weights_topk = torch.topk(dependency_weights, k=10, dim=1)

        utils.print_log("hierachical topic results:")
        for i in range(len(dependency_weights_topk)):
            print("========= root topic-{} =========".format(i))
            print(utils.get_words_by_idx(self.dataset.dictionary, global_topic_words_idx[i]))
            for j, leaf_index in enumerate(dependency_weights_topk[i]):
                print("    #### leaf topic-{}({:.3f}): {}".format(
                    leaf_index,
                    proba_mtx[i][j].item(), 
                    utils.get_words_by_idx(self.dataset.dictionary, local_topic_words_idx[leaf_index])))



    def evaluate_pretrain(self, model, doc_mtx):
        '''
        model: vae model(gsm)
        '''
        beta = model.get_beta()
        coherence_score = utils.get_coherence(beta, doc_mtx)
        diversity_score = utils.get_diversity(beta)
        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score
        }

        return metric_dict


    def evaluate(self, doc_mtx):
        X = torch.Tensor(np.array(self.dataset.vecs)).to(self.device)
        
        X_proba_mtx = utils.gmm_predict_proba_X(
            X, 
            self.model.gmm_mu, 
            torch.exp(self.model.gmm_logvar), 
            self.model.gmm_weights)
        global_topic_proba_mtx = F.softmax(X_proba_mtx.T, dim=1).detach().cpu().numpy()
        global_topic_labels = np.argmax(X_proba_mtx.detach().cpu().numpy(), axis=1)
        local_topic_proba_mtx = self.model.vae.get_beta()
        dependency_mtx = self.model.dependency_weights.detach().cpu().numpy() # size:(n_topic_root,n_topic_leaf)
        # doc_mtx = self.dataset.doc_mtx

        # leaf topic npmi
        coherence_score = utils.get_coherence(local_topic_proba_mtx, doc_mtx)
        # leaf topic diversity
        diversity_score = utils.get_diversity(local_topic_proba_mtx)
        # clnpmi, or
        clnpmi_score, or_score = utils.get_CLNPMI_and_OR(global_topic_proba_mtx, local_topic_proba_mtx, dependency_mtx, doc_mtx)
        # topic specialization
        root_tc_score = utils.get_topic_specialization(global_topic_proba_mtx, doc_mtx)
        leaf_tc_score = utils.get_topic_specialization(local_topic_proba_mtx, doc_mtx)
        # clustering metric
        silhouette_score = utils.get_silhouette_coefficient(self.dataset.vecs, global_topic_labels)
        chi_score = utils.get_CHI(self.dataset.vecs, global_topic_labels)

        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score,
            "clnpmi": clnpmi_score,
            "overlap_rate": or_score,
            "global_topic_specialization": root_tc_score,
            "local_topic_specialization": leaf_tc_score,
            "silhouette_score": silhouette_score,
            "chi_score": chi_score,
        }  

        return metric_dict      