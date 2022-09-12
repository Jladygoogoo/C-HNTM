import os
import time
import wandb
from tqdm import tqdm
import itertools
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.optim import SGD
import utils
import flat_model as model
from dataset.news20_dataset import NewsDataset
from dataset.wiki103_dataset import Wikitext103Dataset
from dataset.reuters_dataset import ReutersDataset


class Runner:
    def __init__(self, args):
        '''
        载入参数和数据集
        '''
        self.device = utils.get_device(args.device) # device=-1表示cpu，其他表示gpu序号
        print("使用device:", self.device)
        self.save_path = "../models/{}/{}_{}.pkl".format(
            args.model, args.model,
            args.data,
            # time.strftime("%Y-%m-%d-%H", time.localtime())
            )
        utils.get_or_create_path(self.save_path)
        
        # 加载数据
        self.data_source_name = args.data
        if args.data == "20news":
            self.dataset = NewsDataset()
        if args.data == "wiki103":
            self.dataset = Wikitext103Dataset()
        if args.data == "reuters":
            self.dataset = ReutersDataset()

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.vecs = torch.tensor(np.array(self.dataset.vecs)).to(self.device)
        print("数据集: {}".format(self.data_source_name))
        print("数据集大小: {}, 词典大小: {}".format(len(self.dataset), self.dataset.vocab_size))

        # 加载常用参数
        self.args = args
        self.vocab_size = self.dataset.vocab_size
        self.num_topics = args.num_topics        


    def train(self):
        pass
    
    def evaluate(self, model):
        topic_words = self.get_topic_words(model)
        coherence_score = utils.get_coherence(self.get_beta(model), self.dataset.doc_mtx)
        diversity_score = utils.calc_topic_diversity(topic_words)
        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score
        }

        return metric_dict


    def get_topic_words(self, model):
        '''
        获取底层 topic words
        '''
        # decode得到的已经是0-1的值，之后无需再进行softmax
        beta_weight = self.get_beta(model, return_tensor=True)
        _, words_index_matrix = torch.topk(beta_weight, k=10, dim=1)
        topic_words = []
        for words_index in words_index_matrix:
            topic_words.append([self.dataset.dictionary.id2token[i.item()] for i in words_index])        
        return topic_words

    
    def get_beta(self, model, return_tensor=False):
        idxes = torch.eye(self.num_topics).to(self.device)
        beta_weight = model.decode(idxes)
        if not return_tensor:
            beta_weight = beta_weight.detach().cpu().numpy()
        return beta_weight        



class NVDM_GSM_Runner(Runner):
    def __init__(self, args):
        super(NVDM_GSM_Runner, self).__init__(args)
        hidden_dim = 256 # 和CHNTM的设置一致
        self.model = model.NVDM_GSM(
            encode_dims=[self.vocab_size, 1024, 512, self.num_topics],
            hidden_dim=hidden_dim
        )
        self.model.to(self.device)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        for epoch in range(self.args.num_epochs):
            epoch_loss = []
            for data in self.dataloader:
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            print("Epoch {} AVG Loss: {:.6f}".format(
                epoch+1, 
                avg_epoch_loss))
            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                print("Epoch {} AVG Coherence(NPMI): {:.6f} AVG Diversity: {:.6f}".format(
                    epoch+1, 
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))
            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                for i, words in enumerate(self.get_topic_words(self.model)):
                    print("topic-{}: {}".format(i, words))

        torch.save(self.model.state_dict(), self.save_path)
        print("将model参数保存至", self.save_path)




class AVITM_Runner(Runner):
    def __init__(self, args):
        super().__init__(args)
        self.model = model.AVITM(
            encode_dims=[self.vocab_size, 512, self.num_topics],
            decode_dims=[self.num_topics, 512, self.vocab_size],
        )
        self.model.to(self.device)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        for epoch in range(self.args.num_epochs):
            epoch_loss = []
            for data in self.dataloader:
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            print("Epoch {} AVG Loss: {:.6f}".format(
                epoch+1, 
                avg_epoch_loss))
            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                print("Epoch {} AVG Coherence(NPMI): {:.6f} AVG Diversity: {:.6f}".format(
                    epoch+1, 
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))
            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                for i, words in enumerate(self.get_topic_words(self.model)):
                    print("topic-{}: {}".format(i, words))

        torch.save(self.model.state_dict(), self.save_path)
        print("将model参数保存至", self.save_path)



class ETM_Runner(Runner):
    def __init__(self, args):
        super().__init__(args)
        
        # 初始化beta
        rho_init = torch.Tensor(self.dataset.vecs) # (vocab_size, embed_dim)

        self.model = model.ETM(
            encode_dims=[self.vocab_size, 1024, 512, self.num_topics],
            embed_dim=rho_init.shape[1],
            rho_init=rho_init,
        )
        self.model.to(self.device)

        # 减小词向量rho矩阵的学习速度
        rho_param_ids = list(map(id, self.model.rho.parameters()))
        base_params = filter(lambda p:id(p) not in rho_param_ids, self.model.parameters())
        self.params = [
            {"params": self.model.rho.parameters(), "lr": 1e-5}, 
            {"params": base_params, "lr":self.args.lr}
        ]

    def train(self):
        self.model.train()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        optimizer = torch.optim.Adam(self.params, lr=self.args.lr)
        for epoch in range(self.args.num_epochs):
            epoch_loss = []
            for data in self.dataloader:
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            print("Epoch {} AVG Loss: {:.6f}".format(
                epoch+1, 
                avg_epoch_loss))
            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                print("Epoch {} AVG Coherence(NPMI): {:.6f} AVG Diversity: {:.6f}".format(
                    epoch+1, 
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))
            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                for i, words in enumerate(self.get_topic_words(self.model)):
                    print("topic-{}: {}".format(i, words))

        torch.save(self.model.state_dict(), self.save_path)
        print("将model参数保存至", self.save_path)           


