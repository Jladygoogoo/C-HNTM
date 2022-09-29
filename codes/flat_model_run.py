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
from my_dataset import MyDataset


class Runner:
    def __init__(self, args):
        '''
        载入参数和数据集
        '''
        self.args = args
        self.device = utils.get_device(args.device) # device=-1表示cpu，其他表示gpu序号
        utils.print_log("device: {}".format(self.device))
        self.model_save_path = "../models/{}/{}_{}_{}.pt".format(
                args.model,
                args.model,
                args.data,
                time.strftime("%y%m%d%H%M", time.localtime())
            )
        utils.print_log("Model will be saved in {}".format(self.model_save_path))
        self.checkpoint_dir = "../models/{}/.ckpt/".format(args.model)
        utils.get_or_create_path(self.model_save_path)
        utils.get_or_create_path(self.checkpoint_dir)
        
        # load data
        self.data_source_name = args.data
        self.dataset = MyDataset(self.data_source_name)
        self.vecs = torch.tensor(np.array(self.dataset.vecs)).to(self.device)
        self.train_dataloader = DataLoader(self.dataset.train_load_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.dataset.test_load_dataset, batch_size=args.batch_size, shuffle=True)

        utils.print_log("Loading data from dataset-[{}]".format(self.data_source_name))
        utils.print_log("dictionary size: {} | train size: {} | test size: {}".format(
            self.dataset.vocab_size, len(self.dataset.train_load_dataset), len(self.dataset.test_load_dataset)
        ))

        # model args
        self.args = args
        self.vocab_size = self.dataset.vocab_size
        self.num_topics = args.num_topics        


    def train(self):
        pass
    

    def evaluate(self, model, doc_mtx):
        '''
        model: vae model(gsm)
        '''
        beta = self.get_beta(model)
        coherence_score = utils.get_coherence(beta, doc_mtx)
        diversity_score = utils.get_diversity(beta)
        metric_dict = {
            "topic_coherence": coherence_score,
            "topic_diversity": diversity_score
        }

        return metric_dict


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
        hidden_dim = 256
        self.model = model.NVDM_GSM(
            encode_dims=[self.vocab_size, 1024, 512, self.num_topics],
            hidden_dim=hidden_dim
        )
        if self.args.resume_train_path is not None:
            self.model.load_state_dict(torch.load(self.args.resume_train_path))
            utils.print_log("Resume Training from: {}".format(self.args.resume_train_path))
        self.model.to(self.device)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        for epoch in range(self.args.num_epochs):
            epoch_loss = []
            for data in self.train_dataloader:
                optimizer.zero_grad()
                bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            utils.print_log("Epoch-[{}]".format(epoch))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            utils.print_log("Loss: {:.6f}".format(avg_epoch_loss))

            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model, self.dataset.train_doc_mtx)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))

            # test
            if (epoch+1) % self.args.test_interval == 0:
                utils.print_log("======= Test =======")
                with torch.no_grad():
                    loss_list = []
                    for batch_data in self.test_dataloader:
                        x = batch_data
                        x = x.to(self.device)
                        d_given_theta, mu, logvar = self.model(x)
                        loss = self.model.loss(x, d_given_theta, mu, logvar)
                        loss_list.append(loss.item()/len(x))
                    avg_loss = np.mean(loss_list)
                utils.wandb_log("test/loss", {"loss": avg_loss}, self.args.wandb)
                utils.print_log("Loss: {:.6f}".format(avg_loss))                
                # test data
                metric_dict = self.evaluate(self.model, self.dataset.test_doc_mtx)
                utils.wandb_log("test/metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))   
                utils.print_log("====================")    


            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                utils.print_log("Topic results:")
                for i, idxs in enumerate(self.get_local_topic_words_idx(self.model)):
                    print("topic-{}: {}".format(i, utils.get_words_by_idx(self.dataset.dictionary, idxs)))

            # checkpoint
            if (epoch+1) % self.args.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, "{}_{}_{}_{}.ckpt".format(
                    self.args.model, self.args.data, time.strftime("%y%m%d%H%M", time.localtime()), epoch+1))
                torch.save(self.model.state_dict(), ckpt_path)
                utils.print_log("Checkpoint saved: {}".format(ckpt_path))                    
 

        torch.save(self.model.state_dict(), self.model_save_path)
        utils.print_log("{} saved: {}".format(self.args.model, self.model_save_path))



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
            for data in self.train_dataloader:
                optimizer.zero_grad()
                bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            utils.print_log("Epoch-[{}]".format(epoch))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            utils.print_log("Loss: {:.6f}".format(avg_epoch_loss))

            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model, self.dataset.train_doc_mtx)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))

            # test
            if (epoch+1) % self.args.test_interval == 0:
                utils.print_log("======= Test =======")
                with torch.no_grad():
                    loss_list = []
                    for batch_data in self.test_dataloader:
                        x = batch_data
                        x = x.to(self.device)
                        d_given_theta, mu, logvar = self.model(x)
                        loss = self.model.loss(x, d_given_theta, mu, logvar)
                        loss_list.append(loss.item()/len(x))
                    avg_loss = np.mean(loss_list)
                utils.wandb_log("test/loss", {"loss": avg_loss}, self.args.wandb)
                utils.print_log("Loss: {:.6f}".format(avg_loss))                
                # test data
                metric_dict = self.evaluate(self.model, self.dataset.test_doc_mtx)
                utils.wandb_log("test/metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))   
                utils.print_log("====================")    

            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                utils.print_log("Topic results:")
                for i, idxs in enumerate(self.get_local_topic_words_idx(self.model)):
                    print("topic-{}: {}".format(i, utils.get_words_by_idx(self.dataset.dictionary, idxs)))

            # checkpoint
            if (epoch+1) % self.args.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, "{}_{}_{}_{}.ckpt".format(
                    self.args.model, self.args.data, time.strftime("%y%m%d%H%M", time.localtime()), epoch+1))
                torch.save(self.model.state_dict(), ckpt_path)
                utils.print_log("Checkpoint saved: {}".format(ckpt_path))
 

        torch.save(self.model.state_dict(), self.model_save_path)
        utils.print_log("{} saved: {}".format(self.args.model, self.model_save_path))



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
        if self.args.resume_train_path is not None:
            self.model.load_state_dict(torch.load(self.args.resume_train_path))
            utils.print_log("Resume Training from: {}".format(self.args.resume_train_path))

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
            for data in self.train_dataloader:
                optimizer.zero_grad()
                bows = data
                x = bows
                x = x.to(self.device)
                prob, mu, logvar = self.model(x)
                loss = self.model.loss(x, prob, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            utils.print_log("Epoch-[{}]".format(epoch))
            # loss
            avg_epoch_loss =sum(epoch_loss)/len(epoch_loss) 
            utils.wandb_log("loss", {"loss": avg_epoch_loss}, self.args.wandb)
            utils.print_log("Loss: {:.6f}".format(avg_epoch_loss))

            # metric
            if (epoch+1) % self.args.metric_log_interval == 0:
                metric_dict = self.evaluate(self.model, self.dataset.train_doc_mtx)
                utils.wandb_log("metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))


            # test
            if (epoch+1) % self.args.test_interval == 0:
                utils.print_log("======= Test =======")
                with torch.no_grad():
                    loss_list = []
                    for batch_data in self.test_dataloader:
                        x = batch_data
                        x = x.to(self.device)
                        d_given_theta, mu, logvar = self.model(x)
                        loss = self.model.loss(x, d_given_theta, mu, logvar)
                        loss_list.append(loss.item()/len(x))
                    avg_loss = np.mean(loss_list)
                utils.wandb_log("test/loss", {"loss": avg_loss}, self.args.wandb)
                utils.print_log("Loss: {:.6f}".format(avg_loss))                
                # test data
                metric_dict = self.evaluate(self.model, self.dataset.test_doc_mtx)
                utils.wandb_log("test/metric", metric_dict, self.args.wandb)
                utils.print_log("Coherence(NPMI): {:.6f} | Diversity: {:.6f}".format(
                    metric_dict["topic_coherence"], 
                    metric_dict["topic_diversity"]))   
                utils.print_log("====================")     


            # topic words
            if (epoch+1) % self.args.topic_log_interval == 0:
                utils.print_log("Topic results:")
                for i, idxs in enumerate(self.get_local_topic_words_idx(self.model)):
                    print("topic-{}: {}".format(i, utils.get_words_by_idx(self.dataset.dictionary, idxs)))

            # checkpoint
            if (epoch+1) % self.args.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, "{}_{}_{}_{}.ckpt".format(
                    self.args.model, self.args.data, time.strftime("%y%m%d%H%M", time.localtime()), epoch+1))
                torch.save(self.model.state_dict(), ckpt_path)
                utils.print_log("Checkpoint saved: {}".format(ckpt_path))                    


        torch.save(self.model.state_dict(), self.model_save_path)
        utils.print_log("{} saved: {}".format(self.args.model, self.model_save_path))      


