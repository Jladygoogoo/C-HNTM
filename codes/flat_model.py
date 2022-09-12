from re import L
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from sklearn.mixture import GaussianMixture

'''
本文件包括所有NTM模型。分为VAE和WAE两大类：
    + VAE 类：VAE, NVDM_GSM, AVITM, ETM
    + WAE 类：WAE
拥有类方法：
    + encode
    + decode
    + reparameterize
    + loss
    + forward
'''

# VAE
class VAE(nn.Module):
    def __init__(self, encode_dims):
        '''
        提供统一的encode和reparameterize函数
        '''        
        super(VAE, self).__init__()
        self.encode_dims = encode_dims
        self.vocab_size = encode_dims[0]
        self.num_topics = encode_dims[-1]

        # encoder
        self.encoder = nn.ModuleList([
            nn.Linear(encode_dims[i], encode_dims[i+1]) for i in range(len(encode_dims)-2)
        ])

        # 期望和对数方差
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])
        nn.init.constant_(self.fc_logvar.weight, 0.0)     
        nn.init.constant_(self.fc_logvar.bias, 0.0)     

        
    def encode(self, x):
        hid = x
        for layer in self.encoder:
            hid = F.relu(layer(hid))
        mu, logvar = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, logvar        

    
    def reparameterize(self, mu, logvar):
        '''
        重参方法，基于encoder给出的期望和对数方差生成z。
        '''
        eps = torch.randn_like(mu) # 返回与mu大小一致且服从0-1正态分布的tensor
        std = torch.exp(logvar/2)
        z = mu + eps*std
        return z    



class NVDM_GSM(VAE):
    '''
    正态分布假设。
    '''
    def __init__(self, encode_dims, hidden_dim):
        super(NVDM_GSM, self).__init__(encode_dims)

        # gsm
        self.fc_gsm = nn.Linear(self.num_topics, self.num_topics)
        # decoder矩阵：beta = phi x psi
        self.phi = nn.Linear(hidden_dim, self.num_topics)
        self.psi = nn.Linear(hidden_dim, self.vocab_size)


    def decode(self, theta):
        # decode得到的已经是0-1的值，之后无需再进行softmax
        weight = self.phi(self.psi.weight) # (vocab_size, num_topic)
        beta_weight = F.softmax(weight, dim=0).transpose(1, 0) # (num_topic, vocab_size)
        logits = torch.matmul(theta, beta_weight)
        # almost_zeros = torch.full_like(logits, 1e-6)
        # logits = logits.add(almost_zeros)
        return logits


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(self.fc_gsm(z), dim=1) # gsm
        prob = self.decode(theta)
        # print(prob)
        return prob, mu, logvar
    

    def loss(self, x, prob, mu, logvar):
        rec_loss = -1 * torch.sum(x * torch.log(prob))
        kld_loss = -0.5 * torch.sum(1 + 2*logvar - mu**2 - torch.exp(2 * logvar)) # 分布约束正则项
        # print("rec loss: {:.5f}, kld loss: {:.5f}".format(rec_loss, kld_loss))
        return rec_loss + kld_loss  


    # def get_beta(self, return_tensor=False):
    #     weight = self.phi(self.psi.weight) # (vocab_size, num_topic)
    #     beta_weight = F.softmax(weight, dim=0).transpose(1, 0) # (num_topic, vocab_size)
    #     if not return_tensor:
    #         beta_weight = beta_weight.detach().cpu().numpy()
    #     return beta_weight


class AVITM(VAE):
    '''
    狄利克雷分布假设
    '''
    def __init__(self, encode_dims, decode_dims):
        super(AVITM, self).__init__(encode_dims)

        # 设置先验分布参数
        self.prior_mean, self.prior_var = map(nn.Parameter, self.prior(self.num_topics))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

        # decoder
        self.decoder = nn.ModuleList([
            nn.Linear(decode_dims[i], decode_dims[i+1]) for i in range(len(decode_dims)-1)
        ])        


    def prior(self, K, alpha=0.3):
        '''
        Prior for the model.
        :K: number of categories
        :alpha: Hyper param of Dir
        :return: mean and variance tensors
        '''
        # Approximate to normal distribution using Laplace approximation
        a = torch.Tensor(1, K).float().fill_(alpha)
        mean = a.log().t() - a.log().mean(1)
        var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
        return mean.t(), var.t() # Parameters of prior distribution after approximation
    

    def decode(self, z):
        # decode得到的已经是0-1的值，之后无需再进行softmax
        hid = z
        for i, layer in enumerate(self.decoder):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(hid)
        prob = F.softmax(hid, dim=1)
        return prob    


    def forward(self, x):
        mu, logvar = self.encode(x)
        z_ = self.reparameterize(mu, logvar)
        z = F.softmax(z_, dim=0) # avitm在中间使用了softmax
        prob = self.decode(z)
        return prob, mu, logvar   


    def loss(self, x, prob, posterior_mean, posterior_logvar):
        logsoftmax= torch.log(prob)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失

        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_logvar)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld_loss = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        return (rec_loss + kld_loss).mean()




class ETM(VAE):
    '''
    正态分布假设
    '''
    def __init__(self, encode_dims, embed_dim, rho_init=None):
        super().__init__(encode_dims)
        self.alpha = nn.Linear(embed_dim, self.num_topics)
        self.rho = nn.Linear(embed_dim, self.vocab_size)
        if rho_init is not None:
            self.rho.weight = nn.Parameter(rho_init)


    def decode(self, theta):
        # decode得到的已经是0-1的值，之后无需再进行softmax
        beta_weight = self.alpha(self.rho.weight) # (vocab_size, num_topic)
        beta_weight = F.softmax(beta_weight, dim=0).transpose(1, 0) # (num_topic, vocab_size)，本质上是decoder
        prob = torch.mm(theta, beta_weight)
        return prob


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=0) # avitm在中间使用了softmax
        prob = self.decode(theta)
        return prob, mu, logvar   


    def loss(self, x, prob, mu, logvar):
        # 和NVDM_GSM是一样的
        rec_loss = -1 * torch.sum(x * torch.log(prob))
        kld_loss = -0.5 * torch.sum(1 + 2*logvar - mu**2 - torch.exp(2 * logvar)) # 分布约束正则项        
        return rec_loss + kld_loss    
            
        
