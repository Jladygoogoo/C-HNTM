from math import gamma
from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F

import utils



class NVDM_GSM(nn.Module):
    '''
    正态分布假设。
    '''
    def __init__(self, encode_dims, hidden_dim):
        super(NVDM_GSM, self).__init__()
        self.encode_dims = encode_dims
        self.hidden_dim = hidden_dim
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

        # gsm
        self.fc_gsm = nn.Linear(self.num_topics, self.num_topics)

        # decoder矩阵：beta = phi x psi
        self.phi = nn.Linear(hidden_dim, self.num_topics)
        self.psi = nn.Linear(hidden_dim, self.vocab_size)

    def reparameterize(self, mu, logvar):
        '''
        重参方法，基于encoder给出的期望和对数方差生成z。
        '''
        eps = torch.randn_like(mu) # 返回与mu大小一致且服从0-1正态分布的tensor
        std = torch.exp(logvar/2)
        z = mu + eps*std
        return z        

    def encode(self, x):
        hid = x
        for layer in self.encoder:
            hid = F.relu(layer(hid))
        mu, logvar = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, logvar

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
        logits = self.decode(theta)
        # print(logits)
        return logits, mu, logvar
    
    def loss(self, x, logits, mu, logvar):
        rec_loss = -1 * torch.sum(x * torch.log(logits))
        kld_loss = -0.5 * torch.sum(1 + 2*logvar - mu**2 - torch.exp(2 * logvar)) # 分布约束正则项
        # print("rec loss: {:.5f}, kld loss: {:.5f}".format(rec_loss, kld_loss))
        return rec_loss + kld_loss  

    def get_beta(self, return_tensor=False):
        weight = self.phi(self.psi.weight) # (vocab_size, num_topic)
        beta_weight = F.softmax(weight, dim=0).transpose(1, 0) # (num_topic, vocab_size)
        if not return_tensor:
            beta_weight = beta_weight.detach().cpu().numpy()
        return beta_weight





class C_HNTM(nn.Module):
    def __init__(self, n_topic_root, n_topic_leaf, encode_dims, embed_dim, hidden_dim, device):
        '''
        基于VaDE改造的层次神经网络。
        数据流：
        >> x:(batch_size, vocab_ size) 
        => encoder
        >> mu:(batch_size, n_topic_leaf), logvar:(batch_size, n_topic_leaf)
        => reparametize
        => decoder 
        >> z:(batch_size, n_topic_leaf) 
        >> reconst_x:(batch_size, vocab_size)
        '''
        super(C_HNTM, self).__init__()
        self.device = device

        # nn.Parameter 参与参数优化
        self.gmm_pi = nn.Parameter(torch.zeros(n_topic_root)) # gmm param
        self.gmm_mu = nn.Parameter(torch.randn(n_topic_root, embed_dim)) # gmm param
        self.gmm_logvar = nn.Parameter(torch.randn(n_topic_root, embed_dim)) # gmm param

        self.vae = NVDM_GSM(encode_dims, hidden_dim)

        self.dependency = nn.Linear(n_topic_root, n_topic_leaf)

    
    def init_gmm(self, gmm):
        '''
        由GMM模型初始化参数
        '''
        self.gmm_mu.data = torch.from_numpy(gmm.means_).float()
        self.gmm_logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()
        self.gmm_pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()

    
    def init_dependency_mtx(self, init_mtx):
        self.dependency.weight = nn.Parameter(init_mtx.T)


    @property
    def weights(self):
        return torch.softmax(self.gmm_pi, dim=0)         

    def forward(self, x):
        return self.vae(x)

    def loss(self, x, logits, mu, logvar, vecs):
        '''
        x: size=(batch_size, vocab_size)
        mu: size=(batch_size, n_topic_leaf)
        logvar: size=(batch_size, n_topic_leaf)
        vecs: size=(vocab_size, embed_dim)
        '''
        batch_size = x.shape[0]
        dependency = F.softmax(self.dependency.weight, dim=0).T # size=(n_topic_root, n_topic_leaf)
        n_topic_leaf = dependency.shape[1]

        # tau size=(batch_size, n_topic_root)
        # q(z_i|x)
        # https://stats.stackexchange.com/questions/321947/expectation-of-the-softmax-transform-for-gaussian-multivariate-variables
        tau = utils.get_softmax_guassian_multivirate_expectation(mu, logvar, self.device)

        # gamma
        # q(r_i|x)
        gamma = utils.predict_proba_gmm_doc(x, vecs, self.gmm_mu, torch.exp(self.gmm_logvar), self.weights)
        # print(value)
        # print(gamma)

        # (1)
        # p(x|z)
        l1_weight = 0.01
        l1 = torch.sum(x*torch.log(logits)) * l1_weight
        # (2)
        l2 = torch.sum(gamma * torch.mm(tau, torch.log(dependency.T)))
        # (3)
        l3 = torch.sum(torch.mm(gamma, torch.log(self.weights).unsqueeze(1))) # 注意此处使用self.weights而不是self.gmm_pi
        # (4)
        l4 = torch.sum(tau * torch.log(tau))
        # (5)
        l5 = torch.sum(gamma * torch.log(gamma))

        # dependency loss 用于避免dependency矩阵的趋同化
        cor_mtx = torch.matmul(dependency, dependency.T)
        norm = torch.norm(dependency, p=2, dim=1)
        norm_deno = torch.mm(norm.unsqueeze(dim=1), norm.unsqueeze(dim=0)) 
        norm_cor_mtx = cor_mtx / norm_deno
        dependency_loss = torch.sum(norm_cor_mtx)

        loss = -(l1 + l2 + l3 - l4 - l5)
        loss = -(l1 + l2 + l3 - l4 - l5) + dependency_loss

        loss_dict = {
            "l1": -l1,
            "l2": -l2,
            "l3": -l3,
            "l4": l4,
            "l5": l5,
            "dependency_loss": dependency_loss,
            "loss": loss
        }

        return loss_dict



if __name__ == "__main__":
    model = C_HNTM(10654, 20, 100, [2048, 1024, 512, 100], 300)
    print(model.beta.weight)
