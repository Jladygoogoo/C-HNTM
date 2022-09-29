import torch
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel



class PretrainBERT:
    def __init__(self, model_path=None, n_layers=4, mode="mean") -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.layers = list(range(-n_layers, 0))
        self.mode = mode
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

    def get_embedding(self, word, return_tensor=False):
        '''
        获取BERT词向量
        param: layers: 指定参与计算词向量的隐藏层，-1表示最后一层
        param: mode: 隐藏层合并策略
        return: torch.Tensor, size=[768]
        '''
        output = self.model(**self.tokenizer(word, return_tensors='pt'))
        specified_hidden_states = [output.hidden_states[i] for i in self.layers]
        specified_embeddings = torch.stack(specified_hidden_states, dim=0)
        # layers to one strategy
        if self.mode == "sum":
            token_embeddings = torch.squeeze(torch.sum(specified_embeddings, dim=0))
        elif self.mode == "mean":
            token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))        
        # tokens to one strategy
        word_embedding = torch.mean(token_embeddings, dim=0)
        if not return_tensor:
            word_embedding = word_embedding.detach().numpy()
        return word_embedding
    
    def has_word(self, word):
        return True



class PretrainWord2Vec:
    def __init__(self):
        self.model = Word2Vec.load("/Users/inkding/程序/my-projects/毕设-网易云评论多模态/netease2/models/w2v/c4.mod")
    
    def has_word(self, word):
        return self.model.wv.__contains__(word)

    def get_embedding(self, word):
        return self.model.wv[word]


if __name__ == "__main__":
    model_path = "../models/bert/wiki103_bert.pt"
    bert = PretrainBERT(model_path=model_path)