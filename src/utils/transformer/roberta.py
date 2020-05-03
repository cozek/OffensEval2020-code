import torch
import torch.nn as nn


class RobertaAttention(nn.Module):
    """Implements Attention Head Classifier
    on Pretrained Roberta Transformer representations.
    Attention Head Implementation based on: https://www.aclweb.org/anthology/P16-2034/
    """
    def penalized_tanh(self,x):
        """
        http://aclweb.org/anthology/D18-1472
        """
        alpha = 0.25
        return torch.max(torch.tanh(x), alpha*torch.tanh(x))
    def swish(self, x):
        """
        Simple implementation of Swish activation function
        https://arxiv.org/pdf/1710.05941.pdf
        """
        return x * torch.sigmoid(x)
    
    def mish(self, x):
        """
        Simple implementation of Mish activation Function
        https://arxiv.org/abs/1908.08681
        """
        tanh = nn.Tanh()
        softplus = nn.Softplus()
        return x * tanh( softplus(x))
    
    def __init__(self, model_name, num_labels):
        """
        Args:
            model_name: model name, eg, roberta-base'
        """
        super().__init__()
        self.w = nn.Linear(768,1, bias=False)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.prediction_layer = nn.Linear(768, num_labels)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.prediction_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.w.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def forward(self, input_ids,attention_mask):
        """
        Args:
            input_ids: sent encoded into indices
            attention_mask: their respective attention masks,
        """
        #elmo layer takes care of padding
        embeddings = self.roberta(input_ids = input_ids,
                  attention_mask = attention_mask)
        H = embeddings[0] #final hidden layer outputs 
#         print(H.shape)
        M = self.penalized_tanh(H)
        alpha = torch.softmax(self.w(M), dim=1)
        r = torch.bmm(H.permute(0,2,1),alpha)
        h_star = self.penalized_tanh(r)
        preds = self.prediction_layer(h_star.permute(0,2,1))
        return preds
    
class RobertaAttentionReg(nn.Module):
    """Implements Attention Head Classifier
    on Pretrained Roberta Transformer representations.
    Attention Head Implementation based on: https://www.aclweb.org/anthology/P16-2034/
    """
    def swish(self, x):
        """
        Simple implementation of Swish activation function
        https://arxiv.org/pdf/1710.05941.pdf
        """
        return x * torch.sigmoid(x)
    
    def mish(self, x):
        """
        Simple implementation of Mish activation Function
        https://arxiv.org/abs/1908.08681
        """
        tanh = nn.Tanh()
        softplus = nn.Softplus()
        return x * tanh( softplus(x))
    
    def __init__(self, model_name, num_labels):
        """
        Args:
            model_name: model name, eg, roberta-base'
        """
        super().__init__()
        self.w = nn.Linear(768,1, bias=False)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.prediction_layer = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.prediction_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)
        for name, param in self.w.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)
        
    def forward(self, input_ids,attention_mask):
        """
        Args:
            input_ids: sent encoded into indices
            attention_mask: their respective attention masks,
        """
        #elmo layer takes care of padding
        embeddings = self.roberta(input_ids = input_ids,
                  attention_mask = attention_mask)
        
        H = embeddings[0] #final hidden layer outputs 
#         print(H.shape)
        M = self.mish(H)
        alpha = torch.softmax(self.w(M), dim=1)
        alpha = self.dropout(alpha)
        
        r = torch.bmm(H.permute(0,2,1),alpha)        
        
        h_star = self.mish(r)
        h_star = self.dropout(h_star)
        
        preds = self.prediction_layer(h_star.permute(0,2,1))
        return preds

class RobertaAttentionNorm(nn.Module):
    """Implements Attention Head Classifier
    on Pretrained Roberta Transformer representations.
    Attention Head Implementation based on: https://www.aclweb.org/anthology/P16-2034/
    """
    def swish(self, x):
        """
        Simple implementation of Swish activation function
        https://arxiv.org/pdf/1710.05941.pdf
        """
        return x * torch.sigmoid(x)
    
    def mish(self, x):
        """
        Simple implementation of Mish activation Function
        https://arxiv.org/abs/1908.08681
        """
        tanh = nn.Tanh()
        softplus = nn.Softplus()
        return x * tanh( softplus(x))
    
    def __init__(self, model_name, num_labels, max_seq_len):
        """
        Args:
            model_name: model name, eg, roberta-base'
        """
        super().__init__()
        self.w = nn.Linear(768,1, bias=False)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.prediction_layer = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm = nn.BatchNorm1d(max_seq_len)
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.prediction_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)
        for name, param in self.w.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)
        
    def forward(self, input_ids,attention_mask):
        """
        Args:
            input_ids: sent encoded into indices
            attention_mask: their respective attention masks,
        """
        #elmo layer takes care of padding
        embeddings = self.roberta(input_ids = input_ids,
                  attention_mask = attention_mask)
        
        H = embeddings[0] #final hidden layer outputs 

        H = self.batchnorm(H)

        M = self.swish(H)
        alpha = torch.softmax(self.w(M), dim=1)
        alpha = self.dropout(alpha)
        
        r = torch.bmm(H.permute(0,2,1),alpha)        
        
        h_star = self.swish(r)
        h_star = self.dropout(h_star)
        
        preds = self.prediction_layer(h_star.permute(0,2,1))

        return preds