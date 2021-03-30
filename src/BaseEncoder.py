import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

class BaseEncoder(nn.Module):
    def __init__(self, model_name_or_path="indolem/indobert-base-uncased"):
        super(BaseEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
    
    def tokenize(self, sentence, max_sentence_length=40):
        x = torch.LongTensor(self.tokenizer.encode(sentence, padding='max_length', max_length=max_sentence_length)).view(1,-1)       
        x = x[0, :max_sentence_length]
        return x
    
    def forward(self, x, max_sentence_length=40):
        x = self.model(x)[0]
        return x