import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

class BaseEncoder(nn.Module):
    def __init__(self, model_name_or_path="indolem/indobert-base-uncased"):
        super(BaseEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
    
    def forward(self, x, max_sentence_length=40):
        x = self.model(x)[0]
        return x