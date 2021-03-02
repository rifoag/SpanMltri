import torch
from transformers import AutoTokenizer, AutoModel

class BaseEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        self.model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
    
    def encode(self, sentence):
        x = torch.LongTensor(self.tokenizer.encode(sentence)).view(1,-1)
        return(self.model(x)[0])