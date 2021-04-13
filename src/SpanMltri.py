import torch
from BaseEncoder import BaseEncoder
from TermScorer import TermScorer
from RelationScorer import RelationScorer
from SpanGenerator import SpanGenerator
from torch import nn

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class SpanMltri(nn.Module):
    def __init__(self, encoder_model_name_or_path="indolem/indobert-base-uncased",
                 d_hidden=768, max_sentence_length=40, num_of_class=3, max_span_length=8, k=0.4):
        super(SpanMltri, self).__init__()
        self.base_encoder = BaseEncoder(encoder_model_name_or_path) # input 
        self.span_generator = SpanGenerator(max_span_length)
        self.term_scorer = TermScorer(d_hidden, max_sentence_length, num_of_class)
        self.relation_scorer = RelationScorer(d_hidden, max_sentence_length, num_of_class, k)
        
    def forward(self, x):
        x = self.base_encoder(x)
        x, span_ranges = self.span_generator.generate_spans(x)
        
        logits_term_scorer = self.term_scorer(x)
        logits_relation_scorer, span_pair_ranges = self.relation_scorer(x, span_ranges)
        
        return logits_term_scorer, span_ranges, logits_relation_scorer, span_pair_ranges
    
    def show_model_structure(self):
        print("Model structure: ", self, "\n\n")

        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
