import torch
from BaseEncoder import BaseEncoder
from TermScorer import TermScorer
from SpanGenerator import SpanGenerator
from torch import nn

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class SpanMltri(nn.Module):
    def __init__(self, encoder_model_name_or_path="indolem/indobert-base-uncased",
                 d_hidden=768, max_sentence_length=40, num_of_class=3, max_span_length=8):
        super(SpanMltri, self).__init__()
        self.base_encoder = BaseEncoder(encoder_model_name_or_path)
        self.span_generator = SpanGenerator(max_span_length)
        self.term_scorer = TermScorer(max_span_length*d_hidden, max_sentence_length, num_of_class)
        
    def forward(self, x):
        x = self.base_encoder(x)
        x, span_ranges = self.span_generator.generate_spans(x)
        x = torch.flatten(x, start_dim=2)
        logits_term_scorer = self.term_scorer(x)
        return logits_term_scorer, span_ranges
