import torch
from TermScorer import TermScorer
from RelationScorer import RelationScorer
from torch import nn

# Define model
class SpanMltriLite(nn.Module):
    def __init__(self, d_hidden=768, max_sentence_length=40, num_of_te_class=3, num_of_relation_class=4, k=0.4):
        super(SpanMltriLite, self).__init__()
        self.term_scorer = TermScorer(d_hidden, max_sentence_length, num_of_te_class)
#         self.relation_scorer = RelationScorer(d_hidden=768, max_sentence_length=40, num_of_class=4, k=0.4)
        
    def forward(self, x, span_ranges):
        logits_term_scorer = self.term_scorer(x)
#         logits_relation_scorer, span_pair_ranges = self.relation_scorer(x, span_ranges)
#         return logits_term_scorer, logits_relation_scorer, span_pair_ranges
        return logits_term_scorer
    
    def show_model_structure(self):
        print("Model structure: ", self, "\n\n")

        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    