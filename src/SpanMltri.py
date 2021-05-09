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
                 d_hidden=768, max_sentence_length=40, num_of_class=4, max_span_length=8, k=0.4):
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
    
    def decode_te_label(self, te_label_dict, tokens):
        # input : 
        #     te_label_dict = Dict; Key : span range (string), Value :  term type label (ASPECT, SENTIMENT, O)
        #     tokens = list of tokens from the sentence
        # output : List of pair (phrase, term_type_label)
        te_label_list = []

        for span_range in te_label_dict:
            start_idx, end_idx = span_range.split('-')
            start_idx, end_idx = int(start_idx), int(end_idx)
            sentence = ' '.join(tokens[start_idx:end_idx+1])
            te_label = te_label_dict[span_range]
            te_label_list.append((sentence, te_label))

        return te_label_list

    def decode_relation_label(self, relation_dict, tokens):
        # input : 
        #     relation = Dict; Key : span range pair (aspect_term_span_range, opinion_term_span_range), Value : sentiment polarity label (PO, NG, NT)
        #     tokens = list of tokens from the sentence
        # output : list of triples (aspect_term, opinion_term, polarity)
        relation_list = []

        for span_range_pair in relation_dict:
            aspect_term_span_range, opinion_term_span_range = span_range_pair

            aspect_term_start_idx, aspect_term_end_idx = aspect_term_span_range.split('-')
            aspect_term_start_idx, aspect_term_end_idx = int(aspect_term_start_idx), int(aspect_term_end_idx)
            aspect_term = ' '.join(tokens[aspect_term_start_idx:aspect_term_end_idx+1])

            opinion_term_start_idx, opinion_term_end_idx = opinion_term_span_range.split('-')
            opinion_term_start_idx, opinion_term_end_idx = int(opinion_term_start_idx), int(opinion_term_end_idx)
            opinion_term = ' '.join(tokens[opinion_term_start_idx:opinion_term_end_idx+1])

            relation_label = relation_dict[span_range_pair]

            relation_list.append((aspect_term, opinion_term, relation_label))

        return relation_list
