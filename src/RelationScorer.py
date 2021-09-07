import torch
from torch import nn

# Define model
class RelationScorer(nn.Module):
    def __init__(self, d_hidden=768, max_sentence_length=40, num_of_class=4, k=0.4):
        super(RelationScorer, self).__init__()
        self.k = k
        self.dropout = nn.Dropout(0.10)
        self.span_scorer = nn.Sequential(
            nn.Linear(d_hidden, 768),
            nn.ReLU(),
            nn.Linear(768, num_of_class) # input x (batch_size, num_of_spans, d_hidden), output h (batch_size, num_of_spans, num_of_class)
        )
        
        self.pair_scorer = nn.Sequential(
            nn.Linear(3*d_hidden, 768),
            nn.ReLU(),
            nn.Linear(768, num_of_class) # input dim (batch_size, num_of_span_pairs, 3*d_hidden)
        )
            
    def get_top_k_percent_span_indexes(self, h, k=0.4):
        top_k_indexes = [] # (batch_size, k*num_of_spans)
        batch_size, num_of_spans = h.shape
         
        for i in range(batch_size):
            top_k_indexes_ = []
            for j in range(num_of_spans):
                if h[i,j].item() < k*num_of_spans:
                    top_k_indexes_.append(j)
            top_k_indexes.append(top_k_indexes_)
        
        return top_k_indexes
    
    def get_top_k_percent_spans(self, x, h, span_ranges, k=0.4):
        # input : x (batch_size, num_of_spans, d_hidden), h (batch_size, num_of_spans, 1)
        # output : x_ranked (batch_size, k*num_of_span, d_hidden), span_ranges_ranked (batch_size, k*num_of_span)
        batch_size, num_of_spans = x.shape[0], x.shape[1]
        
        h = h[:, :, [1, 2, 3]]
        h = h.mean(dim=-1) # output dim (batch_size, num_of_spans)
        
        h = torch.argsort(h, dim=-1, descending=True)
        top_k_indexes = self.get_top_k_percent_span_indexes(h, k)

        x_ranked = []
        for i in range(batch_size):
            x_ranked.append(x[i, top_k_indexes[i], :])
        x_ranked = torch.stack(x_ranked)
        
        span_ranges_ranked = []
        for i in range(batch_size):
            span_ranges_ranked_ = [span_ranges[j] for j in top_k_indexes[i]]
            span_ranges_ranked.append(span_ranges_ranked_)
        
        return x_ranked, span_ranges_ranked
        
    def get_pair_representations(self, x_ranked, span_ranges_ranked):
        # input : x_ranked (batch_size, num_of_span, d_hidden)
        # output : x_paired (batch_size, num_of_span_pairs, 3**d_hidden), span_pair_ranges (batch_size, num_of_span_pairs)
        batch_size, num_of_span, _ = x_ranked.shape
        
        x_paired = []
        for batch in range(batch_size):
            x_paired_ = []
            for i in range(num_of_span):
                for j in range(num_of_span):
                    f_mult = x_ranked[batch, i, :] * x_ranked[batch, j, :]
                    x_paired_.append(torch.cat([x_ranked[batch, i, :], x_ranked[batch, j, :], f_mult]))
            x_paired.append(torch.stack(x_paired_))
        x_paired = torch.stack(x_paired)
        
        span_pair_ranges = []
        for batch in range(batch_size):
            span_pair_ranges_ = []
            for i in range(num_of_span):
                for j in range(num_of_span):
                    f_i_range = span_ranges_ranked[batch][i]
                    f_j_range = span_ranges_ranked[batch][j]
                    span_pair_ranges_.append((f_i_range, f_j_range))
            span_pair_ranges.append(span_pair_ranges_)

        return x_paired, span_pair_ranges
    
    def get_sum_of_span_scores_and_pair_score(self, logits_span_scorer, logits_pair_scorer, span_ranges, span_pair_ranges):
        # input : logits_span_scorer (batch_size, num_of_spans, num_of_class), logits_relation_scorer (batch_size, num_of_span_pairs, num_of_class)
        # output : summed_logits(batch_size, num_of_span_pairs, num_of_class)
        num_of_spans = logits_span_scorer.shape[1]
        batch_size = logits_span_scorer.shape[0]
        
        logits_span_scorer_1 = logits_span_scorer.repeat(1, num_of_spans, 1)
        logits_span_scorer_2 = torch.repeat_interleave(logits_span_scorer, num_of_spans, dim=1)
        summed_logits = logits_pair_scorer + logits_span_scorer_1 + logits_span_scorer_2
        return summed_logits
        
    def forward(self, x, span_ranges):
        # input : x (batch_size, num_of_spans, max_span_length*d_hidden)
        # output : y (batch_size, num_of_span_pairs, num_of_class), filtered 2d list of span pair range (batch_size, num_of_span_pairs)
                        
        with torch.no_grad():
            h = self.span_scorer(x) # compute the probability that a span is in a relation
            x_ranked, span_ranges_ranked = self.get_top_k_percent_spans(x, h, span_ranges, self.k)
            x_paired, span_pair_ranges = self.get_pair_representations(x_ranked, span_ranges_ranked)
        
        x_ranked = self.dropout(x_ranked)
        logits_span_scorer = self.span_scorer(x_ranked)
        logits_pair_scorer = self.pair_scorer(x_paired)
        
        summed_logits = self.get_sum_of_span_scores_and_pair_score(logits_span_scorer, logits_pair_scorer, span_ranges, span_pair_ranges)
        
        logits_relation_scorer = summed_logits
        return logits_relation_scorer, span_pair_ranges
