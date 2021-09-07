import torch
from torch import nn

# Define model
class TermScorer(nn.Module):
    def __init__(self, d_hidden=768, max_sentence_length=40, num_of_class=3):
        super(TermScorer, self).__init__()
        self.dropout = nn.Dropout(0.10)
        self.hidden = nn.Sequential(
            nn.Linear(d_hidden, 768),
            nn.ReLU()
        )

        self.linear = nn.Linear(768, num_of_class)

    def forward(self, x):
        # input x (batch_size, num_of_span, d_hidden)
        x = self.dropout(x)
        h = self.hidden(x)
        logits = self.linear(h)
        return logits
