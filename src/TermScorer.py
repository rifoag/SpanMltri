import torch
from torch import nn

# Define model
class TermScorer(nn.Module):
    def __init__(self, d_hidden=768, max_sentence_length=40, num_of_class=3):
        super(TermScorer, self).__init__()
        self.linear_softmax = nn.Sequential(
            nn.Linear(d_hidden, num_of_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_softmax(x)
        return logits
