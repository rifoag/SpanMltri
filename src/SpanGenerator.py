import torch
from torch import nn

CLS_OFFSET = 1

class SpanGenerator(nn.Module):
    def __init__(self, max_span_length=8):
        super(SpanGenerator, self).__init__()
        self.max_span_length = max_span_length
    
    def forward(self, tensor):
        # input tensor shape : (batch_size, sentence_size, d_hidden)
        # output tensor shape: (batch_size, sentence_size, num_of_span, d_hidden)
        batch_size, sentence_size, d_hidden = tensor.shape
        
        span_tensors_batch = []
        
        for batch in range(batch_size):
            span_tensors = []
            span_ranges = []
            for span_length in range(1, self.max_span_length+1):
                start_idx = 1
                end_idx = span_length
                for i in range(sentence_size-span_length):
                    span_tensor = tensor[batch, start_idx+i:end_idx+i+1, :]
                    span_tensors.append(span_tensor.sum(dim=0))
                    span_ranges.append(f'{start_idx+i-1}-{end_idx+i-1}')
            span_tensors_batch.append(torch.stack(span_tensors))
                
        return torch.stack(span_tensors_batch), span_ranges
    