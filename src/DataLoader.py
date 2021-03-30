import torch
from torch.utils.data import DataLoader, Dataset
from BaseEncoder import BaseEncoder
    
def read_examples_from_file(file_path):
    texts = []
    te_label_sequences = []
    relations = [] # dictionary, key = pair of token id span <ASPECT_SPAN, OPINION_SPAN>, value = sentiment polarity of the relation
    with open(file_path) as f:
        count_sentence = 0

        tokens = []
        te_labels = []
        relation_labels = []
        for line in f:
            line = line.strip()
            if not line and tokens: # empty line                    
                # convert relation to dict
                new_relation_labels = {}
                for relation_label in relation_labels:
                    relation_polarity = relation_label.split('[')[0]
                    ids = relation_label.split('[')[1]

                    ate_label_id = ids.split('_')[0]
                    ote_label_id = ids.split('_')[1][:-1]

                    ate_label_idx = []
                    for i in range(len(te_labels)):
                        if '_' not in te_labels[i]:
                            label_id = te_labels[i].split('[')[1][:-1]
                            if ate_label_id == label_id:
                                ate_label_idx.append(i)

                    ote_label_idx = []
                    for i in range(len(te_labels)):
                        if '_' not in te_labels[i]:
                            label_id = te_labels[i].split('[')[1][:-1]
                            if ote_label_id == label_id:
                                ote_label_idx.append(i)

                    new_relation_labels[(f'{ote_label_idx[0]}-{ote_label_idx[-1]}', f'{ate_label_idx[0]}-{ate_label_idx[-1]}')] = relation_polarity

                # convert te label to iob
                prev_label_id = -1
                new_te_labels = []
                for te_label in te_labels:
                    if te_label == '_':
                        new_te_labels.append('O')
                    else:
                        label = te_label.split('[')[0]
                        label_id = te_label.split('[')[1][:-1]
                        if label_id == prev_label_id:
                            new_te_labels.append(f'I-{label}')
                        else:
                            new_te_labels.append(f'B-{label}')
                        prev_label_id = label_id


                texts.append(tokens)
                te_label_sequences.append(new_te_labels)
                relations.append(new_relation_labels)

                tokens = []
                te_labels = []
                relation_labels = []
                count_sentence += 1

            if count_sentence == 5:
                break

            line = line.split('\t')
            if len(line) == 8:
                token = line[2]
                te_and_relation_labels = line[3].split('|')

                if len(te_and_relation_labels) > 1:
                    te_label = te_and_relation_labels[0] if ('ASPECT' or 'SENTIMENT') in te_and_relation_labels[0] else te_and_relation_labels[1]
                    relation_label = te_and_relation_labels[1] if ('ASPECT' or 'SENTIMENT') in te_and_relation_labels[0] else te_and_relation_labels[0]
                    relation_labels.append(relation_label)
                else:
                    te_label = te_and_relation_labels[0]

                tokens.append(token)
                te_labels.append(te_label)
    
    return texts, te_label_sequences, relations

def transform_labels(te_label_sequence, max_sentence_length=40):
    te_label_map = {
        'B-ASPECT': [1, 0, 0, 0, 0],
        'I-ASPECT': [0, 1, 0, 0, 0],
        'B-SENTIMENT': [0, 0, 1, 0, 0],
        'I-SENTIMENT': [0, 0, 0, 1, 0],
        'O': [0, 0, 0, 0, 1]
    }
    
    te_label_sequence = [te_label_map[te_label] for te_label in te_label_sequence]
    if len(te_label_sequence) < max_sentence_length:
        for i in range(max_sentence_length - len(te_label_sequence)):
            te_label_sequence.append([0, 0, 0, 0, 1])
    else:
        te_label_sequence = te_label_sequence[:max_sentence_length]
        
    return torch.Tensor(te_label_sequence)

class ReviewDataset(Dataset):
    def __init__(self, file_path, target_transform=transform_labels, model_name_or_path="indolem/indobert-base-uncased", max_sentence_length=40):
        self.base_encoder= BaseEncoder(model_name_or_path)
        self.texts, self.te_label_sequences, self.relations = read_examples_from_file(file_path)
        self.transform = self.base_encoder.tokenize
        self.target_transform = target_transform
        self.max_sentence_length = max_sentence_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        te_label_sequence = self.te_label_sequences[idx]
        relations = self.relations[idx].copy()
        
        relation_label_map = {
            'PO': torch.Tensor([1, 0, 0]),
            'NG': torch.Tensor([0, 1, 0]),
            'O': torch.Tensor([0, 0, 1])
        }
        
        for key in relations:
            relations[key] = relation_label_map[relations[key]]
        
        if self.transform:
            text = self.transform(text, self.max_sentence_length)
        if self.target_transform:
            te_label_sequence = self.target_transform(te_label_sequence, self.max_sentence_length)
            
        return text, te_label_sequence