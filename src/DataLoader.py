import torch
from torch.utils.data import DataLoader, Dataset
from BaseEncoder import BaseEncoder

USED_SAMPLE = 200

def read_examples_from_file(file_path, used_sample=USED_SAMPLE):
    texts = []
    te_label_list = [] # dictionary, key = string of token id span "TOKEN_SPAN", value = term type label (ASPECT, OPINION)
    relations = [] # dictionary, key = pair of token id span <ASPECT_SPAN, OPINION_SPAN>, value = sentiment polarity of the relation
    with open(file_path) as f:
        count_sentence = 0

        tokens = []
        te_labels = []
        relation_labels = []
        for line in f:
            line = line.strip()
            
            if line:
                line = line.split('\t')
                if len(line) == 8:
                    token = line[2]
                    te_and_relation_labels = line[3].split('|')
                    
                    for te_and_relation_label in te_and_relation_labels:
                        if any(x in te_and_relation_label for x in ['ASPECT', 'SENTIMENT']) or te_and_relation_label == '_':
                            te_label = te_and_relation_label
                        else:
                            relation_label = te_and_relation_label
                            relation_labels.append(relation_label)
                    
                    te_labels.append(te_label)
                    tokens.append(token)     
            elif tokens: # empty line    
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

                # convert te_labels to dict
                new_te_labels = {}
                for te_label in te_labels:
                    if '_' in te_label:
                        continue
                        
                    term_type = te_label.split('[')[0]
                    te_label_id = te_label.split('[')[1][:-1]
                    
                    te_label_idx = []
                    for i in range(len(te_labels)):
                        if '_' not in te_labels[i]:
                            current_label_id = te_labels[i].split('[')[1][:-1]
                            if te_label_id == current_label_id:
                                te_label_idx.append(i)
                    
                    new_te_labels[f'{te_label_idx[0]}-{te_label_idx[-1]}'] = term_type


                texts.append(tokens)
                te_label_list.append(new_te_labels)
                relations.append(new_relation_labels)

                tokens = []
                te_labels = []
                relation_labels = []
                count_sentence += 1

            if count_sentence == used_sample: # FOR TESTING
                break
                
    return texts, te_label_list, relations


class ReviewDataset(Dataset):
    def __init__(self, file_path, model_name_or_path="indolem/indobert-base-uncased", max_sentence_length=40):
        self.base_encoder= BaseEncoder(model_name_or_path)
        self.texts, self.te_label_list, self.relations = read_examples_from_file(file_path)
        self.transform = self.base_encoder.tokenize
        self.max_sentence_length = max_sentence_length
                                  
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        te_label_sequence = self.te_label_list[idx]
        
        if self.transform:
            text = self.transform(text, self.max_sentence_length)
            
        return text