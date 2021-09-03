import torch
from torch.utils.data import DataLoader, Dataset
from BaseEncoder import BaseEncoder
from transformers import AutoTokenizer

class ReviewDataset(Dataset):
    def __init__(self, file_path, model_name_or_path="indolem/indobert-base-uncased", max_sentence_length=40):
        self.base_encoder= BaseEncoder(model_name_or_path)
        self.max_sentence_length = max_sentence_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.texts, self.te_label_dict, self.relation_dict = self.read_examples_from_file(file_path)
        self.transform = AutoTokenizer.from_pretrained(model_name_or_path)
        
    
    def align_tokens_and_labels(self, tokens, unaligned_labels):
        tokenized_inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                add_special_tokens=False
            )

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append('_')
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(unaligned_labels[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(unaligned_labels[word_idx])
            previous_word_idx = word_idx


        tokenized_inputs["labels"] = label_ids
        tokenized_inputs["tokens"] = self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"], skip_special_tokens=True)
        return tokenized_inputs["tokens"], tokenized_inputs["labels"]

    def read_examples_from_file(self, file_path):
        texts = []
        te_label_dict = [] # dictionary, key = string of token id span "TOKEN_SPAN", value = term type label (ASPECT, OPINION)
        relation_dict = [] # dictionary, key = pair of token id span <ASPECT_SPAN, OPINION_SPAN>, value = sentiment polarity of the relation
        label_seq = []
        with open(file_path) as f:
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
                    tokens, te_labels = self.align_tokens_and_labels(tokens, te_labels)
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
                    te_label_dict.append(new_te_labels)
                    relation_dict.append(new_relation_labels)
                    label_seq.append(te_labels)

                    tokens = []
                    te_labels = []
                    relation_labels = []

            if tokens:
                texts.append(tokens)
                te_label_dict.append(new_te_labels)
                relation_dict.append(new_relation_labels)
                label_seq.append(te_labels)

        return texts, te_label_dict, relation_dict
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.transform:
            text = self.transform(text, return_tensors="pt", is_split_into_words=True, padding='max_length', max_length=self.max_sentence_length, truncation=True)
            
        return text['input_ids']