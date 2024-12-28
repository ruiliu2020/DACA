import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def sequence_to_text(self, sequence, reverse=False, padding='post', truncating='post'):
        text = self.tokenizer.convert_ids_to_tokens(sequence)
        return text

class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        num = 0
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            total_context_len = np.sum(text_raw_indices != 0)
            if total_context_len > 85:
                print(lines[i])
                num += 1
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            left_context_with_aspect_len = np.sum(text_left_with_aspect_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1

            src_mask = [0] + [1] * (np.sum(text_raw_indices != 0)) + [0] * (tokenizer.max_seq_len - (np.sum(text_raw_indices != 0)) - 1)
            src_mask = src_mask[:tokenizer.max_seq_len]
            src_mask = np.asarray(src_mask, dtype='int64')

            aspect_mask = [0] + [0] * left_context_len + [1] * aspect_len + [0] * (tokenizer.max_seq_len - left_context_with_aspect_len - 1)
            if aspect_len == 0:
                print(text_left)
            aspect_mask = aspect_mask[:tokenizer.max_seq_len]
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            if left_context_len + 1 >= tokenizer.max_seq_len:
                aspect_mask[0] = 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")

            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)         

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'polarity': polarity,
                'src_mask': src_mask,
                'aspect_mask':aspect_mask
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def build_or_load_dataset(opt,mode='train'):
    """
    Load the corresponding dataset for train/eval
    This is the top wrapper for different tasks, data splits, or exp_type
    """
    # set the tag vocab
    # _, tag2idx, _ = get_tag_vocab(task=args.task, tagging_schema=args.tagging_schema, 
    #                               label_path=args.label_path)

    # get the correponding data file path according to the exp_type and mode
    exp_type = opt.exp_type
    file_name_or_list = []
    if mode == 'dev':
        # "gold-en-dev.tex" as DEV
        file_name_or_list = f'gold-{opt.src_lang}-dev.txt'

    elif mode == 'test':
        # "gold-fr-test" as TEST (to be consistent with previous works)
        file_name_or_list = f"gold-{opt.tgt_lang}-test.txt"

    elif mode == 'unlabeled':
        # use unlabeled data (without considering the labels)
        file_name_or_list = f"gold-{opt.tgt_lang}-train.txt"

    elif mode == 'unlabeled_mtl':
        file_name_or_list = [f"gold-{l}-train.txt" for l in ['fr', 'es', 'nl', 'ru']]

    elif mode == 'train':
        # Supervised setting (to have a "upperbound")
        # e.g., 'gold-en-train.txt'
        if exp_type == 'supervised':
            assert opt.src_lang == opt.tgt_lang, "Src and Tgt langs should be the same under supervised setting!"
            file_name_or_list = f"gold-{opt.src_lang}-train.txt"

        # Translate-train setting
        # need to have the transalted data such as 'smt-fr-train.txt'
        elif exp_type == 'smt':
            file_name_or_list = f"{exp_type}-{opt.tgt_lang}-train.txt"

        elif exp_type.startswith('mtl'):
            file_name_or_list = ['gold-en-train.txt']
            file_name_or_list += [f'smt-{l}-train.txt' for l in ['fr', 'es', 'nl', 'ru']]

        # Proposed ACS method
        # need to have the code switching data such as 'cs-en-fr-train.txt'
        elif exp_type == 'acs':
            file_name_or_list = [f"gold-{opt.src_lang}-train.txt", 
                                 f"cs_{opt.src_lang}-{opt.tgt_lang}-train.txt",
                                 f"cs_{opt.tgt_lang}-{opt.src_lang}-train.txt",
                                 f"smt-{opt.tgt_lang}-train.txt"]           

        elif exp_type == 'acs_mtl':
            lang_list = ['fr', 'es', 'nl', 'ru']
            file_name_or_list = ["gold-en-train.txt"]
            file_name_or_list += [f'smt-{l}-train.txt' for l in lang_list]
            file_name_or_list += [f'cs_en-{l}-train.txt' for l in lang_list]
            file_name_or_list += [f'cs_{l}-en-train.txt' for l in lang_list]
    return file_name_or_list
