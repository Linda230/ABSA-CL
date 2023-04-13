# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


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

class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


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


class ABSAAugDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []

        for i in range(0, len(lines), 4):
            text1_left, _, text1_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polabel = lines[i + 2].strip()
            text2_left, _, text2_right = [s2.lower().strip() for s2 in lines[i+3].partition("$T$")]

            text1_indices = tokenizer.text_to_sequence(text1_left + " " + aspect + " " + text1_right)
            text2_indices = tokenizer.text_to_sequence(text2_left + " " + aspect + " " + text2_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1
            polabel = int(polabel) + 1

            text1_len = np.sum(text1_indices != 0)
            text2_len = np.sum(text2_indices != 0)
            concat_text1_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + text1_left + " " + aspect + " " + text1_right + ' [SEP] ' + aspect + " [SEP]")
            concat_text1_segments_indices = [0] * (text1_len + 2) + [1] * (aspect_len + 1)
            concat_text1_segments_indices = pad_and_truncate(concat_text1_segments_indices, tokenizer.max_seq_len)

            concat_text2_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + text2_left + " " + aspect + " " + text2_right + ' [SEP] ' + aspect + " [SEP]")
            concat_text2_segments_indices = [0] * (text2_len + 2) + [1] * (aspect_len + 1)
            concat_text2_segments_indices = pad_and_truncate(concat_text2_segments_indices, tokenizer.max_seq_len)

            data = {
                'concat_bert_indices': concat_text1_bert_indices,
                'concat_segments_indices': concat_text1_segments_indices,
                'concat_text2_bert_indices': concat_text2_bert_indices,
                'concat_text2_segments_indices': concat_text2_segments_indices,
                'text1_indices': text1_indices,
                'text2_indices': text2_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text1': lines[i],
                'text2': lines[i+3],
                'aspect': aspect,
                'polabel': polabel,
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADataset(Dataset):

    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

        lines = fin.readlines()

        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polabel = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1
            polabel = int(polabel) + 1

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            # attention_mask = [1] * (text_len + 2 + aspect_len + 1)
            # attention_mask = pad_and_truncate(attention_mask, tokenizer.max_seq_len)

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text': lines[i],
                'aspect': aspect,
                'polabel': polabel,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
