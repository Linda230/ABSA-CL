# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
import copy
import math


class BERT_SPC_CL(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC_CL, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs

        text_embed, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers = False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits, pooled_output

