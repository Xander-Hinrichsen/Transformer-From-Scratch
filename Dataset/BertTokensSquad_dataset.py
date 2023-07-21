import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, AutoModel

train = pd.read_csv('Dataset/SquadDataset/SQuAD-v1.1.csv')
train = train.get(['question', 'answer'])
train.question = train.question.apply(lambda x: x if type(x) == str else '')
train.answer = train.answer.apply(lambda x: x if type(x) == str else '')
train = train[(train.question != '') & (train.answer != '')]
train

def get_all_chars(df, field):
    sentences = np.array(df.get(field))
    all_chars = {}
    for i in range(len(sentences)):
        chars = ' '.join(sentences[i])
        for j in range(len(chars)):
            all_chars[chars[j]] = 0
    return np.array(sorted(list(all_chars.keys())))

question_chars = get_all_chars(train, 'question')
answer_chars = get_all_chars(train, 'answer')
all_chars = np.array(sorted(np.unique(np.append(question_chars, answer_chars))))
all_chars

banned_chars = all_chars[92:]
accepted_chars = all_chars[:92]
accepted_chars, banned_chars

#these are the symbols and punctuation tokens
#don't include the ' ' because we're gonna .split(' ')
single_char_tokens = np.array(['!', '"', '#', '$', '%', '&', '(', ')', '*', '+',
                       ',','-', '.', '/',':', ';', '<', '=', '>', '?', '@','[', ']',
                       '_', '`','{', '}','~', "'"])
single_char_tokens

def master_filter(x, banned_chars=banned_chars, single_char_tokens=single_char_tokens):
    for char in banned_chars:
        if char in x:
            return ''
    return x

train.question = train.question.apply(master_filter)
train.answer = train.answer.apply(master_filter)
train = train[(train.question != '') & (train.answer != '')]
train

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def to_idx_quest(x):
    tokens = tokenizer.encode(x)
    return tokens[1:-1]

def to_idx_target(x):
    tokens = tokenizer.encode(x)
    return tokens[1:]

def to_idx_decoder_input(x):
    tokens = tokenizer.encode(x)
    return tokens[:-1]

train = (train.assign(question_tokens=train.question.apply(to_idx_quest))
         .assign(decoder_input=train.answer.apply(to_idx_decoder_input))
         .assign(target=train.answer.apply(to_idx_target)))
train

class Dataset:
    def __init__(self, df=train, fake_len=None):
        self.questions = np.array(df.question)
        self.answer = np.array(df.answer)
        self.question_tokens = np.array(df.question_tokens)
        self.decoder_input = np.array(df.decoder_input)
        self.target = np.array(df.target)
        self.fake_len = fake_len
    def __getitem__(self, i):
        return (torch.LongTensor(self.question_tokens[i]), len(self.question_tokens[i]), 
                torch.LongTensor(self.decoder_input[i]), len(self.decoder_input[i]),
                torch.LongTensor(self.target[i]))
    def __len__(self):
        if self.fake_len == None:
            return len(self.questions)
        else:
            return self.fake_len

class DataLoader:
    def __init__(self, ds, batch_size=64, shuffle=False):
        self.ds = ds
        self.batch_size = batch_size
        self.idxs = np.arange(len(ds))
        if shuffle:
            self.idxs = np.random.permutation(self.idxs)
    def __iter__(self):
        idx = 0
        questions = []
        question_len = []
        decoder_input = []
        decoder_len = []
        target = []
        while True:
            if idx != 0 and ((idx % self.batch_size == 0) or idx == len(self.ds)):
                max_ques = np.max(question_len)
                max_ans = np.max(decoder_len)
                for i in range(len(questions)):
                    questions[i] = np.append(questions[i], np.zeros(max_ques - question_len[i]))
                    decoder_input[i] = np.append(decoder_input[i], np.zeros(max_ans - decoder_len[i]))
                    target[i] = np.append(target[i], np.zeros(max_ans - decoder_len[i]))
                questions = np.array(questions, int)
                decoder_input = np.array(decoder_input, int)
                target = np.array(target, int)
                yield (torch.LongTensor(questions),
                       torch.LongTensor(decoder_input),
                       torch.LongTensor(target))
                questions = []
                question_len = []
                decoder_input = []
                decoder_len = []
                target = []
                if idx == len(self.ds):
                    break
            questions.append(self.ds[self.idxs[idx]][0])
            question_len.append(self.ds[self.idxs[idx]][1])
            decoder_input.append(self.ds[self.idxs[idx]][2])
            decoder_len.append(self.ds[self.idxs[idx]][3])
            target.append(self.ds[self.idxs[idx]][4])
            idx+=1
    def __len__(self):
        length = (len(self.ds) // self.questions)
        if len(self.ds) % self.questions != 0:
            length +=1
        return length