import torch
import sys
import os
from math import exp

import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.audiovocab import SEPARATOR

MODEL = "49fupsao"
STEP_NUM = 42430

DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt" # this is the old audio tokenization dataset
#DATA = "/juice4/scr4/nlp/music/datasets/valid/lakh.midigen.valid.txt" # this is the new midi tokenization dataset

SUBSAMPLE = 1000
SUBSAMPLE_IDX = 0

# initialize the model and tokenizer
model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf/'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'
model = GPT2LMHeadModel.from_pretrained(model_name)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(47)

num_samples = 0

with open(DATA, 'r') as f:
    ce = torch.empty(0)
    for i,line in tqdm(list(enumerate(f))):
        num_samples += 1
        if i % SUBSAMPLE != SUBSAMPLE_IDX: continue
        tokens = [int(token) for token in line.split()]
        #print(tokens)
        tokens = torch.tensor(tokens).unsqueeze(0).cuda()

        with torch.no_grad():
            logits = model(tokens).logits[0].cuda()
            #print(logits.shape)
            #print(tokens.shape)
            cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
            my_cross_entr = cross_entr.cpu()
            ce = torch.cat([ce, my_cross_entr])

    print('ce', ce)
    print('num samples', num_samples)
    print(num_samples * 8192) # 1023 
    L = ce.mean()
    print('Tokens processed:', len(ce))
    print('Log-losses')
    print('  -> per-token log-loss (nats): ', L)

    print('  -> per-event perplexity: ', exp(4*ce.mean()))
    print('  -> 0 perplexity: ', exp(ce[0::4].mean()))
    print('  -> 1 perplexity: ', exp(ce[1::4].mean()))
    print('  -> 2 perplexity: ', exp(ce[2::4].mean()))
    print('  -> 3 perplexity: ', exp(ce[3::4].mean()))