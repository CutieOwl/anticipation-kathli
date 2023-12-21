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

# MIDI models
MODEL = "pu4yo6b5"# "ha05xrd3" # "54labs45" #" #"9qbavecu"
STEP_NUM = 81001 #99920 #50000 #99588 # ##99698  #42430

# AUDIO models
#MODEL = "54labs45" # "vl5058w4" #"9qbavecu"
#STEP_NUM = 99920 #50000 #99588 #42430

#DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt" # this is the old audio tokenization dataset

# don't have a local test set so eval on validation
DATA = "/juice4/scr4/nlp/music/datasets/encodec_fma.audiogen.valid.txt"

#DATA = "/juice4/scr4/nlp/music/temp_test/lakh.midigen.test.txt" # this is the clean midi dataset
#DATA_2 = "/juice4/scr4/nlp/music/temp_test/encodec_fma.trans_midigen.test.txt" # this is the transcribed midi dataset

SUBSAMPLE = 1000
SUBSAMPLE_IDX = 0
NEW_CE_CUTOFF = 255

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

ce = torch.empty(0)
new_ce = torch.empty(0)

def eval_f(datafile, model, num_samples):
    ce = torch.empty(0)
    new_ce = torch.empty(0)
    with open(datafile, 'r') as f:    
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
                cross_entr = F.cross_entropy(logits[3:-1],tokens[0,4:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                ce = torch.cat([ce, my_cross_entr])
                new_ce = torch.cat([new_ce, my_cross_entr[-NEW_CE_CUTOFF:]])
    return ce, new_ce

my_ce, my_new_ce = eval_f(DATA, model, num_samples)
#my_ce2, my_new_ce2 = eval_f(DATA_2, model, num_samples)

ce = my_ce #torch.cat([my_ce, my_ce2])
new_ce = my_new_ce #torch.cat([my_new_ce, my_new_ce2])


print('Model:', MODEL)
print('Step:', STEP_NUM)
print('Data:', DATA)
#print('Data 2:', DATA_2)
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

print('------------------------------')
print(f'Calculations on the last {NEW_CE_CUTOFF} tokens')
L = new_ce.mean()
print('Tokens processed:', len(new_ce))
print('Log-losses')
print('  -> per-token log-loss (nats): ', L)
print('  -> per-event perplexity: ', exp(4*new_ce.mean()))
print('  -> 0 perplexity: ', exp(new_ce[0::4].mean()))
print('  -> 1 perplexity: ', exp(new_ce[1::4].mean()))
print('  -> 2 perplexity: ', exp(new_ce[2::4].mean()))
print('  -> 3 perplexity: ', exp(new_ce[3::4].mean()))