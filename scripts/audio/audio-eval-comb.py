import torch
import sys
import os
import json
from math import exp

import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.audiovocab import SEPARATOR

MODEL = "jqv0lgv4"
STEP_NUM = 99481 #97645

DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt"

SUBSAMPLE = 1000
SUBSAMPLE_IDX = 0

model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'
#model_name = '/nlp/scr/kathli/checkpoints/efficient-sun-259/step-10000/hf'

# initialize the model and tokenizer
model_config_json = json.load(open(f"{model_name}/config.json"))
#model_config["n_positions"] = SHORT_SEQ_LEN
print("n_positions", model_config_json["n_positions"])
model_config = GPT2Config.from_dict(model_config_json)
model = GPT2LMHeadModel.from_pretrained(model_name, config=model_config)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(47)

# define the number of tokens to generate
num_tokens_to_generate = 8192-4

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
            logits = model(tokens).logits[0]
            #print("logits idx 1", F.softmax(logits[1],dim=0))
            #print("logits device", logits.get_device())
            #print("tokens device", tokens.get_device())
            logits = logits.cuda()
            cross_entr = F.cross_entropy(logits[:-4],tokens[0,4:],reduction='none')
            my_cross_entr = cross_entr.cpu()
            ## determine if my_cross_entr[i] > my_cross_entr[i+1] > my_cross_entr[i+2] > my_cross_entr[i+3] for all i
            #for i in range(0, my_cross_entr.shape[0], 4):
            #    if my_cross_entr[i] < my_cross_entr[i+1] and my_cross_entr[i+1] < my_cross_entr[i+2] and my_cross_entr[i+2] < my_cross_entr[i+3]:
            #        print("monotonic at index", i)
                   
            ce = torch.cat([ce, my_cross_entr])
            #break

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

