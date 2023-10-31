import time

from statistics import mean
from math import exp

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import json
import gzip

import sys
#sys.path.append('/home/kml11/transformers-levanter/src')
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from argparse import ArgumentParser
from tqdm import tqdm

from scipy.signal import lfilter

from anticipation import ops
from anticipation.config import M, EVENT_SIZE
from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION, AUTOREGRESS
from anticipation.ops import max_time

DATA_DIR = '/jagupard31/scr1/kathli/openwebtext'
MODEL_NAME = "wcr3aa1k"
STEP_NUMBER = 11763
SEQ_LEN = 1024
NEW_CE_CUTOFF = 255
PRINT_IDX = 0 # must be less than SUBSAMPLE, leave as 0 for default
SUBSAMPLE=1
OUTPUT_DIR = f'/nlp/scr/kathli/eval/2hr'

json_content = []
for i in range(1, 9):
    with gzip.open(f'{DATA_DIR}/openwebtext_val.1-of-8.jsonl.gz', 'rb') as gzip_file:
        for line in gzip_file:  # Read one line.
            line = line.rstrip()
            if line:  # Any JSON data on it?
                obj = json.loads(line)
                json_content.append(obj)

print(len(json_content))

if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized MIDI dataset')
    parser.add_argument('-m', '--model_name',
            help='model to be evaluated',
            default=MODEL_NAME)
    parser.add_argument('-n', '--step_number',
            type=int,
            help='step number of the model to be evaluated',
            default=STEP_NUMBER)     
    parser.add_argument('-s', '--subsample', type=int, default=SUBSAMPLE,
            help='dataset subsampling ratio')   

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    STEP_NUMBER = args.step_number
    SUBSAMPLE = args.subsample

    CHECKPOINT= f"/jagupard31/scr1/kathli/2hr_checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"

    t0 = time.time()

    config = json.load(open(f"{CHECKPOINT}/config.json"))
    print("config", config)
    config = GPT2Config.from_dict(config)
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT)
    print(f'Loaded model ({time.time()-t0} seconds)')

    print(f'Using model {CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    ce = torch.empty(0)
    new_ce = torch.empty(0)
    for i in tqdm(range(len(json_content))):
        num_samples += 1
        if i % SUBSAMPLE != PRINT_IDX: continue
        line = json_content[i]['text']
        
        input_ids = tokenizer.encode(line, return_tensors='pt').cuda()

        if input_ids.shape[1] > 1024:
            #print(f"had to truncate input {i}, original length {input_ids.shape[1]}")
            input_ids = input_ids[:, :1024]

        #print(input_ids.shape)
        #print(input_ids)

        with torch.no_grad():
            logits = model(input_ids).logits[0]
            logits = logits.cuda()
            cross_entr = F.cross_entropy(logits[:-1],input_ids[0,1:],reduction='none')
            my_cross_entr = cross_entr.cpu()
            ce = torch.cat([ce, my_cross_entr])
            new_ce = torch.cat([new_ce, my_cross_entr[-NEW_CE_CUTOFF:]])

    print('ce', ce)
    print('num samples', num_samples)
    print(num_samples * (SEQ_LEN - 1)) # 1023 
    L = ce.mean().numpy().tolist()
    V = ce.var().numpy().tolist()
    print('Tokens processed:', len(ce))
    print('Log-losses')
    print('  -> per-token log-loss (nats): ', L)
    print('  -> log loss variance: ', V)

    ret_obj = {}
    ret_obj['model'] = MODEL_NAME
    ret_obj['step'] = STEP_NUMBER
    ret_obj['num_samples'] = num_samples
    ret_obj['mean'] = L
    ret_obj['variance'] = V
    ret_obj['ce'] = ce.numpy().tolist()

    print('------------------------------')
    print(f'Calculations on the last {NEW_CE_CUTOFF} tokens')
    L = new_ce.mean().numpy().tolist()
    V = new_ce.var().numpy().tolist()
    print('Tokens processed:', len(new_ce))
    print('Log-losses')
    print('  -> per-token log-loss (nats): ', L)
    print('  -> log loss variance: ', V)

    ret_obj['cutoff_mean'] = L
    ret_obj['cutoff_variance'] = V

    # save ret_obj to file
    with open(f'{OUTPUT_DIR}/eval-loss-openwebtext_{MODEL_NAME}.json', 'w') as f:
        f.write(json.dumps(ret_obj))
