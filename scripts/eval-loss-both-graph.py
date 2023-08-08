import time

from statistics import mean
from math import exp

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import json

import sys
#sys.path.append('/home/kml11/transformers-levanter/src')
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from argparse import ArgumentParser
from tqdm import tqdm

from scipy.signal import lfilter

from anticipation import ops
from anticipation.config import M, EVENT_SIZE
from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION, AUTOREGRESS
from anticipation.ops import max_time

DATASET = "lakh-data-inter-16384"
SHORT_MODEL_NAME = "toasty-silence-91"
SHORT_STEP_NUMBER = 115000
LONG_MODEL_NAME = "driven-plant-48"
LONG_STEP_NUMBER = 30000
LONG_SEQ_LEN = 16384 # must be 1 mod 3
SHORT_SEQ_LEN = 1024 # must be 1 mod 3
NEW_CE_CUTOFF = 255 # must be less than SEQ_LEN
PRINT_GRAPH = True
PRINT_IDX = 3 # must be less than SUBSAMPLE, leave as 0 for default
PRINT_UP_TO_IDX = 8652 # none if we print the entire example
ADD_9999 = False
REMOVE_9999 = False
FILTER_CONST = 80
SUBSAMPLE=100
OUTPUT_DIR = f'/nlp/scr/kathli/output/figs'

SHORT_MODEL_LABEL = "single-stage short" #SHORT_MODEL_NAME
LONG_MODEL_LABEL = "finetune long" # LONG_MODEL_NAME

DATA = f"/nlp/scr/kathli/datasets/{DATASET}/test.txt" 
SHORT_CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{SHORT_MODEL_NAME}/step-{SHORT_STEP_NUMBER}/hf"
LONG_CHECKPOINT = f"/nlp/scr/kathli/checkpoints/{LONG_MODEL_NAME}/step-{LONG_STEP_NUMBER}/hf"

t0 = time.time()

short_config = json.load(open(f"{SHORT_CHECKPOINT}/config.json"))
short_config["n_positions"] = SHORT_SEQ_LEN
short_config = GPT2Config.from_dict(short_config)
short_model = GPT2LMHeadModel.from_pretrained(SHORT_CHECKPOINT, config=short_config).cuda() 
print(f'Loaded short model ({time.time()-t0} seconds)')
t1 = time.time()
long_config = json.load(open(f"{LONG_CHECKPOINT}/config.json"))
long_config["n_positions"] = LONG_SEQ_LEN
long_config = GPT2Config.from_dict(long_config)
long_model = GPT2LMHeadModel.from_pretrained(LONG_CHECKPOINT, config=long_config).cuda() 
print(f'Loaded long model ({time.time()-t1} seconds)')

if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized MIDI dataset')
    parser.add_argument('-f', '--filename',
                        help='file containing a tokenized MIDI dataset',
                        default=DATA)
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    args = parser.parse_args()

    print(f'Calculating log-loss for {args.filename}')
    print(f'Using short model {SHORT_CHECKPOINT}')
    print(f'Using long model {LONG_CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    should_print = PRINT_GRAPH
    with open(args.filename, 'r') as f:
        short_ce = torch.empty(0)
        long_ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != PRINT_IDX: continue
            
            tokens = [int(token) for token in line.split()]

            with torch.no_grad():
                idx = 1
                # do sliding window for short sequences
                curr_ce = torch.empty(0)
                while idx < len(tokens):
                    end_idx = min(len(tokens), idx + 1023)
                    curr_tokens = tokens[idx:end_idx]
                    if ADD_9999:
                        curr_tokens[0] = 9999 # make it match the samples a bit more
                    curr_tokens.insert(0, AUTOREGRESS)
                    curr_tokens = torch.tensor(curr_tokens).unsqueeze(0).cuda()
                    logits = short_model(curr_tokens).logits[0]
                    logits = logits.cuda()
                    cross_entr = F.cross_entropy(logits[:-1],curr_tokens[0,1:],reduction='none')
                    my_cross_entr = cross_entr.cpu()
                    if idx == 1:
                        curr_ce = torch.cat([curr_ce, my_cross_entr])
                    else:
                        # only add the last tokens
                        curr_ce = torch.cat([curr_ce, my_cross_entr[(SHORT_SEQ_LEN - NEW_CE_CUTOFF - 1):]])
                    idx += NEW_CE_CUTOFF

                    if idx + (SHORT_SEQ_LEN - NEW_CE_CUTOFF - 1) >= len(tokens):
                        break
                short_ce = torch.cat([short_ce, curr_ce])

                # do long sequences
                #print("len tokens", len(tokens))
                tokens = torch.tensor(tokens).unsqueeze(0).cuda()
                logits = long_model(tokens).logits[0]
                logits = logits.cuda()
                cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                long_ce = torch.cat([long_ce, my_cross_entr])

                if should_print:
                    print(f'printing example {i} to {OUTPUT_DIR}')
                    # print only the 3rd sample, the first 8652 tokens
                    curr_ce = curr_ce[:PRINT_UP_TO_IDX]
                    my_cross_entr = my_cross_entr[:PRINT_UP_TO_IDX]
                    should_print = False
                    # Create an array of indices from 0 to length-1 of `ce`
                    indices = np.arange(len(curr_ce))

                    # create a kalman filter to smooth the plot
                    n = FILTER_CONST  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
      
                    # Plot 1: ce
                    plt.figure(1)
                    plt.plot(indices, lfilter(b, a, curr_ce), label=SHORT_MODEL_LABEL, color='red', alpha=0.7)
                    plt.plot(indices, lfilter(b, a, my_cross_entr), label=LONG_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr')
                    plt.title('Plot of cross_entr values')

                    # Plot 2: ce[0::3]
                    plt.figure(2)
                    plt.plot(indices[0::3], lfilter(b, a, curr_ce[0::3]), label=SHORT_MODEL_LABEL, color='red', alpha=0.7)
                    plt.plot(indices[0::3], lfilter(b, a, my_cross_entr[0::3]), label=LONG_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[0::3]')
                    plt.title('Plot of cross_entr[0::3] values')

                    # Plot 3: ce[1::3]
                    plt.figure(3)
                    plt.plot(indices[1::3], lfilter(b, a, curr_ce[1::3]), label=SHORT_MODEL_LABEL, color='red', alpha=0.7)
                    plt.plot(indices[1::3], lfilter(b, a, my_cross_entr[1::3]), label=LONG_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[1::3]')
                    plt.title('Plot of cross_entr[1::3] values') 

                    # Plot 4: ce[2::3]
                    plt.figure(4)
                    plt.plot(indices[2::3], lfilter(b, a, curr_ce[2::3]), label=SHORT_MODEL_LABEL, color='red', alpha=0.7)
                    plt.plot(indices[2::3], lfilter(b, a, my_cross_entr[2::3]), label=LONG_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[2::3]')
                    plt.title('Plot of cross_entr[2::3] values')

                    # Save all four plots to image files
                    plt.figure(1)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{SHORT_MODEL_NAME}_{LONG_MODEL_NAME}_{DATASET}_{i}.png')

                    plt.figure(2)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{SHORT_MODEL_NAME}_{LONG_MODEL_NAME}_{DATASET}_{i}_0_3.png')

                    plt.figure(3)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{SHORT_MODEL_NAME}_{LONG_MODEL_NAME}_{DATASET}_{i}_1_3.png')

                    plt.figure(4)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{SHORT_MODEL_NAME}_{LONG_MODEL_NAME}_{DATASET}_{i}_2_3.png')
                    #break

        print('num samples', num_samples)
        print(num_samples * (LONG_SEQ_LEN - 1))

        print(f"SHORT MODEL {SHORT_MODEL_NAME}")
        print('ce', short_ce)
        short_L = short_ce.mean()
        print('Tokens processed:', len(short_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', short_L)
        print('  -> bits per second: ', short_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*short_ce.mean()))
            print('  -> onset perplexity: ', exp(short_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(short_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(short_ce[2::3].mean()))

        print(f"LONG MODEL {LONG_MODEL_NAME}")
        print('ce', long_ce)
        long_L = long_ce.mean()
        print('Tokens processed:', len(long_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', long_L)
        print('  -> bits per second: ', long_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*long_ce.mean()))
            print('  -> onset perplexity: ', exp(long_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(long_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(long_ce[2::3].mean()))

