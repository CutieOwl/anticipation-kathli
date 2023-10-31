import time

from statistics import mean
from math import exp

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import json
import os

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
MODEL_NAME = "fh86cy5o"
STEP_NUMBER = 115000
SEQ_LENS = [4, 16, 64, 256, 1024, 4096, 16384] # must be 1 mod 3
MAX_SEQ_LEN = max(SEQ_LENS)
PRINT_GRAPH = True
SUBSAMPLE = 100
SUBSAMPLE_IDX = 0
FILTER_CONST = 50

DATA = f"/nlp/scr/kathli/datasets/{DATASET}/test.txt" 
OUTPUT_DIR = f'/nlp/scr/kathli/eval/rep_struct/figs_mean_var_filt{FILTER_CONST}'

# if output_dir does not exist create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def sliding_window_eval(tokens, model, seq_len, slide_amount=None):
    slide_a = slide_amount
    if slide_a is None:
        slide_a = seq_len // 4
    idx = 1
    # do sliding window for short sequences
    curr_ce = torch.empty(0)
    while idx < len(tokens):
        end_idx = min(len(tokens), idx + seq_len - 1)
        curr_tokens = tokens[idx:end_idx]
        curr_tokens.insert(0, AUTOREGRESS)
        curr_tokens = torch.tensor(curr_tokens).unsqueeze(0).cuda()
        logits = model(curr_tokens).logits[0]
        logits = logits.cuda()
        cross_entr = F.cross_entropy(logits[:-1],curr_tokens[0,1:],reduction='none')
        my_cross_entr = cross_entr.cpu()
        if idx == 1:
            curr_ce = torch.cat([curr_ce, my_cross_entr])
        else:
            # only add the last tokens
            curr_ce = torch.cat([curr_ce, my_cross_entr[(seq_len - slide_a - 1):]])
        idx += slide_a

        if idx + (seq_len - slide_a - 1) >= len(tokens):
            break
    return curr_ce

if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized MIDI dataset')
    parser.add_argument('-f', '--filename',
                        help='file containing a tokenized MIDI dataset',
                        default=DATA)
    parser.add_argument('-m', '--model_name', 
                        help='model to be evaluated',
                        default=MODEL_NAME)
    parser.add_argument('-n', '--step_number',
                        type=int,
                        help='step number of the model to be evaluated',
                        default=STEP_NUMBER)    
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    args = parser.parse_args()

    num_samples = 0
    means = []
    variances = []
    with open(args.filename, 'r') as f:
        file_lines = f.readlines()

    t0 = time.time()

    MODEL_NAME = args.model_name
    STEP_NUMBER = args.step_number
    CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
    config = json.load(open(f"{CHECKPOINT}/config.json"))
    config["n_positions"] = MAX_SEQ_LEN
    config = GPT2Config.from_dict(config)
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda() #
    print(f'Loaded model ({time.time()-t0} seconds)')

    print(f'Calculating log-loss for {args.filename}')
    print(f'Using model {CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')

    avg_ces = []
    for SEQ_LEN in SEQ_LENS:
        ce = torch.empty(0)
        avg_ce = torch.zeros(16383)
        num_subsamples = 0
        with tqdm(total=len(file_lines), desc=f'Processing seq_len {SEQ_LEN}') as pbar:
            for i,line in tqdm(enumerate(file_lines)):
                num_samples += 1
                if i % SUBSAMPLE != SUBSAMPLE_IDX: continue
                num_subsamples += 1
                
                tokens = [int(token) for token in line.split()]
                if tokens[0] != AUTOREGRESS:
                    tokens.insert(0, AUTOREGRESS)

                with torch.no_grad():
                    if SEQ_LEN == MAX_SEQ_LEN:
                        full_tokens = torch.tensor(tokens).unsqueeze(0).cuda()
                        logits = model(full_tokens).logits[0]
                        logits = logits.cuda()
                        cross_entr = F.cross_entropy(logits[:-1],full_tokens[0,1:],reduction='none')
                        curr_ce = cross_entr.cpu()   
                    else:
                        curr_ce = sliding_window_eval(tokens, model, SEQ_LEN)
                        
                    if FILTER_CONST > 0:
                         # create a kalman filter to smooth the plot
                        n = FILTER_CONST  # the larger n is, the smoother curve will be
                        b = [1.0 / n] * n
                        a = 1
                        curr_ce = lfilter(b, a, curr_ce)
                        ce = torch.cat([ce, torch.from_numpy(curr_ce)])
                    else:
                        ce = torch.cat([ce, curr_ce])
                    avg_ce += curr_ce
                pbar.update(SUBSAMPLE)
                
        avg_ce /= num_subsamples
        avg_ces.append(avg_ce)
        means.append(ce.mean()) 
        variances.append(ce.var())
        print(f'Seq_len {SEQ_LEN}: Mean={ce.mean()}, Variance={ce.var()}')

    plt.figure(1)
    plt.semilogx(SEQ_LENS, variances, marker='o', linestyle='-', label='Variance')
    # Label the axes and title
    plt.xlabel('Sequence length of sliding window')
    plt.ylabel('Cross-entropy')
    plt.xscale('log', base=2)
    plt.title(f'Cross-entropies of sliding windows on model {MODEL_NAME}')
    # show the legend
    plt.legend()
    # Save the plot as an image (e.g., PNG, JPEG, or PDF)
    plt.savefig(f'{OUTPUT_DIR}/variances_{MODEL_NAME}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')

    plt.semilogx(SEQ_LENS, means, marker='o', linestyle='-', label='Mean')
    plt.legend()
    plt.xscale('log', base=2)
    # Save the plot as an image (e.g., PNG, JPEG, or PDF)
    plt.savefig(f'{OUTPUT_DIR}/{MODEL_NAME}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')

    for i in range(len(SEQ_LENS)):
        plt.figure(i+2)
        plt.plot(avg_ces[i], label=f'Avg CE')
        # Label the axes and title
        plt.xlabel('Token index')
        plt.ylabel('Cross-entropy')
        plt.title(f'Avg CE of sliding windows on model {MODEL_NAME} with kalman filter {FILTER_CONST}')
        plt.legend()
        plt.savefig(f'{OUTPUT_DIR}/avg_ce_{SEQ_LENS[i]}_{MODEL_NAME}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')

    with open(f'{OUTPUT_DIR}/mean_variance_{MODEL_NAME}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.data.txt', 'w') as file:
        for i in range(len(SEQ_LENS)):
            file.write(f'Seq_len {SEQ_LENS[i]}: Mean={means[i]}, Variance={variances[i]}\n')
