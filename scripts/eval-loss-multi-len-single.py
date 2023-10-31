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
from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION, AUTOREGRESS, SEPARATOR
from anticipation.ops import max_time

DATASET = "lakh-data-inter-16384"
MODEL_NAME = "uk31obfe" #"fh86cy5o"
STEP_NUMBER = 115000
SEQ_LENS = [4, 16, 64, 256, 1024, 4096, 16384] # must be 1 mod 3
MAX_SEQ_LEN = max(SEQ_LENS)
PRINT_GRAPH = True
SUBSAMPLE = 100
SUBSAMPLE_IDX = 0

FILTER_CONST = 50
PRINT_UP_TO_IDX = 8652

DATA_DIR = f"/nlp/scr/kathli/eval/rep_struct/" 
CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
OUTPUT_DIR = f'/nlp/scr/kathli/eval/rep_struct/figs_mean_var_nofilt'

DATAFILE = 'f2eb26e6'

# if output_dir does not exist create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

t0 = time.time()

config = json.load(open(f"{CHECKPOINT}/config.json"))
config["n_positions"] = MAX_SEQ_LEN
config = GPT2Config.from_dict(config)
model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda() #
print(f'Loaded model ({time.time()-t0} seconds)')

def create_graph_and_visualization(my_cross_entr, tokens, i, alpha=0.7, model=MODEL_NAME):
    START_IDX = 0

    print(len(tokens))
    tokens_plot = np.array(tokens[1:])
    if tokens_plot[0] == SEPARATOR:
        START_IDX = 3
    
    tokens_plot = tokens_plot[START_IDX:PRINT_UP_TO_IDX]
    curr_ce = my_cross_entr[START_IDX:PRINT_UP_TO_IDX]
    
    # Create an array of indices from 0 to length-1 of `ce`
    indices = tokens_plot[0::3]
    indices = np.cumsum(indices)
    indices = [time for _, time in enumerate(indices) for _ in range(3)]
    max_time = max(indices)
    
    plt.figure(i)
    plt.plot(indices, curr_ce, label=f"{model}_{SEQ_LENS[i]}", color='blue', alpha=alpha)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('cross_entr')
    plt.title('Plot of cross_entr values')

    # Save image files
    plt.savefig(f'{OUTPUT_DIR}/{model}_{DATAFILE}_{SEQ_LENS[i]}.png', dpi=300)

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
    parser.add_argument('-m', '--model_name', 
                        help='model to be evaluated',
                        default=MODEL_NAME)
    parser.add_argument('-n', '--step_number',
                        type=int,
                        help='step number of the model to be evaluated',
                        default=STEP_NUMBER)    
    parser.add_argument('-d', '--datafile',
                        help='datafile to be evaluated',
                        default=DATAFILE)
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    STEP_NUMBER = args.step_number
    CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
    config = json.load(open(f"{CHECKPOINT}/config.json"))
    config["n_positions"] = MAX_SEQ_LEN
    config = GPT2Config.from_dict(config)
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda() #
    print(f'Loaded model ({time.time()-t0} seconds)')

    DATAFILE = args.datafile

    print(f'Calculating log-loss for {args.datafile}')
    print(f'Using model {CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    means = []
    variances = []
    filtered_means = []
    filtered_variances = []
    with open(f'{DATA_DIR}/{DATAFILE}.txt', 'r') as f:
        file_lines = f.readlines()

    use_filter = True
    for j in range(len(SEQ_LENS)):
        SEQ_LEN = SEQ_LENS[j]
        ce = torch.empty(0)
        filtered_ce = torch.empty(0)
        with tqdm(total=len(file_lines), desc=f'Processing seq_len {SEQ_LEN}') as pbar:
            for i,line in tqdm(enumerate(file_lines)):
                num_samples += 1
                if i % SUBSAMPLE != SUBSAMPLE_IDX: continue
                
                tokens = [int(token) for token in line.split()]
                if tokens[0] != AUTOREGRESS:
                    tokens.insert(0, AUTOREGRESS)

                with torch.no_grad():
                    if SEQ_LEN == MAX_SEQ_LEN:
                        full_tokens = torch.tensor(tokens).unsqueeze(0).cuda()
                        logits = model(full_tokens).logits[0]
                        logits = logits.cuda()
                        cross_entr = F.cross_entropy(logits[:-1],full_tokens[0,1:],reduction='none')
                        my_cross_entr = cross_entr.cpu()
                        ce = torch.cat([ce, my_cross_entr])    
                    else:
                        curr_ce = sliding_window_eval(tokens, model, SEQ_LEN)
                        ce = torch.cat([ce, curr_ce])
                pbar.update(SUBSAMPLE)

                # write ce to file
                with open(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_ce_{SEQ_LENS[j]}.ce.txt', 'w') as file:
                    for k in range(len(curr_ce)):
                        file.write(f'{ce[k]}\n')

                create_graph_and_visualization(curr_ce, tokens, j, model=MODEL_NAME)

                # Plot 1: ce
                plot_ce = curr_ce
                if use_filter:
                    # create a kalman filter to smooth the plot
                    n = FILTER_CONST  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
                    plot_ce = lfilter(b, a, curr_ce)
                    
                filtered_ce = torch.cat([filtered_ce, torch.from_numpy(plot_ce)])
                break
                
        means.append(ce.mean()) 
        variances.append(ce.var())
        filtered_means.append(filtered_ce.mean())
        filtered_variances.append(filtered_ce.var())
        print(f'Seq_len {SEQ_LEN}: Mean={ce.mean()}, Variance={ce.var()}, Filtered Mean={filtered_ce.mean()}, Filtered Variance={filtered_ce.var()}')
        

    plt.figure(len(SEQ_LENS) + 2)
    plt.semilogx(SEQ_LENS, variances, marker='o', linestyle='-', label='Variance')
    # Label the axes and title
    plt.xlabel('Sequence length of sliding window')
    plt.ylabel('Cross-entropy')
    plt.xscale('log', base=2)
    plt.title(f'Cross-entropies of sliding windows on model {MODEL_NAME}')
    # show the legend
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_var_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')
    plt.semilogx(SEQ_LENS, means, marker='o', linestyle='-', label='Mean')
    # Save the plot as an image (e.g., PNG, JPEG, or PDF)
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_mean_var_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')

    plt.figure(len(SEQ_LENS) + 3)
    plt.semilogx(SEQ_LENS, filtered_variances, marker='o', linestyle='-', label=f'Variance')
    # Label the axes and title
    plt.xlabel('Sequence length of sliding window')
    plt.ylabel('Cross-entropy')
    plt.xscale('log', base=2)
    plt.title(f'Cross-entropies of sliding windows on model {MODEL_NAME} with filter {FILTER_CONST}')
    # show the legend
    plt.legend()
    # Save the plot as an image (e.g., PNG, JPEG, or PDF)
    plt.savefig(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_var_filter{FILTER_CONST}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')
    plt.semilogx(SEQ_LENS, filtered_means, marker='o', linestyle='-', label=f'Mean')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_mean_var_filter{FILTER_CONST}_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.png', dpi=300, bbox_inches='tight')

    with open(f'{OUTPUT_DIR}/{MODEL_NAME}_{DATAFILE}_mean_variance_{SEQ_LENS[0]}_{MAX_SEQ_LEN}.data.txt', 'w') as file:
        for i in range(len(SEQ_LENS)):
            file.write(f'Seq_len {SEQ_LENS[i]}: Mean={means[i]}, Variance={variances[i]}, Filtered Mean={filtered_means[i]}, Filtered Variance={filtered_variances[i]}\n')
