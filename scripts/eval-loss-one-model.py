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

DATASET = "lakh-data-aar"
MODEL_NAME = "helpful-fire-22"
STEP_NUMBER = 100000
SEQ_LEN = 1024 # must be 1 mod 3
NEW_CE_CUTOFF = 255 # must be less than SEQ_LEN
PRINT_GRAPH = False
PRINT_IDX = 0 # must be less than SUBSAMPLE, leave as 0 for default
PRINT_UP_TO_IDX = 16384 # none if we print the entire example
FILTER_CONST = 100
SUBSAMPLE=100
REMOVE_9999 = False
ADD_9999 = False

DATA = f"/nlp/scr/kathli/datasets/{DATASET}/test.txt" 
#DATAFILE = 'generated-21'
#DATA = f'/nlp/scr/kathli/output/driven-plant-48/{DATAFILE}.txt'
CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
#CHECKPOINT= f"/jagupard31/scr0/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
OUTPUT_DIR = f'/nlp/scr/kathli/output/{MODEL_NAME}'

t0 = time.time()

config = json.load(open(f"{CHECKPOINT}/config.json"))
config["n_positions"] = SEQ_LEN
config = GPT2Config.from_dict(config)
model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda() #
print(f'Loaded model ({time.time()-t0} seconds)')

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
    print(f'Using model {CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    should_print = PRINT_GRAPH
    with open(args.filename, 'r') as f:
        ce = torch.empty(0)
        new_ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != PRINT_IDX: continue
            
            tokens = [int(token) for token in line.split()]
            if REMOVE_9999:
                if tokens[1] == 9999:
                    tokens[1] = 0
            elif ADD_9999:
                if tokens[1] != 9999:
                    tokens[1] = 9999
            if tokens[0] != AUTOREGRESS:
                tokens.insert(0, AUTOREGRESS)

            tokens = torch.tensor(tokens).unsqueeze(0).cuda()

            with torch.no_grad():
                logits = model(tokens).logits[0]
                logits = logits.cuda()
                cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                ce = torch.cat([ce, my_cross_entr])
                new_ce = torch.cat([new_ce, my_cross_entr[-NEW_CE_CUTOFF:]])

                if should_print:
                    print(f'printing example {i} to {OUTPUT_DIR}')
                    # print only the 3rd sample, the first PRINT_UP_TO_IDX tokens
                    if PRINT_UP_TO_IDX is not None:
                        curr_ce = curr_ce[:PRINT_UP_TO_IDX]
                    should_print = False
                    # Create an array of indices from 0 to length-1 of `ce`
                    indices = np.arange(len(curr_ce))

                    # create a kalman filter to smooth the plot
                    n = FILTER_CONST  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
      
                    # Plot 1: ce
                    plt.figure(1)
                    plt.plot(indices, lfilter(b, a, curr_ce))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr')
                    plt.title('Plot of cross_entr values')

                    # Plot 2: ce[0::3]
                    plt.figure(2)
                    plt.plot(indices[0::3], lfilter(b, a, curr_ce[0::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[0::3]')
                    plt.title('Plot of cross_entr[0::3] values')

                    # Plot 3: ce[1::3]
                    plt.figure(3)
                    plt.plot(indices[1::3], lfilter(b, a, curr_ce[1::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[1::3]')
                    plt.title('Plot of cross_entr[1::3] values') 

                    # Plot 4: ce[2::3]
                    plt.figure(4)
                    plt.plot(indices[2::3], lfilter(b, a, curr_ce[2::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[2::3]')
                    plt.title('Plot of cross_entr[2::3] values')

                    # Save all four plots to image files
                    plt.figure(1)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{DATASET}_{i}.png')

                    plt.figure(2)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{DATASET}_{i}_0_3.png')

                    plt.figure(3)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{DATASET}_{i}_1_3.png')

                    plt.figure(4)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{DATASET}_{i}_2_3.png')

                    # save curr_ce to file
                    with open(f'{OUTPUT_DIR}/cross_entr_{DATASET}_{i}.txt', 'w') as f:
                        f.write(' '.join([str(tok) for tok in curr_ce]))

        print('ce', ce)
        print('num samples', num_samples)
        print(num_samples * (SEQ_LEN - 1)) # 1023 
        L = ce.mean()
        print('Tokens processed:', len(ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * (SEQ_LEN - 1) / (560.98*3600)))
        #print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * (SEQ_LEN - 1) / (7827*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*ce.mean()))
            print('  -> onset perplexity: ', exp(ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(ce[1::3].mean()))
            print('  -> note perplexity: ', exp(ce[2::3].mean()))

        print('------------------------------')
        print(f'Calculations on the last {NEW_CE_CUTOFF} tokens')
        L = new_ce.mean()
        print('Tokens processed:', len(new_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * (SEQ_LEN - 1) / (560.98*3600)))
        #print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * (SEQ_LEN - 1) / (7827*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*new_ce.mean()))
            print('  -> onset perplexity: ', exp(new_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(new_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(new_ce[2::3].mean()))
    
