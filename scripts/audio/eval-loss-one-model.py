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

MODEL_NAME = "jqv0lgv4"
STEP_NUMBER = 99481 #97645
SEQ_LEN = 8192 # must be 1 mod 3
PRINT_GRAPH = True
PRINT_IDX = 0 # must be less than SUBSAMPLE, leave as 0 for default
USE_FILTER = False
PRINT_UP_TO_IDX = 720
FILTER_CONST = 100
SUBSAMPLE=1000

DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt"
#DATAFILE = 'generated-21'
#DATA = f'/nlp/scr/kathli/output/driven-plant-48/{DATAFILE}.txt'
CHECKPOINT= f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf'
#CHECKPOINT= f"/jagupard31/scr0/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"
OUTPUT_DIR = f'/nlp/scr/kathli/output/audio/{MODEL_NAME}'

t0 = time.time()

config = json.load(open(f"{CHECKPOINT}/config.json"))
print("config", config)
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
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != PRINT_IDX: continue
            
            tokens = [int(token) for token in line.split()]

            tokens = torch.tensor(tokens).unsqueeze(0).cuda()

            with torch.no_grad():
                logits = model(tokens).logits[0]
                logits = logits.cuda()
                cross_entr = F.cross_entropy(logits[:-4],tokens[0,4:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                ce = torch.cat([ce, my_cross_entr])

                if should_print:
                    print(f'printing example {i} to {OUTPUT_DIR}')
                    # print only the 3rd sample, the first PRINT_UP_TO_IDX tokens
                    should_print = False

                    curr_ce = my_cross_entr[:PRINT_UP_TO_IDX]
                    # Create an array of indices from 0 to length-1 of `ce`
                    indices = np.arange(len(curr_ce))

                    # create a kalman filter to smooth the plot
                    n = FILTER_CONST  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
      
                    # Plot 1: ce
                    plt.figure(1)
                    if USE_FILTER:
                        plt.plot(indices, lfilter(b, a, curr_ce))
                    else:
                        plt.plot(indices, curr_ce)
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr')
                    plt.title('Plot of cross_entr values')

                    # Plot 2: ce[0::4]
                    plt.figure(2)
                    if USE_FILTER:
                        plt.plot(indices[0::4], lfilter(b, a, curr_ce[0::4]))
                    else:
                        plt.plot(indices[0::4], curr_ce[0::4])
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[0::4]')
                    plt.title('Plot of cross_entr[0::4] values')

                    # Plot 3: ce[1::4]
                    plt.figure(3)
                    if USE_FILTER:
                        plt.plot(indices[1::4], lfilter(b, a, curr_ce[1::4]))
                    else:
                        plt.plot(indices[1::4], curr_ce[1::4])
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[1::4]')
                    plt.title('Plot of cross_entr[1::4] values') 

                    # Plot 4: ce[2::4]
                    plt.figure(4)
                    if USE_FILTER:
                        plt.plot(indices[2::4], lfilter(b, a, curr_ce[2::4]))
                    else:
                        plt.plot(indices[2::4], curr_ce[2::4])
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[2::4]')
                    plt.title('Plot of cross_entr[2::4] values')

                    # Plot 5: ce[3::4]
                    plt.figure(5)
                    if USE_FILTER:
                        plt.plot(indices[3::4], lfilter(b, a, curr_ce[3::4]))
                    else:
                        plt.plot(indices[3::4], curr_ce[3::4])
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[3::4]')
                    plt.title('Plot of cross_entr[3::4] values')

                    # Save all four plots to image files
                    plt.figure(1)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}.png')

                    plt.figure(2)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_0.png')

                    plt.figure(3)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_1.png')

                    plt.figure(4)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_2.png')

                    plt.figure(5)
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_3.png')

                    # save curr_ce to file
                    with open(f'{OUTPUT_DIR}/cross_entr_{i}.txt', 'w') as f:
                        f.write(' '.join([str(tok) for tok in curr_ce]))

        print('ce', ce)
        print('num samples', num_samples)
        print(num_samples * (SEQ_LEN - 4)) # 1023 
        L = ce.mean()
        V = ce.var()
        print('Tokens processed:', len(ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> log loss variance: ', V)
        print('  -> 0 perplexity: ', exp(ce[0::4].mean()))
        print('  -> 1 perplexity: ', exp(ce[1::4].mean()))
        print('  -> 2 perplexity: ', exp(ce[2::4].mean()))
        print('  -> 3 perplexity: ', exp(ce[3::4].mean()))

        print('  -> 0 mean: ', ce[0::4].mean())
        print('  -> 1 mean: ', ce[1::4].mean())
        print('  -> 2 mean: ', ce[2::4].mean())
        print('  -> 3 mean: ', ce[3::4].mean())

        print('  -> 0 var: ', ce[0::4].var())
        print('  -> 1 var: ', ce[1::4].var())
        print('  -> 2 var: ', ce[2::4].var())
        print('  -> 3 var: ', ce[3::4].var())

        print('  -> 0 median: ', ce[0::4].median())
        print('  -> 1 median: ', ce[1::4].median())
        print('  -> 2 median: ', ce[2::4].median())
        print('  -> 3 median: ', ce[3::4].median())


