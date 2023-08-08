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
SHORTF_MODEL_NAME = "fiery-vortex-95"
SHORTF_STEP_NUMBER = 15000
SHORTS_MODEL_NAME = "toasty-silence-91"
SHORTS_STEP_NUMBER = 115000
LONG_MODEL_NAME = "driven-plant-48"
LONG_STEP_NUMBER = 30000
LONG_SEQ_LEN = 16384 # must be 1 mod 3
SHORT_SEQ_LEN = 1024 # must be 1 mod 3
NEW_CE_CUTOFF = 255 # must be less than SEQ_LEN
PRINT_GRAPH = True
PRINT_IDX = 0 # must be less than SUBSAMPLE, leave as 0 for default
PRINT_UP_TO_IDX = 16384 #8652 # none if we print the entire example
ADD_9999 = False
REMOVE_9999 = False
FILTER_CONST = 90
SUBSAMPLE=100
OUTPUT_DIR = f'/nlp/scr/kathli/eval/rep_struct/figs'

SHORTF_MODEL_LABEL = "finetune short"
SHORTS_MODEL_LABEL = "single-stage short" #SHORT_MODEL_NAME
LONG_MODEL_LABEL = "finetune long" # LONG_MODEL_NAME

DATAFILE = 'generated-20'
DATA = f'/nlp/scr/kathli/output/driven-plant-48/{DATAFILE}.txt' #f'/nlp/scr/kathli/eval/rep_struct/{DATAFILE}.txt'  # f"/nlp/scr/kathli/datasets/{DATASET}/test.txt" 
SHORTF_CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{SHORTF_MODEL_NAME}/step-{SHORTF_STEP_NUMBER}/hf"
SHORTS_CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{SHORTS_MODEL_NAME}/step-{SHORTS_STEP_NUMBER}/hf"
LONG_CHECKPOINT = f"/nlp/scr/kathli/checkpoints/{LONG_MODEL_NAME}/step-{LONG_STEP_NUMBER}/hf"

t0 = time.time()

shortf_config = json.load(open(f"{SHORTF_CHECKPOINT}/config.json"))
shortf_config["n_positions"] = SHORT_SEQ_LEN
shortf_config = GPT2Config.from_dict(shortf_config)
shortf_model = GPT2LMHeadModel.from_pretrained(SHORTF_CHECKPOINT, config=shortf_config).cuda() \

shorts_config = json.load(open(f"{SHORTS_CHECKPOINT}/config.json"))
shorts_config["n_positions"] = SHORT_SEQ_LEN
shorts_config = GPT2Config.from_dict(shorts_config)
shorts_model = GPT2LMHeadModel.from_pretrained(SHORTS_CHECKPOINT, config=shorts_config).cuda() 
print(f'Loaded short models ({time.time()-t0} seconds)')
t1 = time.time()
long_config = json.load(open(f"{LONG_CHECKPOINT}/config.json"))
long_config["n_positions"] = LONG_SEQ_LEN
long_config = GPT2Config.from_dict(long_config)
long_model = GPT2LMHeadModel.from_pretrained(LONG_CHECKPOINT, config=long_config).cuda() 
print(f'Loaded long model ({time.time()-t1} seconds)')

def sliding_window_eval(tokens, short_model):
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
    return curr_ce

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
    print(f'Using short finetune model {SHORTF_CHECKPOINT}')
    print(f'Using short singlestage model {SHORTS_CHECKPOINT}')
    print(f'Using long model {LONG_CHECKPOINT}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    should_print = PRINT_GRAPH
    with open(args.filename, 'r') as f:
        shortf_ce = torch.empty(0)
        shorts_ce = torch.empty(0)
        long_ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != PRINT_IDX: continue
            
            tokens = [int(token) for token in line.split()]
            if tokens[0] != AUTOREGRESS:
                print("inserting AUTOREGRESS")
                tokens.insert(0, AUTOREGRESS)

            with torch.no_grad():
                curr_ce_f = sliding_window_eval(tokens, shortf_model)
                shortf_ce = torch.cat([shortf_ce, curr_ce_f])

                curr_ce_s = sliding_window_eval(tokens, shorts_model)
                shorts_ce = torch.cat([shorts_ce, curr_ce_s])

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
                    curr_ce_f = curr_ce_f[:PRINT_UP_TO_IDX]
                    curr_ce_s = curr_ce_s[:PRINT_UP_TO_IDX]
                    my_cross_entr = my_cross_entr[:PRINT_UP_TO_IDX]
                    should_print = False
                    # Create an array of indices from 0 to length-1 of `ce`
                    indices = np.arange(len(my_cross_entr))

                    # create a kalman filter to smooth the plot
                    n = FILTER_CONST  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
      
                    # Plot 1: ce
                    plt.figure(1)
                    plt.plot(indices, lfilter(b, a, curr_ce_f), label=SHORTF_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.plot(indices, lfilter(b, a, curr_ce_s), label=SHORTS_MODEL_LABEL, color='green', alpha=0.7)
                    plt.plot(indices, lfilter(b, a, my_cross_entr), label=LONG_MODEL_LABEL, color='red', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr')
                    plt.title('Plot of cross_entr values')

                    # Plot 2: ce[0::3]
                    plt.figure(2)
                    plt.plot(indices[0::3], lfilter(b, a, curr_ce_f[0::3]), label=SHORTF_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.plot(indices[0::3], lfilter(b, a, curr_ce_s[0::3]), label=SHORTS_MODEL_LABEL, color='green', alpha=0.7)
                    plt.plot(indices[0::3], lfilter(b, a, my_cross_entr[0::3]), label=LONG_MODEL_LABEL, color='red', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[0::3]')
                    plt.title('Plot of cross_entr[0::3] values')

                    # Plot 3: ce[1::3]
                    plt.figure(3)
                    plt.plot(indices[1::3], lfilter(b, a, curr_ce_f[1::3]), label=SHORTF_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.plot(indices[1::3], lfilter(b, a, curr_ce_s[1::3]), label=SHORTS_MODEL_LABEL, color='green', alpha=0.7)
                    plt.plot(indices[1::3], lfilter(b, a, my_cross_entr[1::3]), label=LONG_MODEL_LABEL, color='red', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[1::3]')
                    plt.title('Plot of cross_entr[1::3] values') 

                    # Plot 4: ce[2::3]
                    plt.figure(4)
                    plt.plot(indices[2::3], lfilter(b, a, curr_ce_f[2::3]), label=SHORTF_MODEL_LABEL, color='blue', alpha=0.7)
                    plt.plot(indices[2::3], lfilter(b, a, curr_ce_s[2::3]), label=SHORTS_MODEL_LABEL, color='green', alpha=0.7)
                    plt.plot(indices[2::3], lfilter(b, a, my_cross_entr[2::3]), label=LONG_MODEL_LABEL, color='red', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[2::3]')
                    plt.title('Plot of cross_entr[2::3] values')

                    # Save all four plots to image files
                    plt.figure(1)
                    plt.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{SHORTF_MODEL_NAME}_{SHORTS_MODEL_NAME}_{LONG_MODEL_NAME}_{i}.png', dpi=300)

                    plt.figure(2)
                    plt.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{SHORTF_MODEL_NAME}_{SHORTS_MODEL_NAME}_{LONG_MODEL_NAME}_{i}_0_3.png', dpi=300)

                    plt.figure(3)
                    plt.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{SHORTF_MODEL_NAME}_{SHORTS_MODEL_NAME}_{LONG_MODEL_NAME}_{i}_1_3.png', dpi=300)

                    plt.figure(4)
                    plt.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{SHORTF_MODEL_NAME}_{SHORTS_MODEL_NAME}_{LONG_MODEL_NAME}_{i}_2_3.png', dpi=300)
                    #break

        print('num samples', num_samples)
        print(num_samples * (LONG_SEQ_LEN - 1))

        print(f"SHORTF MODEL {SHORTF_MODEL_NAME}")
        #print('ce', shortf_ce)
        short_L = shortf_ce.mean()
        print('Tokens processed:', len(shortf_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', short_L)
        print('  -> bits per second: ', short_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*shortf_ce.mean()))
            print('  -> onset perplexity: ', exp(shortf_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(shortf_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(shortf_ce[2::3].mean()))


        print(f"SHORTS MODEL {SHORTS_MODEL_NAME}")
        #print('ce', shorts_ce)
        short_L = shorts_ce.mean()
        print('Tokens processed:', len(shorts_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', short_L)
        print('  -> bits per second: ', short_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*shorts_ce.mean()))
            print('  -> onset perplexity: ', exp(shorts_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(shorts_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(shorts_ce[2::3].mean()))

        print(f"LONG MODEL {LONG_MODEL_NAME}")
        #print('ce', long_ce)
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

