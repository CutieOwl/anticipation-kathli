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

from scipy.signal import lfilter
import matplotlib.pyplot as plt

# MIDI models
MODEL = "incneoi4"#"multi-head"# "ha05xrd3" # "54labs45" #" #"9qbavecu"
STEP_NUM = 99758 #99802 #98517 #99920 #50000 #99588 # ##99698  #42430

# AUDIO models
#MODEL = "54labs45" # "vl5058w4" #"9qbavecu"
#STEP_NUM = 99920 #50000 #99588 #42430

#DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt" # this is the old audio tokenization dataset

# don't have a local test set so eval on validation
#DATA = "/juice4/scr4/nlp/music/datasets/encodec_fma.audiogen.valid.txt"

DATA = "/juice4/scr4/nlp/music/temp_test/encodec_fma.audiogen.valid-small.txt"
#DATA = "/juice4/scr4/nlp/music/temp_test/lakh.midigen.test.txt" # this is the clean midi dataset
#DATA_2 = "/juice4/scr4/nlp/music/temp_test/encodec_fma.trans_midigen.test.txt" # this is the transcribed midi dataset

# save the generated sequence to a file
OUTPUT_DIR = f'/nlp/scr/kathli/output/mm/{MODEL}'

SUBSAMPLE = 100
SUBSAMPLE_IDX = 0
NEW_CE_CUTOFF = 255
FILTER_CONST = 50

# initialize the model and tokenizer
model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf/'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'
#model_name = '/juice4/scr4/nlp/music/prelim-checkpoints/skewed/{MODEL}/step-{STEP_NUM}/hf/'
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
    avg_ce = None
    should_print = True
    num_avg = 0

    # create a kalman filter to smooth the plot
    n = FILTER_CONST  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    with open(datafile, 'r') as f:    
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != SUBSAMPLE_IDX: continue
            num_avg += 1
            tokens = [int(token) for token in line.split()]
            #print(tokens)
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()

            with torch.no_grad():
                logits = model(tokens).logits[0].cuda()
                #print(logits.shape)
                #print(tokens.shape)
                cross_entr = F.cross_entropy(logits[:-4],tokens[0,4:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                ce = torch.cat([ce, my_cross_entr])
                new_ce = torch.cat([new_ce, my_cross_entr[-NEW_CE_CUTOFF:]])
                if avg_ce is None:
                    avg_ce = my_cross_entr  
                else:
                    avg_ce += my_cross_entr

            if should_print:
                should_print = False
                curr_ce = my_cross_entr
                # Create an array of indices from 0 to length-1 of `ce`
                indices = np.arange(len(curr_ce))
    
                # Plot 1: ce
                plt.figure(1)
                plt.plot(indices, lfilter(b, a, curr_ce))
                plt.xlabel('Index')
                plt.ylabel('cross_entr')
                plt.title('Plot of cross_entr values')

                # Save all four plots to image files
                plt.figure(1)
                plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}.png')
                print('Saved cross_entr plot to image file', f'{OUTPUT_DIR}/cross_entr_{i}.png')

    avg_ce = avg_ce / num_avg
    indices = np.arange(len(avg_ce))
    plt.figure(2)
    plt.plot(indices[:100], avg_ce[:100])
    plt.xlabel('Index')
    plt.ylabel('cross_entr')
    plt.title('Plot of cross_entr values')
    plt.figure(2)
    plt.savefig(f'{OUTPUT_DIR}/cross_entr_avg_nofilt_100.png')
    print('Saved cross_entr plot to image file', f'{OUTPUT_DIR}/cross_entr_avg_nofilt_100.png')

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