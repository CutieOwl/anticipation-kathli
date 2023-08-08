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

#DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/train.txt"
#DATA = "/home/kml11/lakh-data-inter-16384/test.txt"
DATA = "/nlp/scr/kathli/datasets/lakh-data-inter-16384/test.txt" #
#DATA = "/nlp/scr/kathli/output/test/test-16-17.txt"
#DATA = "/nlp/scr/jthickstun/anticipation/datasets/interarrival/test.txt"

#DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/maestro-test.txt"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/genial-firefly-238/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/efficient-sun-259/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/still-night-260/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/amber-yogurt-821/step-100000/hf"
#CHECKPOINT= "/nlp/scr/kathli/checkpoints/exalted-grass-86/step-10000/hf"
#CHECKPOINT = "/nlp/scr/jthickstun/anticipation/checkpoints/efficient-sun-259/step-100000/hf"
#CHECKPOINT = "/home/kml11/driven-plant-48/hf"
#CHECKPOINT= "/nlp/scr/kathli/checkpoints/driven-plant-48/step-30000/hf" #
CHECKPOINT= "/nlp/scr/kathli/checkpoints/driven-plant-48/step-30000/hf"
#CHECKPOINT= "/nlp/scr/kathli/checkpoints/olive-universe-204/step-60000/hf"
#CHECKPOINT= "/afs/cs.stanford.edu/u/kathli/repos/levanter-midi/checkpoints/upbeat-deluge-127/step-10000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/dashing-salad-267/step-100000/hf"
#CHECKPOINT = "/nlp/scr/jthickstun/anticipation/checkpoints/dainty-elevator-270/step-200000/hf"

OUTPUT_DIR = f'/nlp/scr/kathli/output/driven-plant-48'

SUBSAMPLE=100

t0 = time.time()

config = json.load(open(f"{CHECKPOINT}/config.json"))
config["n_positions"] = 1024 #16384
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
    printed = True
    with open(args.filename, 'r') as f:
        ce = torch.empty(0)
        new_ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != 3: continue
            tokens = [int(token) for token in line.split()]
            tokens.insert(0, AUTOREGRESS)
            #print(tokens)
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()

            with torch.no_grad():
                logits = model(tokens).logits[0]
                #print("logits idx 1", F.softmax(logits[1],dim=0))
                #print("logits device", logits.get_device())
                #print("tokens device", tokens.get_device())
                logits = logits.cuda()
                cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
                my_cross_entr = cross_entr.cpu()
                ce = torch.cat([ce, my_cross_entr])
                new_ce = torch.cat([new_ce, my_cross_entr[-2382:]])

                if i == 3:
                    print('printing example')
                    # print only the 3rd sample, the first 8652 tokens
                    my_cross_entr = my_cross_entr[:8652]
                    printed = True
                    # Create an array of indices from 0 to length-1 of `ce`
                    indices = np.arange(len(my_cross_entr))

                    # create a kalman filter to smooth the plot
                    n = 30  # the larger n is, the smoother curve will be
                    b = [1.0 / n] * n
                    a = 1
      
                    # Plot 1: ce
                    plt.figure(1)
                    plt.plot(indices, lfilter(b, a, my_cross_entr))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr')
                    plt.title('Plot of cross_entr values')

                    # Plot 2: ce[0::3]
                    plt.figure(2)
                    plt.plot(indices[0::3], lfilter(b, a, my_cross_entr[0::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[0::3]')
                    plt.title('Plot of cross_entr[0::3] values')

                    # Plot 3: ce[1::3]
                    plt.figure(3)
                    plt.plot(indices[1::3], lfilter(b, a, my_cross_entr[1::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[1::3]')
                    plt.title('Plot of cross_entr[1::3] values') 

                    # Plot 4: ce[2::3]
                    plt.figure(4)
                    plt.plot(indices[2::3], lfilter(b, a, my_cross_entr[2::3]))
                    plt.xlabel('Index')
                    plt.ylabel('cross_entr[2::3]')
                    plt.title('Plot of cross_entr[2::3] values')

                    # Save all four plots to image files
                    plt.figure(1)
                    #plt.savefig(f'{OUTPUT_DIR}/cross_entr_16-17_plot.png')
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_plot.png')

                    plt.figure(2)
                    #plt.savefig(f'{OUTPUT_DIR}/cross_entr_16-17_0_3_plot.png')
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_0_3_plot.png')

                    plt.figure(3)
                    #plt.savefig(f'{OUTPUT_DIR}/cross_entr_16-17_1_3_plot.png')
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_1_3_plot.png')

                    plt.figure(4)
                    #plt.savefig(f'{OUTPUT_DIR}/cross_entr_16-17_2_3_plot.png')
                    plt.savefig(f'{OUTPUT_DIR}/cross_entr_{i}_2_3_plot.png')
                    break

        print('ce', ce)
        print('num samples', num_samples)
        print(num_samples * 16383) # 1023 
        L = ce.mean()
        print('Tokens processed:', len(ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * 16383 / (560.98*3600)))
        #print('  -> bits per second: ', L*1.442695*(125050497 / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*ce.mean()))
            print('  -> onset perplexity: ', exp(ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(ce[1::3].mean()))
            print('  -> note perplexity: ', exp(ce[2::3].mean()))

        print('------------------------------')
        print('new caluculations')
        L = new_ce.mean()
        print('Tokens processed:', len(new_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> bits per second: ', L*np.log2(np.e)*(num_samples * 16383 / (560.98*3600)))
        #print('  -> bits per second: ', SUBSAMPLE*L*np.log2(np.e)*(len(new_ce) * 2 / (560.98*3600)))
        #print('  -> bits per second: ', L*1.442695*(125050497 / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*new_ce.mean()))
            print('  -> onset perplexity: ', exp(new_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(new_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(new_ce[2::3].mean()))
    
