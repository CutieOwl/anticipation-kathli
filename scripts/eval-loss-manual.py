import time

from statistics import mean
from math import exp

import torch
import torch.nn.functional as F

import json

import sys
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from argparse import ArgumentParser
from tqdm import tqdm

from anticipation import ops
from anticipation.config import M, EVENT_SIZE
from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION
from anticipation.ops import max_time

#DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/train.txt"
DATA = "/nlp/scr/kathli/datasets/lakh-data-inter/test.txt"
#DATA = "/nlp/scr/jthickstun/anticipation/datasets/interarrival/test.txt"

#DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/maestro-test.txt"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/genial-firefly-238/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/efficient-sun-259/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/still-night-260/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/amber-yogurt-821/step-100000/hf"
#CHECKPOINT= "/nlp/scr/kathli/checkpoints/exalted-grass-86/step-10000/hf"
#CHECKPOINT = "/nlp/scr/jthickstun/anticipation/checkpoints/efficient-sun-259/step-100000/hf"
CHECKPOINT= "/nlp/scr/kathli/checkpoints/rare-cherry-24/step-100000/hf"
#CHECKPOINT= "/nlp/scr/kathli/checkpoints/olive-universe-204/step-60000/hf"
#CHECKPOINT= "/afs/cs.stanford.edu/u/kathli/repos/levanter-midi/checkpoints/upbeat-deluge-127/step-10000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/dashing-salad-267/step-100000/hf"
#CHECKPOINT = "/nlp/scr/jthickstun/anticipation/checkpoints/dainty-elevator-270/step-200000/hf"

SUBSAMPLE=100

t0 = time.time()

config = json.load(open(f"{CHECKPOINT}/config.json"))
config["n_positions"] = 1024
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
    with open(args.filename, 'r') as f:
        ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(f))):
            if i % SUBSAMPLE != 0: continue

            tokens = [int(token) for token in line.split()]
            #print(tokens)
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()
            
            with torch.no_grad():
               logits = model(tokens).logits[0]
               #print("logits idx 1", F.softmax(logits[1],dim=0))
               #print("logits device", logits.get_device())
               #print("tokens device", tokens.get_device())
               logits = logits.cuda()
               cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
               ce = torch.cat([ce, cross_entr.cpu()])

        L = ce.mean()
        print('Tokens processed:', len(ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', L)
        print('  -> bits per second: ', L*1.442695*(125050497 / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*ce.mean()))
            print('  -> onset perplexity: ', exp(ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(ce[1::3].mean()))
            print('  -> note perplexity: ', exp(ce[2::3].mean()))
