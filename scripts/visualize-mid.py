import time

import numpy as np
import torch

import sys
import os
import time
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation import ops
from anticipation.sample_inter import generate_inter
from anticipation.convert import events_to_midi
from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *

from anticipation.vocab import AUTOREGRESS, ANTICIPATE
from anticipation.vocab import SEPARATOR

# read in tokens from OUTPUT_DIR/generated-10.txt
#with open(f'{OUTPUT_DIR}/generated-10.txt', 'r') as f:
#    tokens = np.array([int(tok) for tok in f.read().split()])

# call visualize(tokens, OUTPUT_DIR/generated-10.png)
#visualize(tokens, f'{OUTPUT_DIR}/generated-10.png')

FILE = 'generated-16'
FILE_DIR = '/nlp/scr/kathli/output/driven-plant-48'
#FILE = '/nlp/scr/kathli/datasets/lakh-data-inter-16384/test-16-17.txt'
OUTPUT_DIR = '/nlp/scr/kathli/output/driven-plant-48/'
INDEX = 0

INDEX1 = 16
INDEX2 = 17

ret = []

with open(f'{FILE_DIR}/{FILE}.txt', 'r') as f:
    for i, line in enumerate(f):
        if i < INDEX:
            continue

        if i > INDEX: 
            break

        tokens = [int(token) for token in line.split()]
        if tokens[0] in [AUTOREGRESS, ANTICIPATE]:
            tokens = tokens[1:] # strip control codes

        '''
        # concatenate tokens with ret
        if i == INDEX1:
            ret.extend(tokens[14985:])
        else:
            ret.extend(tokens[:3366])
        '''

        #visualize(tokens, f'{OUTPUT_DIR}/test-.png')
        
        mid = events_to_midi(tokens, debug=True)

        #mid.save(f'{OUTPUT_DIR}/test-{i}.mid')
        visualize(tokens, f'{FILE_DIR}/{FILE}.png', length=(int(mid.length)) * TIME_RESOLUTION)
        print(f'{i} Tokenized MIDI Length: {mid.length} seconds ({len(tokens)} tokens)')

        # save tokens[:8670] to OUTPUT_DIR/test-INDEX.txt
        #with open(f'{OUTPUT_DIR}/test-{i}.txt', 'w') as f:
        #    f.write(' '.join([str(tok) for tok in tokens[:8670]]))

'''
mid = events_to_midi(ret, debug=True)
mid.save(f'{OUTPUT_DIR}/test-{INDEX1}:{INDEX2}.mid')
print(f'{i} Tokenized MIDI Length: {mid.length} seconds ({len(ret)} tokens)')
visualize(ret, f'{OUTPUT_DIR}/test-{INDEX1}:{INDEX2}.png')
with open(f'{OUTPUT_DIR}/test-{INDEX1}:{INDEX2}.txt', 'w') as f:
    f.write(' '.join([str(tok) for tok in ret]))
'''