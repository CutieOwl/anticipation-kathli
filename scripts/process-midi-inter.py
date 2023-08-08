'''
This script truncates a midi to 16384 tokens and then visualizes it as a piano roll.
You need a processed mid.compound.txt file to run this script.
'''

import os
from argparse import ArgumentParser

from anticipation.config import *
from anticipation.tokenize_inter import tokenize_single_inter
from anticipation.convert import events_to_midi

from anticipation.vocab import AUTOREGRESS, ANTICIPATE
from anticipation.vocab import SEPARATOR

from anticipation.convert import midi_to_compound

from anticipation.visuals import visualize

FILENAME = 'f0b84c86e6d047b8f2b990e4e048ace4'
SHORT_FILENAME = FILENAME[:8]

FILE_DIR = '/nlp/scr/kathli/eval/rep_struct'
FILE = f'{FILE_DIR}/{FILENAME}.mid.compound.txt'

if not os.path.exists(FILE):
    MIDI_FILE = f'{FILE_DIR}/{FILENAME}.mid'
    tokens = midi_to_compound(MIDI_FILE, debug=True)

    with open(FILE, 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))

OUTPUT_FILE = f'/{FILE_DIR}/{SHORT_FILENAME}.txt'
AUGMENT = 1

results = tokenize_single_inter(FILE, OUTPUT_FILE, AUGMENT, 0)

with open(OUTPUT_FILE, 'r') as f:
    line = f.readline()
    tokens = [int(token) for token in line.split()]
    if tokens[0] in [AUTOREGRESS, ANTICIPATE]:
        tokens = tokens[1:] # strip control codes

    mid = events_to_midi(tokens, debug=True)
    mid.save(f'{FILE_DIR}/{SHORT_FILENAME}.mid')
    print(f'{SHORT_FILENAME} Tokenized MIDI Length: {mid.length} seconds ({len(tokens)} tokens)')

    visualize(tokens, f'{FILE_DIR}/{SHORT_FILENAME}.png', length=int(mid.length))
    
