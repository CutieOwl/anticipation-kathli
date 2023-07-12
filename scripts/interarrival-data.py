import os
import numpy as np

from tqdm import tqdm

from anticipation.config import *
from anticipation.vocab import *
from anticipation.ops import print_tokens

DATA_DIR = "/nlp/scr/kathli/datasets/lakh-data-auto"
OUTPUT_DIR = "/nlp/scr/kathli/datasets/lakh-data-inter"

FILE_NAME = "train.txt"

j = 0

UNK = 9999 # special value for unknown arrival time

def find_element_index(arr, element):
    indices = np.where(arr == element)[0]
    if indices.size > 0:
        return indices[0]  # Return the first occurrence of the element
    else:
        return -1 # Return -1 if element does not exist in arr

def set_time_diffs(control_tokens, begin, sep):
    time_tokens = control_tokens[begin:sep:3]
    if len(time_tokens) >= 2:
        diffs = time_tokens[1:] - time_tokens[:-1]
        time_tokens[1:] = diffs
    if len(time_tokens) >= 1:
        if begin == 1:
            time_tokens[0] = UNK
        else:
            time_tokens[0] = time_tokens[0]
        control_tokens[begin:sep:3] = time_tokens

# read data from DATA/FILE_NAME and write data with arrival times
# converted to interarrival times to OUTPUT_DIR/FILE_NAME
with open(os.path.join(OUTPUT_DIR, FILE_NAME), 'w') as o_f:
    with open(os.path.join(DATA_DIR, FILE_NAME), 'r') as i_f:
        for i,line in tqdm(list(enumerate(i_f))):
            control_tokens = np.array([int(token) for token in line.split()])
            begin = 1
            sep = find_element_index(control_tokens[begin:], SEPARATOR)
            while sep != -1:
                sep += begin
                set_time_diffs(control_tokens, begin, sep)
                begin = sep + 3
                sep = find_element_index(control_tokens[begin:], SEPARATOR)
            # last section after SEP
            sep = len(control_tokens)
            set_time_diffs(control_tokens, begin, sep)

            o_f.write(' '.join([str(token) for token in control_tokens]) + '\n')





'''
            tokens_without_control = tokens[1:]
            note_tokens = tokens_without_control[3:]
            while SEPARATOR in note_tokens:
            #print("SEPARATOR found at ", note_tokens.index(SEPARATOR) + 3)
            sep_index = note_tokens.index(SEPARATOR)
            note_tokens = note_tokens[sep_index+3:]
            #print_tokens(tokens_without_control)
            #j += 1
            time_tokens = note_tokens[0::3]
            dur_tokens = note_tokens[1::3]
            max_dur = max(dur_tokens)
            if max_dur >= 200:
                print("dur was greater than 200, index", dur_tokens.index(max_dur) + 1)
                print_tokens(tokens_without_control)
                j += 1
            #token_diffs = np.array(time_tokens[1:]) - np.array(time_tokens[:-1])
            #if len(token_diffs) <= 1:
                #print("empty sequence i guess")
                #print_tokens(tokens_without_control)
            #    continue
            #max_diff = np.argmax(token_diffs)
            #if (token_diffs[max_diff] > 100):
            #    print("diff was greater than 100, index", max_diff + 1)
            #    print_tokens(tokens_without_control)
            #    j += 1
            #if SEPARATOR in note_tokens:
            #    print("SEPARATOR found at ", note_tokens.index(SEPARATOR) + 3)
            #    print_tokens(tokens_without_control)
            #    j += 1
            #if REST in note_tokens:
            #    print("REST found at ", note_tokens.index(REST) + 3)
            #    print_tokens(tokens_without_control)
            #    j += 1
            if j > 20:
                print("20 examples found by index ", i)
                break
'''

