import time

import numpy as np
import torch
import json

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

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 120

MODEL_NAME = 'fh86cy5o'
STEP_NUMBER = 115000

SEQ_LEN = 16384

OUTPUT_DIR = f'/nlp/scr/kathli/output/{MODEL_NAME}'

#model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf').cuda()

CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf"

config = json.load(open(f"{CHECKPOINT}/config.json"))
config["n_positions"] = SEQ_LEN
config = GPT2Config.from_dict(config)
model = GPT2LMHeadModel.from_pretrained(CHECKPOINT, config=config).cuda() #

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0
start = time.time()

TOP_P = 0.99

GET_PROMPT = True
PROMPT_LENGTH=47
PROMPT_FILE = f'/nlp/scr/kathli/output/test/test-3.txt'
if GET_PROMPT:
    with open(PROMPT_FILE, 'r') as f:
        prompt = [int(token) for token in f.read().split()]
    #prompt = prompt[:1116]
    prompt = prompt[447:3000]
    generated_tokens = generate_inter(model, LENGTH_IN_SECONDS, prompt=prompt, top_p=TOP_P, debug=False)
else:
    PROMPT_LENGTH = 0
    #null_prompt = [SEPARATOR, SEPARATOR, SEPARATOR]
    #generated_tokens = generate_inter(model, LENGTH_IN_SECONDS, prompt=null_prompt, top_p=TOP_P, debug=False)
    generated_tokens = generate_inter(model, LENGTH_IN_SECONDS, top_p=TOP_P, debug=False)
print("first 100 tokens", generated_tokens[:100])
print("last 99 tokens", generated_tokens[-99:])
print("num tokens generated", len(generated_tokens))
#if generated_tokens[0] == 9999:
#    generated_tokens[0] = 0
#generated_tokens[0::3] = np.cumsum(generated_tokens[0::3])
#print("new", generated_tokens)
print("min tok", np.min(generated_tokens))
end = time.time()
#ops.print_tokens(generated_tokens)
print("instruments", ops.get_instruments(generated_tokens))
print("time", end - start)
mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
while os.path.exists(f'{OUTPUT_DIR}/generated-{i}.mid'):
    i += 1
mid.save(f'{OUTPUT_DIR}/generated-{i}.mid')
print(f'saved at {OUTPUT_DIR}/generated-{i}.mid')
print(f'top_p {TOP_P}')
visualize(generated_tokens, f'{OUTPUT_DIR}/generated-{i}.png', length=(PROMPT_LENGTH + LENGTH_IN_SECONDS) * TIME_RESOLUTION)

# write generated_tokens to generated-{i}.txt
with open(f'{OUTPUT_DIR}/generated-{i}.txt', 'w') as f:
    f.write(' '.join([str(tok) for tok in generated_tokens]))

if GET_PROMPT:
    prompt_mid = events_to_midi(prompt, debug=False)
    prompt_mid.save(f'{OUTPUT_DIR}/prompt-{i}.mid')
