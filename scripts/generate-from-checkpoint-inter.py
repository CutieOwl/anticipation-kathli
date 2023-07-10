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

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 30

MODEL_NAME = 'rare-cherry-24'
STEP_NUMBER = 100000

OUTPUT_DIR = f'/nlp/scr/kathli/output/{MODEL_NAME}'

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf').cuda()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0
start = time.time()
generated_tokens = generate_inter(model, LENGTH_IN_SECONDS, top_p=0.98, debug=False)
print("org", generated_tokens)
if generated_tokens[0] == 9999:
    generated_tokens[0] = 0
generated_tokens[0::3] = np.cumsum(generated_tokens[0::3])
print("new", generated_tokens)
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
#visualize(generated_tokens, f'{OUTPUT_DIR}/generated-{i}.png')


