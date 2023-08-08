import time

import numpy as np
import torch

import sys
import os
import time
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation import ops
from anticipation.sample_inter import generate_inter_eval
from anticipation.convert import events_to_midi
from anticipation.visuals import visualize

from anticipation.vocab import SEPARATOR

import matplotlib.pyplot as plt

from scipy.signal import lfilter

LENGTH_IN_SECONDS = 120

MODEL_NAME = 'driven-plant-48'
STEP_NUMBER = 30000

FILTER_CONST = 100

OUTPUT_DIR = f'/nlp/scr/kathli/output/{MODEL_NAME}/eval'

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/kathli/checkpoints/{MODEL_NAME}/step-{STEP_NUMBER}/hf').cuda()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0
start = time.time()

TOP_P = 1.0 #0.99

GET_PROMPT = False
PROMPT_FILE = f'/nlp/scr/kathli/output/test/test-3.txt'
if GET_PROMPT:
    with open(PROMPT_FILE, 'r') as f:
        prompt = [int(token) for token in f.read().split()]
    #prompt = prompt[:1116]
    prompt = prompt[447:3000]
    generated_tokens, max_probs = generate_inter_eval(model, LENGTH_IN_SECONDS, prompt=prompt, top_p=TOP_P, debug=False)
else:
    generated_tokens, max_probs = generate_inter_eval(model, LENGTH_IN_SECONDS, top_p=TOP_P, debug=False)
print("first 100 tokens", generated_tokens[:100])
print("last 99 tokens", generated_tokens[-99:])
print("num tokens generated", len(generated_tokens))
print("last max_probs", max_probs[-99:])

while os.path.exists(f'{OUTPUT_DIR}/generated-{i}.mid'):
    i += 1

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
mid.save(f'{OUTPUT_DIR}/generated-{i}.mid')
print(f'saved at {OUTPUT_DIR}/generated-{i}.mid')

# Create an array of indices from 0 to length-1 of `ce`
indices = np.arange(len(max_probs))

# create a kalman filter to smooth the plot
n = FILTER_CONST  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1


# Plot 1: max_probs
plt.figure(1)
plt.plot(indices, lfilter(b, a, max_probs))
plt.xlabel('Index')
plt.ylabel('cross_entr')
plt.title('Plot of cross_entr values')

# Plot 2: max_probs[0::3]
plt.figure(2)
plt.plot(indices[0::3], lfilter(b, a, max_probs[0::3]))
plt.xlabel('Index')
plt.ylabel('cross_entr[0::3]')
plt.title('Plot of cross_entr[0::3] values')

# Plot 3: max_probs[1::3]
plt.figure(3)
plt.plot(indices[1::3], lfilter(b, a, max_probs[1::3]))
plt.xlabel('Index')
plt.ylabel('cross_entr[1::3]')
plt.title('Plot of cross_entr[1::3] values') 

# Plot 4: max_probs[2::3]
plt.figure(4)
plt.plot(indices[2::3], lfilter(b, a, max_probs[2::3]))
plt.xlabel('Index')
plt.ylabel('cross_entr[2::3]')
plt.title('Plot of cross_entr[2::3] values')

# Save all four plots to image files
plt.figure(1)
plt.savefig(f'{OUTPUT_DIR}/generated-{i}-max_probs.png')

plt.figure(2)
plt.savefig(f'{OUTPUT_DIR}/generated-{i}-max_probs_0_3.png')

plt.figure(3)
plt.savefig(f'{OUTPUT_DIR}/generated-{i}-max_probs_1_3.png')

plt.figure(4)
plt.savefig(f'{OUTPUT_DIR}/generated-{i}-max_probs_2_3.png')

visualize(generated_tokens, f'{OUTPUT_DIR}/generated-{i}.png', length=LENGTH_IN_SECONDS)

# write generated_tokens to generated-{i}.txt
with open(f'{OUTPUT_DIR}/generated-{i}.txt', 'w') as f:
    f.write(' '.join([str(tok) for tok in generated_tokens]))

if GET_PROMPT:
    prompt_mid = events_to_midi(prompt, debug=False)
    prompt_mid.save(f'{OUTPUT_DIR}/prompt-{i}.mid')
