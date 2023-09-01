import time

import numpy as np
import torch

import sys
import os
import time
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.visuals import visualize
from anticipation.config import *

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 60

model_name = 'major-lion-2'
step_number = 100000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/kathli/checkpoints/{model_name}/step-{step_number}/hf').cuda()
#model = AutoModelForCausalLM.from_pretrained(f'/nlp/scr/jthickstun/anticipation/checkpoints/{model_name}/step-{step_number}/hf', ignore_mismatched_sizes=True).cuda()
# prompt = [SEPARATOR, SEPARATOR, SEPARATOR]

prompt = []
my_dir = f'/nlp/scr/kathli/output/{model_name}'
if not os.path.exists(my_dir):
    os.makedirs(my_dir)

i = 0
start = time.time()
generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, [], top_p=0.90, debug=False)
end = time.time()
#ops.print_tokens(generated_tokens)
print("time", end - start)
mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
while os.path.exists(f'{my_dir}/generated-{i}.mid'):
    i += 1
mid.save(f'{my_dir}/generated-{i}.mid')
#visualize(generated_tokens, f'{my_dir}/generated-{i}.png')


