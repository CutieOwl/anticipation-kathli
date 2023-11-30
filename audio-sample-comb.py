import torch
import sys
import os
import json
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.mmvocab import *
#from anticipation.audiovocab import SEPARATOR

MODEL = "jqv0lgv4"
STEP_NUM = 99481 #97645

DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt"

SUBSAMPLE = 100
SUBSAMPLE_IDX = 0

model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'

# initialize the model and tokenizer
model_config_json = json.load(open(f"{model_name}/config.json"))
#model_config["n_positions"] = SHORT_SEQ_LEN
print("n_positions", model_config_json["n_positions"])
model_config = GPT2Config.from_dict(model_config_json)
model = GPT2LMHeadModel.from_pretrained(model_name, config=model_config)

USE_PROMPT = False
USE_SAFE_LOGITS = True

def safe_logits(logits, idx):
    #logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') # don't generate controls
    #logits[SPECIAL_OFFSET:] = -float('inf')               # don't generate special tokens

    # don't generate stuff in the wrong time slot
    if idx % 604 == 0:
        #print("idx", idx, "scale")
        logits[:SCALE_OFFSET] = -float('inf')
        logits[SCALE_OFFSET+SCALE_RESOLUTION:] = -float('inf')
    else:
        logits[SCALE_OFFSET:SCALE_OFFSET+SCALE_RESOLUTION] = -float('inf')
    #elif idx % 3 == 1:
        #logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        #logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE] = -float('inf')
    #elif idx % 3 == 2:
        # logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        # logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')

    return logits

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(42)

SEPARATOR = 0

# set the prompt
#input_ids = torch.tensor([SEPARATOR, SEPARATOR, SEPARATOR, SEPARATOR]).to(device)
input_ids = torch.tensor([6109, 6113, 6108, 6108, 0, 0, 0, 0]).to(device)

# initialize the past_key_values tensor to None
past_key_values = None

if USE_PROMPT:
    prompt_file = '/juice4/scr4/nlp/music/audio/dataset/test.txt'
    prompt_idx = 0
    prompt_upto = 100
    with open(prompt_file, 'r') as f:
        for i, line in enumerate(f):
            if i < prompt_idx:
                continue

            if i > prompt_idx:
                break

            tokens = [int(token) for token in line.split()]
            input_ids = torch.tensor(tokens[:prompt_upto]).to(device)

print("Prompt:", input_ids)

# define the number of tokens to generate
num_tokens_to_generate = 8192-input_ids.shape[0]

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    fine_num = 0
    for i in range(0, num_tokens_to_generate, 4):
        # generate the logits and update past_key_values
        ##print("input ids shape", input_ids.shape)
        with torch.no_grad():
            logits = model(input_ids, past_key_values=past_key_values).logits[-4:,:]

        ##print(outputs.logits.shape)
        #logits = outputs
        ##print("logits shape", logits.shape)
        #past_key_values = outputs.past_key_values

        # safe the logits
        if USE_SAFE_LOGITS:
            for idx in range(4):
                logits[idx] = safe_logits(logits[idx], fine_num)
                fine_num += 1

        # sample the next token
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        ##print(probabilities.shape)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze()
        ##print(next_token.shape)
        #next_token = next_token[-4:,:].squeeze()
        ##print(next_token)

        # append the next token to the input_ids
        # print(input_ids.device)
        # print(next_token.device)
        # print(input_ids.shape)
        # print(next_token.shape)
        next_token = next_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        #print(input_ids.shape, input_ids.min(), input_ids.max())
        progress.update(4)

# print the generated sequence
generated_sequence = ' '.join([str(tok) for tok in input_ids.cpu().numpy().tolist()])
print(generated_sequence)

# save the generated sequence to a file
OUTPUT_DIR = f'/nlp/scr/kathli/output/audio/{MODEL}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0

while os.path.exists(f'{OUTPUT_DIR}/generated-{i}.txt'):
    i += 1
with open(f'{OUTPUT_DIR}/generated-{i}.txt', 'w') as f:
    f.write(generated_sequence)

print(f'Printed to {OUTPUT_DIR}/generated-{i}.txt')
