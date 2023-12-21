import torch
import sys
import os
import json
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from torch.nn import functional as F

from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.audiovocab import SEPARATOR

MODEL = "pu4yo6b5"# "54labs45" #"ha05xrd3" #"pu4yo6b5" #  #"9qbavecu"
STEP_NUM = 81001 # 99920 # #81001 # #99588 #42430

AUDIO_DATA = "/juice4/scr4/nlp/music/datasets/encodec_fma.audiogen.valid.txt"
MIDI_DATA = "/juice4/scr4/nlp/music/temp_test/lakh.midigen.test.txt"
SUBSAMPLE_IDX = 0

USE_PROMPT = False

ADD_SEPARATOR = False

MODE = "trans"

TOP_P = 0.99

model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'

# initialize the model and tokenizer
model_config_json = json.load(open(f"{model_name}/config.json"))
#model_config["n_positions"] = SHORT_SEQ_LEN
print("n_positions", model_config_json["n_positions"])
n_positions = model_config_json["n_positions"]
model_config = GPT2Config.from_dict(model_config_json)
model = GPT2LMHeadModel.from_pretrained(model_name, config=model_config)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)

def nucleus(logits, top_p):
    # from HF implementation
    if top_p < 1.0:
        #print("nucleus sampling with top_p", top_p)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")
    
    return logits

PROMPT_IDX = 0
PROMPT_PART = 4

# set the prompt
if MODE == "synth":
    prompt_file = "/nlp/scr/kathli/synthesize_valid.txt"
    PROMPT_IDX
    with open(prompt_file, 'r') as f:
        for i, line in enumerate(f):
            if i < prompt_idx:
                continue

            if i > prompt_idx:
                break

            tokens = [int(token) for token in line.split()]

            # find first occurrence of 6 7 10 7 in tokens
            prompt_upto = 0
            for j in range(len(tokens)):
                if tokens[j] == 6 and tokens[j+1] == 7 and tokens[j+2] == 10 and tokens[j+3] == 7:
                    prompt_upto = j+4
                    break
            input_ids = torch.tensor(tokens[:prompt_upto]).to(device)
else:
    prompt_file = "/nlp/scr/kathli/transcribe_valid.txt"
    with open(prompt_file, 'r') as f:
        for i, line in enumerate(f):
            if i < PROMPT_IDX:
                continue

            if i > PROMPT_IDX:
                break

            tokens = [int(token) for token in line.split()]

            # find prompt_part-th occurrence of 5 10 7 7 in tokens
            prompt_start = 0
            start_occurrence = 0
            if PROMPT_PART > 0:
                for j in range(0, len(tokens), 4):
                    if tokens[j] == 5 and tokens[j+1] == 10 and tokens[j+2] == 7 and tokens[j+3] == 7:
                        prompt_start = j
                        if start_occurrence == PROMPT_PART:
                            break
                        start_occurrence += 1
            print("prompt_start", prompt_start)

            prompt_upto = 0
            end_occurrence = 0
            got_break = False
            for j in range(0, len(tokens), 4):
                prompt_upto = j+4
                if tokens[j] == 5 and tokens[j+1] == 10 and tokens[j+2] == 7 and tokens[j+3] == 10:
                    #prompt_upto = j+4
                    if end_occurrence == PROMPT_PART:
                        got_break = True
                        break
                    end_occurrence += 1
            print("prompt_upto", prompt_upto)
            input_ids = torch.tensor(tokens[prompt_start:prompt_upto]).to(device)
            if PROMPT_PART > 0:
                input_ids = torch.cat([torch.tensor([5,10,7,2,0,0,0,0]).to(device), input_ids], dim=-1)
            if not got_break:
                input_ids = torch.cat([input_ids, torch.tensor([5,10,7,10]).to(device)], dim=-1)

print("INPUT_IDS", input_ids)

if ADD_SEPARATOR:
    input_ids = torch.cat([input_ids, torch.tensor([0,0,0,0]).to(device)], dim=-1)

if USE_PROMPT:
    if MODE == "midi":
        prompt_file = MIDI_DATA
    else: 
        prompt_file = AUDIO_DATA
    prompt_idx = 10
    prompt_upto = 1216
    with open(prompt_file, 'r') as f:
        for i, line in enumerate(f):
            if i < prompt_idx:
                continue

            if i > prompt_idx:
                break

            tokens = [int(token) for token in line.split()]
            input_ids = torch.tensor(tokens[:prompt_upto]).to(device)

# initialize the past_key_values tensor to None
past_key_values = None

# define the number of tokens to generate
num_tokens_to_generate = n_positions - input_ids.size(0)

print("MODEL", MODEL)
print("STEP_NUM", STEP_NUM)
print("MODE", MODE)
print("PROMPT_PART", PROMPT_PART)

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    for i in range(0, num_tokens_to_generate, 1):
        # generate the logits and update past_key_values
        with torch.no_grad():
            logits = model(input_ids, past_key_values=past_key_values).logits[-1,:]
        
        logits = nucleus(logits, TOP_P)
        if MODE == "trans": # only generate midi
            logits[11:4208] = -float('inf')
        else: # only generate audio
            logits[4208:] = -float('inf')
        # sample the next token
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        next_token = torch.multinomial(probabilities, num_samples=1)
    
        next_token = next_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token < 11:
            break
        #past_key_values = output.past_key_values

        progress.update(1)

# make sure all sequences have length multiple of 4
if input_ids.size(0) % 4 != 0:
    extra_zeros = 4 - input_ids.size(0) % 4
    input_ids = torch.cat([input_ids, torch.tensor([0] * extra_zeros).to(device)], dim=-1)

# print the generated sequence
generated_sequence = ' '.join([str(tok) for tok in input_ids.cpu().numpy().tolist()])
print(generated_sequence)

# save the generated sequence to a file
OUTPUT_DIR = f'/nlp/scr/kathli/output/mm/{MODEL}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0

while os.path.exists(f'{OUTPUT_DIR}/generated-{MODE}-{PROMPT_PART}-{PROMPT_IDX}.txt'):
    i += 1
with open(f'{OUTPUT_DIR}/generated-{MODE}-{PROMPT_PART}-{PROMPT_IDX}.txt', 'w') as f:
    f.write(generated_sequence)

print(f'Printed to {OUTPUT_DIR}/generated-{MODE}-{PROMPT_PART}-{PROMPT_IDX}.txt')
