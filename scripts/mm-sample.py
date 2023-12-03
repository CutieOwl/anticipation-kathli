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

MODEL = "bf5v4qyg"
STEP_NUM = 99001 #42430 #97645

AUDIO_DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt"
MIDI_DATA = "/juice4/scr4/nlp/music/datasets/valid/lakh.midigen.valid.txt"
SUBSAMPLE_IDX = 0

USE_PROMPT = False

ADD_SEPARATOR = True

MODE = "midi"

TOP_P = 0.98

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
torch.manual_seed(42)

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


# set the prompt
if MODE == "midi":
    input_ids = torch.tensor([4, 8, 2, 2]).to(device)
else:
    input_ids = torch.tensor([6109, 6113, 6108, 6108]).to(device)

if ADD_SEPARATOR:
    input_ids = torch.cat([input_ids, torch.tensor([0,0,0,0]).to(device)], dim=-1)

# define the number of tokens to generate
num_tokens_to_generate = n_positions - input_ids.size(0)

if USE_PROMPT:
    if MODE == "midi":
        prompt_file = MIDI_DATA
    else: 
        prompt_file = AUDIO_DATA
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

# initialize the past_key_values tensor to None
past_key_values = None

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    for _ in range(0, num_tokens_to_generate, 1):
        # generate the logits and update past_key_values
        with torch.no_grad():
            logits = model(input_ids, past_key_values=past_key_values).logits[-1,:]
        
        logits = nucleus(logits, TOP_P)
        # sample the next token
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        next_token = torch.multinomial(probabilities, num_samples=1)
    
        next_token = next_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        #past_key_values = output.past_key_values

        progress.update(1)

# print the generated sequence
generated_sequence = ' '.join([str(tok) for tok in input_ids.cpu().numpy().tolist()])
print(generated_sequence)

# save the generated sequence to a file
OUTPUT_DIR = f'/nlp/scr/kathli/output/mm/{MODEL}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0

while os.path.exists(f'{OUTPUT_DIR}/generated-{MODE}-{i}.txt'):
    i += 1
with open(f'{OUTPUT_DIR}/generated-{MODE}-{i}.txt', 'w') as f:
    f.write(generated_sequence)

print(f'Printed to {OUTPUT_DIR}/generated-{MODE}-{i}.txt')
