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

MODEL = "incneoi4"#"multi-head"# "ha05xrd3" # "54labs45" #" #"9qbavecu"
STEP_NUM = 99758 #99802 #98517 #99920 #50000 #99588 # ##99698  #42430

#AUDIO_DATA = "/juice4/scr4/nlp/music/datasets/encodec_fma.audiogen.valid.txt"
AUDIO_DATA = "/juice4/scr4/nlp/music/temp_test/encodec_fma.audiogen.valid-small.txt"
#MIDI_DATA = "/juice4/scr4/nlp/music/temp_test/lakh.midigen.test.txt"
MIDI_DATA = "/juice4/scr4/nlp/music/temp_test/lakh-rev-skew.midigen-1024.valid-small.txt"
SUBSAMPLE_IDX = 0

USE_PROMPT = False

ADD_SEPARATOR = False

MODE = "audio"

TOP_P = 0.98 #0.98

model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-{STEP_NUM}/hf'
#model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf'
#model_name = f'/juice4/scr4/nlp/music/prelim-checkpoints/{MODEL}/step-{STEP_NUM}/hf/'

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

def safe_logits(logits, idx):
    #logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') # don't generate controls
    #logits[SPECIAL_OFFSET:] = -float('inf')               # don't generate special tokens

    # don't generate stuff in the wrong time slot
    if idx % 604 == 0:
        #print("idx", idx, "scale")
        logits[:4108] = -float('inf')
        logits[4208:] = -float('inf')
    else:
        logits[4108:4208] = -float('inf')

    return logits

# set the prompt
if MODE == "midi":
    input_ids = torch.tensor([4, 8, 2, 2]).to(device)
else:
    input_ids = torch.tensor([3, 7, 2, 2]).to(device)

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

# define the number of tokens to generate
num_tokens_to_generate = n_positions - input_ids.size(0)

print("MODEL", MODEL)
print("STEP_NUM", STEP_NUM)
print("MODE", MODE)
print("TOP_P", TOP_P)
print("NUM_TOKENS_TO_GENERATE", num_tokens_to_generate)

# initialize the past_key_values tensor to None
past_key_values = None

input_ids = input_ids.unsqueeze(0)
output_ids = input_ids.clone()

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    for i in range(0, num_tokens_to_generate, 4):
        # generate the logits and update past_key_values
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)

        #past_key_values = outputs.past_key_values

        #print("outputs.logits", outputs.logits.shape)

        # create an empty array to hold the next token
        for j in range(4):
            # sample the next token
            logits = outputs.logits[0,-4+j]
            #logits = safe_logits(logits, idx)
            #print("logits", logits.shape)
            logits = nucleus(logits, TOP_P)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = next_token.unsqueeze(0).to(output_ids.device)
            #print("next_token", next_token.shape)
            #print("output_ids", output_ids.shape)
            #output_ids = torch.cat([output_ids, next_token], dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        # make input_ids last 4 tokens of output ids
        #input_ids = output_ids[:,-4:].clone()

        # #logits = nucleus(logits, TOP_P)
        # print("logits", logits.shape)
        # probabilities = torch.softmax(logits, dim=-1).squeeze()
        # print("probabilities", probabilities.shape)
        # next_token = torch.multinomial(probabilities, num_samples=1).squeeze()

        # next_token = next_token.unsqueeze(0).to(output_ids.device)

        # print("next_token", next_token.shape)

        # print("output_ids", output_ids.shape)
        # # append the next token to the input_ids
        # output_ids = torch.cat([output_ids, next_token], dim=-1)
        # #print("output_ids", output_ids.shape)
        # input_ids = next_token

        progress.update(4)

#output_ids = output_ids.squeeze().cpu().numpy()
input_ids = input_ids.squeeze().cpu().numpy()

# print the generated sequence
generated_sequence = ' '.join([str(tok) for tok in input_ids])
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
