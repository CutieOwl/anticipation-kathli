import torch
import sys
import os
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.audiovocab import SEPARATOR

MODEL = "fsowdi0i"
#MODEL = "teeu4qs9"

# initialize the model and tokenizer
model_name = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}/step-97517/hf/'
#model_name = f'/juice4/scr4/nlp/music/audio-checkpoints/{MODEL}/step-80000/hf'
model = GPT2LMHeadModel.from_pretrained(model_name)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(42)

# set the prompt
input_ids = torch.tensor([SEPARATOR, SEPARATOR, SEPARATOR, SEPARATOR]).to(device)

# define the number of tokens to generate
num_tokens_to_generate = 8192-4

# initialize the past_key_values tensor to None
past_key_values = None

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    for _ in range(0, num_tokens_to_generate, 1):
        # generate the logits and update past_key_values
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values)
        logits = outputs.logits[-1,:]
        # print(logits.shape)
        #past_key_values = outputs.past_key_values

        # sample the next token
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        # print(probabilities.shape)
        next_token = torch.multinomial(probabilities, num_samples=1)
        #print(next_token.shape)
        next_token = next_token[-1:]

        # append the next token to the input_ids
        # print(input_ids.device)
        # print(next_token.device)
        #print(input_ids.shape)
        #print(next_token.shape)
        next_token = next_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        #print(input_ids.shape, input_ids.min(), input_ids.max())
        progress.update(1)

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
