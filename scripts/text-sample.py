import torch
import sys
import os
from tqdm import tqdm
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import AutoTokenizer, GPT2Tokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from anticipation.audiovocab import SEPARATOR

#MODEL = "fsowdi0i"
MODEL = "kly1r553"

# initialize the model and tokenizer
model_name = f'/nlp/scr/kathli/checkpoints/{MODEL}/step-115000/hf/'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(42)

# Text prompt
prompt = "Q: 5 + 9 = ? A: 14 Q: 2 + 2 = ? A: 4 Q: 14 + 17 = ? A: 31 Q: 7 + 2 = ? A:"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# define the number of tokens to generate
num_tokens_to_generate = 1024

# initialize the past_key_values tensor to None
past_key_values = None

# generate the tokens
with tqdm(range(num_tokens_to_generate)) as progress:
    for _ in range(0, num_tokens_to_generate, 1):
        # generate the logits and update past_key_values
        with torch.no_grad():
            logits = model(input_ids, past_key_values=past_key_values).logits[:,-1,:]
        #print(outputs.logits.shape)
        
        # sample the next token
        #print(logits.shape)
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        #print(probabilities.shape)
        next_token = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)
        #print(next_token.shape)
        #print(input_ids.shape)
        
        next_token = next_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        progress.update(1)

# Decode the generated sequence
generated_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)

print("Generated Sequence:")
print(generated_sequence)

# save the generated sequence to a file
OUTPUT_DIR = f'/nlp/scr/kathli/output/{MODEL}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

i = 0

while os.path.exists(f'{OUTPUT_DIR}/text_generated-{i}.txt'):
    i += 1
with open(f'{OUTPUT_DIR}/text_generated-{i}.txt', 'w') as f:
    f.write(generated_sequence)

print(f'Printed to {OUTPUT_DIR}/text_generated-{i}.txt')

# print the generated sequence
# generated_sequence = ' '.join([str(tok) for tok in input_ids.cpu().numpy().tolist()])
