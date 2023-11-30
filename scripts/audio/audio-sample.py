import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from anticipation.audio import SEPARATOR

# initialize the model and tokenizer
model_name = '/juice4/scr4/nlp/music/audio-checkpoints/teeu4qs9/step-80000/hf/'
model = AutoModelForCausalLM.from_pretrained(model_name)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
#torch.manual_seed(42)
torch.manual_seed(47)

# set the prompt
input_ids = torch.tensor([SEPARATOR, SEPARATOR, SEPARATOR, SEPARATOR]).to(device)

# define the number of tokens to generate
num_tokens_to_generate = 8192-4

# initialize the past_key_values tensor to None
past_key_values = None

# generate the tokens
for _ in range(num_tokens_to_generate):
    # generate the logits and update past_key_values
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values)
    logits = outputs.logits[-1,:]
    #past_key_values = outputs.past_key_values

    # sample the next token
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    next_token = torch.multinomial(probabilities, num_samples=1)[-1].unsqueeze(0)

    # append the next token to the input_ids
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    #print(input_ids.shape, input_ids.min(), input_ids.max())

# print the generated sequence
print(' '.join([str(tok) for tok in input_ids.cpu().numpy().tolist()]))
