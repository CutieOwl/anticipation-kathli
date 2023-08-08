"""
API functions for sampling from anticipatory infilling models.
"""

import math

import torch
import torch.nn.functional as F

from tqdm import tqdm

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *


def nucleus(logits, top_p):
    # from HF implementation
    if top_p < 1.0:
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


def instr_logits_inter(logits, full_history):
    """ don't sample more than 16 instruments """
    instrs = ops.get_instruments(full_history)
    #print(instrs)
    if len(instrs) < 15:
        return logits

    #print(MAX_INSTR)
    #print(MAX_PITCH)
    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits


def safe_logits_inter(logits, idx):
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') # don't generate controls
    logits[SPECIAL_OFFSET+1:] = -float('inf')               # don't generate special tokens (except SEP)
    #logits[9999] = -float('inf')                          # don't generate UNK

    # don't generate stuff in the wrong time slot
    if idx == 0:
        # only generate values between 0 and 100 or SEP and UNK
        #logits[101:9999] = -float('inf')
        #logits[1000:SEPARATOR] = -float('inf')
        #logits[SEPARATOR+1:] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE+1] = -float('inf') # add +1 to prevent sampling rests
    elif idx == 1:
        # only generate values between DUR_OFFSET and DUR_OFFSET+MAX_DUR or SEP
        #logits[:DUR_OFFSET] = -float('inf')
        #logits[DUR_OFFSET+MAX_DUR:SEPARATOR] = -float('inf')
        #logits[SEPARATOR+1:] = -float('inf')
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE+1] = -float('inf') # add +1 to prevent sampling rests
    elif idx == 2:
        # only generate values between NOTE_OFFSET and NOTE_OFFSET+MAX_NOTE+1 or SEP or REST
        #logits[:NOTE_OFFSET] = -float('inf')
        #logits[NOTE_OFFSET+MAX_NOTE+1:SEPARATOR] = -float('inf')
        #logits[SEPARATOR+1:] = -float('inf')
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')

    return logits

    
def add_token_inter(model, z, tokens, top_p, current_time, debug=False, past=None):
    assert len(tokens) % 3 == 0

    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    #history = history[lookback:] # Markov window
    #offset = ops.min_time(history, seconds=False)
    #history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            #print("input_tokens", input_tokens)
            if past is not None:
                #print('use past')
                output = model(input_tokens[:,-1:], past_key_values=past)
            else:
                output = model(input_tokens)
        
            #print("output_logits", output.logits.shape)
            logits = output.logits[0, -1, :]
            #print("logits", logits.shape)
            idx = input_tokens.shape[1]-1

            logits = safe_logits_inter(logits, idx)
            if i == 2:
                logits = instr_logits_inter(logits, tokens)
            logits = nucleus(logits, top_p)
            probs = F.softmax(logits, dim=-1)

            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

            past = output.past_key_values

    #new_token[0] += offset # revert to full sequence timing
    return new_token, past


def add_token_inter_short(model, z, tokens, top_p, current_time, debug=False, past=None, use_9999=False):
    assert len(tokens) % 3 == 0
    #print("inside short inter add token")
    history = tokens.copy()
    lookback = max(len(tokens) - 1011, 0)
    history = history[lookback:] # Markov window
    #offset = ops.min_time(history, seconds=False)
    #history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            if use_9999:
                if input_tokens.shape[1] > 1:
                    input_tokens[0, 1] = 9999
            #print("input_tokens", input_tokens)
            #print("input tokens shape", input_tokens.shape)
            output = model(input_tokens)
        
            #print("output_logits", output.logits.shape)
            logits = output.logits[0, -1, :]
            #print("logits", logits.shape)
            idx = input_tokens.shape[1]-1
            logits = safe_logits_inter(logits, idx)
            if i == 2:
                logits = instr_logits_inter(logits, tokens)
            logits = nucleus(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

            past = None #output.past_key_values

    #new_token[0] += offset # revert to full sequence timing
    return new_token, past


def add_token_inter_eval(model, z, tokens, top_p, current_time, debug=False, past=None):
    assert len(tokens) % 3 == 0

    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    #history = history[lookback:] # Markov window
    #offset = ops.min_time(history, seconds=False)
    #history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    max_probs = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            #print("input_tokens", input_tokens)
            if past is not None:
                #print('use past')
                output = model(input_tokens[:,-1:], past_key_values=past)
            else:
                output = model(input_tokens)
        
            #print("output_logits", output.logits.shape)
            logits = output.logits[0, -1, :]
            eval_logits = output.logits
           
            idx = input_tokens.shape[1]-1

            logits = safe_logits_inter(logits, idx)
            if i == 2:
                logits = instr_logits_inter(logits, tokens)
            logits = nucleus(logits, top_p)
            probs = F.softmax(logits, dim=-1)

            token = torch.multinomial(probs, 1)
            new_token.append(int(token))
            
            max_probs.append(probs.max().item())

            past = output.past_key_values

    #new_token[0] += offset # revert to full sequence timing
    return new_token, past, max_probs


def generate_inter(model, end_time, prompt = None, top_p=1.0, debug=False, use_9999=False, add_token=None):
    start_time = 0
    end_time = int(TIME_RESOLUTION*end_time)

    z = [AUTOREGRESS]
    if debug:
        print('AR Mode' if z[0] == AUTOREGRESS else 'AAR Mode')

    current_time = 0
    tokens = []
    
    if prompt is not None:
        tokens = prompt.copy()

    past_kv = None
    with tqdm(range(end_time)) as progress:
        while True:
            #print("tokens", tokens)
            if add_token == "short":
                new_token, past_kv = add_token_inter_short(model, z, tokens, top_p, max(start_time,current_time), past=past_kv, use_9999=use_9999)
            else:
                new_token, past_kv = add_token_inter(model, z, tokens, top_p, max(start_time,current_time), past=past_kv)
            #print("new token", new_token)
            dt = new_token[0]
            assert dt >= 0
            if dt == 9999:
                print("interarrival time 9999, using 0")
               #new_token[0] = 0
                dt = 0
            if dt == SEPARATOR:
                print("interarrival time SEPARATOR?")
                dt = 0

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                #print("instr", new_instr)
                new_pitch = new_note - (2**7)*new_instr
                print('C', dt + current_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            current_time += dt
            if current_time >= end_time:
                print("current time", current_time, "end time", end_time, "exceeding token", new_token)
                break

            tokens.extend(new_token)
            
            progress.update(dt)

    return tokens


def generate_inter_eval(model, end_time, prompt = None, top_p=1.0, debug=False, use_9999=False):
    start_time = 0
    end_time = int(TIME_RESOLUTION*end_time)

    z = [AUTOREGRESS]
    if debug:
        print('AR Mode' if z[0] == AUTOREGRESS else 'AAR Mode')

    current_time = 0
    tokens = []
    
    if prompt is not None:
        tokens = prompt.copy()

    max_probs = []
    past_kv = None
    with tqdm(range(end_time)) as progress:
        while True:
            #print("tokens", tokens)
            new_token, past_kv, new_probs = add_token_inter_eval(model, z, tokens, top_p, max(start_time,current_time), past=past_kv)
            #print("new token", new_token)
            dt = new_token[0]
            assert dt >= 0
            if dt == 9999:
                print("interarrival time 9999, using 0")
               #new_token[0] = 0
                dt = 0
            if dt == SEPARATOR:
                print("interarrival time SEPARATOR?")
                dt = 0

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                #print("instr", new_instr)
                new_pitch = new_note - (2**7)*new_instr
                print('C', dt + current_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            current_time += dt
            if current_time >= end_time:
                print("current time", current_time, "end time", end_time, "exceeding token", new_token)
                break

            tokens.extend(new_token)
            max_probs.extend(new_probs)
            
            progress.update(dt)

    return tokens, max_probs

