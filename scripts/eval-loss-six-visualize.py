import time

from statistics import mean
from math import exp

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

import json

import sys
#sys.path.append('/home/kml11/transformers-levanter/src')
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from argparse import ArgumentParser
from tqdm import tqdm

from scipy.signal import lfilter

from anticipation import ops
from anticipation.config import M, EVENT_SIZE
from anticipation.config import *
from anticipation.vocab import *
from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION, AUTOREGRESS
from anticipation.ops import max_time

DATASET = "lakh-data-inter-16384"
SHORTF_MODEL_NAME = "fiery-vortex-95"
SHORTF_STEP_NUMBER = 15000
SHORTS_MODEL_NAME = "toasty-silence-91"
SHORTS_STEP_NUMBER = 115000
LONG_MODELS_NAME = ["driven-plant-48", "genial-sea-181", "w7uq05r3", "fh86cy5o"]
LONG_STEP_NUMBERS = [30000, 115000, 115000, 115000]
LONG_SEQ_LEN = 16384 # must be 1 mod 3
SHORT_SEQ_LEN = 1024 # must be 1 mod 3
NEW_CE_CUTOFF = 255 # must be less than SEQ_LEN
PRINT_GRAPH = True
PRINT_IDX = 0 # must be less than SUBSAMPLE, leave as 0 for default
PRINT_UP_TO_IDX = 16384 #8652 # none if we print the entire example
ADD_9999 = False
REMOVE_9999 = False
FILTER_CONST = 75
SUBSAMPLE=100
OUTPUT_DIR = f'/nlp/scr/kathli/eval/rep_struct/figs_six'
USE_FILTER = True
SINGLE_SPECIFIC = None #12108

SHORTF_MODEL_LABEL = "finetune short"
SHORTS_MODEL_LABEL = "short only" #SHORT_MODEL_NAME
LONG_MODELS_LABEL = ["finetune long", "long only", "2-stage", "mixed"] # LONG_MODEL_NAME

DATAFILE = 'f00f4d6b'
#DATAFILE = 'generated-19'
#DATA = f'/nlp/scr/kathli/output/genial-sea-181/{DATAFILE}.txt'
DATA = f'/nlp/scr/kathli/eval/rep_struct/{DATAFILE}.txt' #f'/nlp/scr/kathli/output/driven-plant-48/{DATAFILE}.txt' # f"/nlp/scr/kathli/datasets/{DATASET}/test.txt" 
SHORTF_CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{SHORTF_MODEL_NAME}/step-{SHORTF_STEP_NUMBER}/hf"
SHORTS_CHECKPOINT= f"/nlp/scr/kathli/checkpoints/{SHORTS_MODEL_NAME}/step-{SHORTS_STEP_NUMBER}/hf"
LONG_CHECKPOINTS = []
for i in range(len(LONG_MODELS_NAME)):
    LONG_CHECKPOINTS.append(f"/nlp/scr/kathli/checkpoints/{LONG_MODELS_NAME[i]}/step-{LONG_STEP_NUMBERS[i]}/hf")

t0 = time.time()

shortf_config = json.load(open(f"{SHORTF_CHECKPOINT}/config.json"))
shortf_config["n_positions"] = SHORT_SEQ_LEN
shortf_config = GPT2Config.from_dict(shortf_config)
shortf_model = GPT2LMHeadModel.from_pretrained(SHORTF_CHECKPOINT, config=shortf_config).cuda() \

shorts_config = json.load(open(f"{SHORTS_CHECKPOINT}/config.json"))
shorts_config["n_positions"] = SHORT_SEQ_LEN
shorts_config = GPT2Config.from_dict(shorts_config)
shorts_model = GPT2LMHeadModel.from_pretrained(SHORTS_CHECKPOINT, config=shorts_config).cuda() 
print(f'Loaded short models ({time.time()-t0} seconds)')
t1 = time.time()

long_models = []
for i in range(len(LONG_MODELS_NAME)):
    long_config = json.load(open(f"{LONG_CHECKPOINTS[i]}/config.json"))
    long_config["n_positions"] = LONG_SEQ_LEN
    long_config = GPT2Config.from_dict(long_config)
    one_long_model = GPT2LMHeadModel.from_pretrained(LONG_CHECKPOINTS[i], config=long_config).cuda() 
    long_models.append(one_long_model)
    
print(f'Loaded long models ({time.time()-t1} seconds)')

LONG_COLORS = ['red', 'sienna', 'darkorange', 'gold']

def sliding_window_eval(tokens, short_model):
    idx = 1
    # do sliding window for short sequences
    curr_ce = torch.empty(0)
    while idx < len(tokens):
        end_idx = min(len(tokens), idx + 1023)
        curr_tokens = tokens[idx:end_idx]
        if ADD_9999:
            curr_tokens[0] = 9999 # make it match the samples a bit more
        curr_tokens.insert(0, AUTOREGRESS)
        curr_tokens = torch.tensor(curr_tokens).unsqueeze(0).cuda()
        logits = short_model(curr_tokens).logits[0]
        logits = logits.cuda()
        cross_entr = F.cross_entropy(logits[:-1],curr_tokens[0,1:],reduction='none')
        my_cross_entr = cross_entr.cpu()
        if idx == 1:
            curr_ce = torch.cat([curr_ce, my_cross_entr])
        else:
            # only add the last tokens
            curr_ce = torch.cat([curr_ce, my_cross_entr[(SHORT_SEQ_LEN - NEW_CE_CUTOFF - 1):]])
        idx += NEW_CE_CUTOFF

        if idx + (SHORT_SEQ_LEN - NEW_CE_CUTOFF - 1) >= len(tokens):
            break
    return curr_ce


def visualize(tokens, my_plt, selected=None, length=120):
    #colors = ['white', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan',
    #          'dodgerblue', 'slategray', 'navy', 'mediumpurple', 'mediumorchid', 'magenta', 'lightpink']
    colors = ['white', '#426aa0', '#b26789', '#de9283', '#eac29f', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan', 'dodgerblue', 'slategray', 'navy']
 
    #print("tokens", tokens[:100])
    max_time = length #ops.max_time(tokens, seconds=False)
    grid = np.zeros([max_time, MAX_PITCH])
    instruments = list(sorted(list(ops.get_instruments(tokens).keys())))
    if 128 in instruments:
        instruments.remove(128)

    print_time = 1000

    cur_time = 0
    for j, (tm, dur, note) in enumerate(zip(tokens[0::3],tokens[1::3],tokens[2::3])):
        if note == SEPARATOR:
            assert tm == SEPARATOR and dur == SEPARATOR
            print(j, 'SEPARATOR')
            continue

        cur_time += tm
        tm = cur_time - TIME_OFFSET
        dur = dur - DUR_OFFSET
        note = note - NOTE_OFFSET
        instr = note//2**7
        pitch = note - (2**7)*instr

        if note == REST:
            continue

        assert note < CONTROL_OFFSET

        if instr == 128: # drums
            continue     # we don't visualize this

        if selected and instr not in selected:
            continue

        if cur_time > print_time:
            print_time += 1000
            print("passed time", cur_time, "at event", j, "token", j * 3)

        if tm+dur > max_time:
            print('time has exceeded max_time', tm+dur, max_time)
            print('exceeded at event', j)
            break
        grid[tm:tm+dur, pitch] = 1+instruments.index(instr)

    #print('cur_time', cur_time)

    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = list(range(MAX_TRACK_INSTR)) + [16]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    grid_data = np.flipud(grid.T)
    np.set_printoptions(threshold=np.inf)
    #print("grid_shape", grid_data[:128,:300])

    my_plt.imshow(grid_data, aspect='auto', cmap=cmap, norm=norm, interpolation='none')

    patches = [matplotlib.patches.Patch(color=colors[i+1], label=f"{instruments[i]}")
               for i in range(len(instruments))]
    my_plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


def create_graph_and_visualization(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, use_filter=False, single_specific=None, alpha=0.7):
    START_IDX = 0

    tokens_plot = np.array(tokens[0,1:].cpu())
    if tokens_plot[0] == SEPARATOR:
        START_IDX = 3
    
    tokens_plot = tokens_plot[START_IDX:]
    curr_ce_f = curr_ce_f[START_IDX:PRINT_UP_TO_IDX]
    curr_ce_s = curr_ce_s[START_IDX:PRINT_UP_TO_IDX]
    for i in range(len(LONG_MODELS_NAME)):
        my_cross_entr[i] = my_cross_entr[i][START_IDX:PRINT_UP_TO_IDX]
    should_print = False
    
    # Create an array of indices from 0 to length-1 of `ce`
    indices = tokens_plot[0::3]
    #print("time_indices", indices[-100:])
    indices = np.cumsum(indices)
    #print("cumsum time_indices", indices[-100:])
    indices = [time for i, time in enumerate(indices) for _ in range(3)]
    #print("indices", indices[-100:])
    max_time = max(indices)
    #print("max_time", max_time)

    # Plot 1: ce
    fig = plt.figure(4, figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])  # Ratio determines plot heights

    plt1 = plt.subplot(gs[0])
    plot_f = curr_ce_f
    plot_s = curr_ce_s
    plot_l = []

    if use_filter:
        # create a kalman filter to smooth the plot
        n = FILTER_CONST  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        plot_f = lfilter(b, a, curr_ce_f)
        plot_s = lfilter(b, a, curr_ce_s)
        for i in range(len(LONG_MODELS_NAME)):
            plot_l.append(lfilter(b, a, my_cross_entr[i]))
    else:
        plot_l = my_cross_entr 
    
    plt1.plot(indices, plot_f, label=SHORTF_MODEL_LABEL, color='blue', alpha=alpha)
    plt1.plot(indices, plot_s, label=SHORTS_MODEL_LABEL, color='green', alpha=alpha)
    for i in range(len(LONG_MODELS_NAME)):
        plt1.plot(indices, plot_l[i], label=LONG_MODELS_LABEL[i], color=LONG_COLORS[i], alpha=alpha)

    if single_specific:
        marker_indices = [time for i, time in enumerate(indices) if tokens_plot[i - (i % 3) + 2] == single_specific]
        marker_ce_f = [ce for i, ce in enumerate(plot_f) if tokens_plot[i - (i % 3) + 2] == single_specific]
        marker_ce_s = [ce for i, ce in enumerate(plot_s) if tokens_plot[i - (i % 3) + 2] == single_specific]
        marker_my_ce = [ce for i, ce in enumerate(plot_l) if tokens_plot[i - (i % 3) + 2] == single_specific]
        plt1.scatter(marker_indices, marker_ce_f, color='blue', alpha=alpha)
        plt1.scatter(marker_indices, marker_ce_s, color='green', alpha=alpha)
        plt1.scatter(marker_indices, marker_my_ce, color='red', alpha=alpha)
    plt1.legend()
    plt1.set_xlabel('Time')
    plt1.set_ylabel('cross_entr')
    plt1.set_title('Plot of cross_entr values')

    plt2 = plt.subplot(gs[1], sharex=plt1)
    visualize(tokens_plot, plt2, length=max_time)

    fig.tight_layout()

    # Save all four plots to image files
    fig.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{i}.png', dpi=300)
    fig.show()


def create_graph_and_visualization_idx(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, idx, use_filter=False, alpha=0.7):
    START_IDX = 0

    tokens_plot = np.array(tokens[0,1:].cpu())
    if tokens_plot[0] == SEPARATOR:
        START_IDX = 3
    
    tokens_plot = tokens_plot[START_IDX:]
    curr_ce_f = curr_ce_f[START_IDX:PRINT_UP_TO_IDX]
    curr_ce_s = curr_ce_s[START_IDX:PRINT_UP_TO_IDX]
    for i in range(len(LONG_MODELS_NAME)):
        my_cross_entr[i] = my_cross_entr[i][START_IDX:PRINT_UP_TO_IDX]
    should_print = False
    # Create an array of indices from 0 to length-1 of `ce`
    time_indices = tokens_plot[0::3]
    print("time_indices", time_indices[-100:])
    indices = np.cumsum(time_indices)
    print("indices", indices[-100:])
    max_time = max(indices)
    print("max_time", max_time)

    # Plot 1: ce
    fig = plt.figure(idx+1, figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])  # Ratio determines plot heights

    plt1 = plt.subplot(gs[0])
    if use_filter:
        # create a kalman filter to smooth the plot
        n = FILTER_CONST  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        plt1.plot(indices, lfilter(b, a, curr_ce_f[idx::3]), label=SHORTF_MODEL_LABEL, color='blue', alpha=alpha)
        plt1.plot(indices, lfilter(b, a, curr_ce_s[idx::3]), label=SHORTS_MODEL_LABEL, color='green', alpha=alpha)
        for i in range(len(LONG_MODELS_NAME)):
            plt1.plot(indices, lfilter(b, a, my_cross_entr[i][idx::3]), label=LONG_MODELS_LABEL[i], color=LONG_COLORS[i], alpha=alpha)
    else:
        plt1.plot(indices, curr_ce_f[idx::3], label=SHORTF_MODEL_LABEL, color='blue', alpha=alpha)
        plt1.plot(indices, curr_ce_s[idx::3], label=SHORTS_MODEL_LABEL, color='green', alpha=alpha)
        for i in range(len(LONG_MODELS_NAME)):
            plt1.plot(indices, my_cross_entr[i][idx::3], label=LONG_MODELS_LABEL[i], color=LONG_COLORS[i], alpha=alpha)
    plt1.legend()
    plt1.set_xlabel('Time')
    plt1.set_ylabel('cross_entr')
    plt1.set_title(f'Plot of cross_entr[{idx}::3] values')

    plt2 = plt.subplot(gs[1], sharex=plt1)
    visualize(tokens_plot, plt2, length=max_time)

    fig.tight_layout()

    # Save all four plots to image files
    fig.savefig(f'{OUTPUT_DIR}/{DATAFILE}_{i}_{idx}_3.png', dpi=300)
    fig.show()



if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized MIDI dataset')
    parser.add_argument('-f', '--filename',
                        help='file containing a tokenized MIDI dataset',
                        default=DATA)
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    args = parser.parse_args()

    print(f'Calculating log-loss for {args.filename}')
    print(f'Using short finetune model {SHORTF_CHECKPOINT}')
    print(f'Using short singlestage model {SHORTS_CHECKPOINT}')
    print(f'Using long model {LONG_CHECKPOINTS}')
    print(f'Sub-sampling results at rate {SUBSAMPLE}')
    num_samples = 0
    should_print = PRINT_GRAPH
    with open(args.filename, 'r') as f:
        shortf_ce = torch.empty(0)
        shorts_ce = torch.empty(0)
        long_ce = []
        for i in range(len(LONG_MODELS_NAME)):
            long_ce.append(torch.empty(0))
        for i,line in tqdm(list(enumerate(f))):
            num_samples += 1
            if i % SUBSAMPLE != PRINT_IDX: continue
            
            tokens = [int(token) for token in line.split()]
            #tokens = tokens[:8192]
            if tokens[0] != AUTOREGRESS:
                print("inserting AUTOREGRESS")
                tokens.insert(0, AUTOREGRESS)

            with torch.no_grad():
                curr_ce_f = sliding_window_eval(tokens, shortf_model)
                shortf_ce = torch.cat([shortf_ce, curr_ce_f])

                curr_ce_s = sliding_window_eval(tokens, shorts_model)
                shorts_ce = torch.cat([shorts_ce, curr_ce_s])

                # do long sequences
                #print("len tokens", len(tokens))
                tokens = torch.tensor(tokens).unsqueeze(0).cuda()
                my_cross_entr = []
                for i in range(len(LONG_MODELS_NAME)):
                    long_model = long_models[i]
                    logits = long_model(tokens).logits[0]
                    logits = logits.cuda()
                    cross_entr = F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none')
                    my_cross_entr_i = cross_entr.cpu()
                    my_cross_entr.append(my_cross_entr_i)
                    long_ce[i] = torch.cat([long_ce[i], my_cross_entr_i])

                if should_print:
                    print(f'printing example {i} to {OUTPUT_DIR}')
                    # print only the 3rd sample, the first 8652 tokens
                    create_graph_and_visualization(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, use_filter=USE_FILTER, single_specific=SINGLE_SPECIFIC)
                    create_graph_and_visualization_idx(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, 0, use_filter=USE_FILTER)
                    create_graph_and_visualization_idx(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, 1, use_filter=USE_FILTER)
                    create_graph_and_visualization_idx(curr_ce_f, curr_ce_s, my_cross_entr, tokens, i, 2, use_filter=USE_FILTER)


        print('num samples', num_samples)
        print(num_samples * (LONG_SEQ_LEN - 1))

        print(f"SHORTF MODEL {SHORTF_MODEL_NAME}")
        #print('ce', shortf_ce)
        short_L = shortf_ce.mean()
        print('Tokens processed:', len(shortf_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', short_L)
        print('  -> bits per second: ', short_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*shortf_ce.mean()))
            print('  -> onset perplexity: ', exp(shortf_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(shortf_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(shortf_ce[2::3].mean()))


        print(f"SHORTS MODEL {SHORTS_MODEL_NAME}")
        #print('ce', shorts_ce)
        short_L = shorts_ce.mean()
        print('Tokens processed:', len(shorts_ce))
        print('Log-losses')
        print('  -> per-token log-loss (nats): ', short_L)
        print('  -> bits per second: ', short_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
        if not args.interarrival:
            print('  -> per-event perplexity: ', exp(EVENT_SIZE*shorts_ce.mean()))
            print('  -> onset perplexity: ', exp(shorts_ce[0::3].mean()))
            print('  -> duration perplexity: ', exp(shorts_ce[1::3].mean()))
            print('  -> note perplexity: ', exp(shorts_ce[2::3].mean()))

        for i in range(len(LONG_MODELS_NAME)):
            print(f"LONG MODEL {LONG_MODELS_NAME[i]}")
            #print('ce', long_ce)
            long_L = long_ce[i].mean()
            print('Tokens processed:', len(long_ce[i]))
            print('Log-losses')
            print('  -> per-token log-loss (nats): ', long_L)
            #print('  -> bits per second: ', long_L*np.log2(np.e)*(num_samples * (LONG_SEQ_LEN - 1) / (560.98*3600)))
            if not args.interarrival:
                print('  -> per-event perplexity: ', exp(EVENT_SIZE*long_ce[i].mean()))
                print('  -> onset perplexity: ', exp(long_ce[i][0::3].mean()))
                print('  -> duration perplexity: ', exp(long_ce[i][1::3].mean()))
                print('  -> note perplexity: ', exp(long_ce[i][2::3].mean()))

