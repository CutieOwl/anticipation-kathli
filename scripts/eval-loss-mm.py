import os,csv,time,json,sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')

from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from anticipation.config import EVENT_SIZE

MODEL = "49fupsao"
STEP_NUM = 99481
SUBSAMPLE = 1000
MODEL_DIR = f'/nlp/scr/kathli/checkpoints/audio-checkpoints/{MODEL}'
DATA = "/juice4/scr4/nlp/music/audio/dataset/test.txt" # this is the old audio tokenization dataset
#DATA = "/juice4/scr4/nlp/music/datasets/valid/lakh.midigen.valid.txt" # this is the new midi tokenization dataset
OUTPUT_DIR = f'/nlp/scr/kathli/output/audio/{MODEL}'
OUTPUT_FILE = f'eval.csv'

def log_loss(model, datafile, subsample):
    with open(datafile, 'r') as data:
        ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(data))):
            if i % subsample != 0:
                continue

            tokens = [int(token) for token in line.split()]
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(tokens).logits[0]
                ce = torch.cat([ce, F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none').cpu()])

    return ce


def main(args):
    print(f'Sub-sampling results at rate {args.subsample}')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    results = os.path.join(args.output, OUTPUT_FILE)
    print(f'Storing results at {results}')

    checkpoints = [os.path.join(f.path, 'hf') for f in os.scandir(args.model) if
            f.is_dir() and os.path.basename(f).startswith('step-')]

    if args.all:
        print('Calculating log-loss for checkpoints:')
        for ckpt in checkpoints:
            print('  ', ckpt)
    else:
        steps = [int(ckpt.split(os.sep)[-2][5:]) for ckpt in checkpoints]
        checkpoints = [os.path.join(args.model, f'step-{max(steps)}', 'hf')]
        print('Calculating log-loss for final checkpoint:')
        print('  ', checkpoints[0])

    print('Calculating log-loss on dataset:')
    print('  ', args.filename)
    with open(results, 'w', newline='') as f:
        fields = ['step', 'loss']
        if not args.interarrival:
            fields.extend(['event_ppl', '0_ppl', '1_ppl', '2_ppl', '3_ppl'])

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ckpt in checkpoints:
            step = int(ckpt.split(os.sep)[-2][5:])
            print(f'Loading checkpoint (step {step}):')
            print('  ', ckpt)
            t0 = time.time()
            model_config_json = json.load(open(f"{ckpt}/config.json"))
            print("n_positions", model_config_json["n_positions"])
            model_config = GPT2Config(model_config_json)
            model = GPT2LMHeadModel.from_pretrained(ckpt, config=model_config).cuda()
            print(f'  loaded in {time.time()-t0} seconds')

            ce = log_loss(model, args.filename, args.subsample)

            res = {}
            res['step'] = step
            res['loss'] = np.round(ce.mean().item(), 3)
            if not args.interarrival:
                res['event_ppl'] = np.round(np.exp(EVENT_SIZE*ce.mean().item()), 3)
                res['0_ppl'] = np.round(np.exp(ce[0::4].mean().item()), 3)
                res['1_ppl'] = np.round(np.exp(ce[1::4].mean().item()), 3)
                res['2_ppl'] = np.round(np.exp(ce[2::4].mean().item()), 3)
                res['3_ppl'] = np.round(np.exp(ce[3::4].mean().item()), 3)

            writer.writerow(res)


if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized dataset')
    parser.add_argument('-f', '--filename', help='file containing a tokenized dataset', default=DATA)
    parser.add_argument('-m', '--model', help='file containing a model to evaluate', default=MODEL_DIR)
    parser.add_argument('-o', '--output', help='output dir', default=OUTPUT_DIR)
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose console output')
    parser.add_argument('-a', '--all', action='store_true',
            help='calculate loss for all checkpoints')
    parser.add_argument('--bpe', action='store_true',
            help='calculate loss for all checkpoints')
    parser.add_argument('-i', '--interarrival', action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    parser.add_argument('-s', '--subsample', type=int, default=SUBSAMPLE,
            help='dataset subsampling ratio')

    main(parser.parse_args())
