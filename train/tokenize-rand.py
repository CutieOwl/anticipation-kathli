import os, math
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

from anticipation.mmvocab import vocab
from anticipation.audio import read_ecdc, skew
from anticipation.audio import tokenize as tokenize_audio
from anticipation.tokenize import maybe_tokenize


def pack_tokens_rand(output, seqlen, num_seqs, seed):
    files = bad_files = seqcount = 0
    vocab_size = vocab['config']['size']

    np.random.seed(seed)

    with open(output, 'w') as outfile:
        for i in range(num_seqs):
            random_integers = np.random.randint(0, vocab_size + 1, seqlen)
            outfile.write(' '.join([str(tok) for tok in random_integers]) + '\n')
            seqcount += 1

    return (files, bad_files, seqcount)


def main(args):
    print('Tokenization parameters:')
    print(f"  context = {args.context}")
    print(f"  workers = {args.workers}")
    print(f"  debug = {args.debug}")
    print(f"  numseqs = {args.numseqs}")
    print(f"  seed = {args.seed}")

    n = args.numseqs // args.workers
    context = args.workers*[args.context]
    num_seqs = args.workers*[n]
    seeds = args.workers*[args.seed]
    outfiles = os.path.join(args.outdir, 'random.shard-{s:03}.txt')
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(s=s) for s in range(args.workers)]

    print('Processing...')
    if args.debug:
        results = pack_tokens_rand(outputs[0], context[0], num_seqs[0], seeds[0])
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            results = pool.starmap(pack_tokens_rand, zip(outputs, context, num_seqs, seeds))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a multimodal dataset (ecdc audio paired with midi)')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('context', type=int, help='context length for packing training sequences')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')
    parser.add_argument('--numseqs', type=int, default=6400000, help='number of sequences to generate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    main(parser.parse_args())
