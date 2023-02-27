from argparse import ArgumentParser
from pathlib import Path

from anticipation.convert import events_to_midi

if __name__ == '__main__':
    parser = ArgumentParser(description='inspect a MIDI dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            tokens = tokens[1:] # strip control codes
            mid = events_to_midi(tokens)
            mid.save(f'output/{Path(args.filename).stem}{i}.mid')
            print(f'{i} Tokenized MIDI Length: {mid.length} seconds ({len(tokens)} tokens)')
