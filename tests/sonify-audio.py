from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch, torchaudio

from anticipation import audio
from anticipation import audiovocab
from anticipation.audiovocab import SEPARATOR, SCALE_OFFSET

from encodec.model import EncodecModel


if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized audio dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()

    model = EncodecModel.encodec_model_48khz()
    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            print(f'{i} Tokenized Audio Length: {len(tokens)} tokens')
            print("separator", SEPARATOR)
            print("scale_offset", SCALE_OFFSET)

            separator_indices = [i for i, x in enumerate(tokens) if x == SEPARATOR]
            print("separator_indices", separator_indices)

            #new_start = ((max(separator_indices)+3) // 4) * 4
            #print("new_start", new_start)
            #tokens = tokens[new_start:]
            # seek for the first complete frame
            firstseek = -1
            for seek, tok in enumerate(tokens):
                if SCALE_OFFSET < tok < SCALE_OFFSET + 100:
                    print("scale token at:", seek)
                    if firstseek == -1:
                        firstseek = seek
                    #break

            print("firstseek", firstseek)
            tokens = tokens[firstseek:]
            
            # stop sonifying at EOS
            try:
                eos = tokens.index(SEPARATOR)
            except ValueError:
                pass
            else:
                tokens = tokens[:eos]

            print(f'{i} Tokenized Audio Length: {len(tokens)} tokens')

            frames, scales = audio.detokenize(tokens, audiovocab.vocab)
            print("Frames len", len(frames))
            print("Scales len", len(scales))
            print("Scales", scales)
            print(frames[0].shape, frames[-1].shape)
            print(frames[0].min(), frames[0].max(), frames[-1].min(), frames[-1].max())
            with torch.no_grad():
                wav = model.decode(zip(frames, [torch.tensor(s/100.).view(1) for s in scales]))[0]
            lastDot = args.filename.rfind(".")
            #filename_audio = f'{args.filename[:lastDot]}.wav'
            filename_audio = '/nlp/scr/kathli/output/audio/dataset/test-{i}.wav'
            torchaudio.save(filename_audio, wav, model.sample_rate)