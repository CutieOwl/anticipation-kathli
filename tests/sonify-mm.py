from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch, torchaudio

from anticipation import audio
from anticipation.convert import compound_to_midi

from encodec.model import EncodecModel


def mm_to_compound(blocks, vocab, debug=False):
    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    rest = vocab['rest'] - pitch_offset

    tokens = blocks.T.flatten().tolist()

    out = 5*(len(tokens)//4)*[None]
    out[0::5] = [tok - time_offset for tok in tokens[0::4]]
    out[1::5] = [tok - dur_offset for tok in tokens[3::4]]
    out[2::5] = [tok - pitch_offset for tok in tokens[1::4]]
    out[3::5] = [tok - instr_offset for tok in tokens[2::4]]
    out[4::5] = (len(tokens)//4)*[72] # default velocity

    # convert interarrival times to arrival times
    time = 0
    out_norest = []
    for idx in range(len(out)//5):
        time += out[5*idx]
        if out[5*idx+2] == rest:
            continue

        out_norest.extend(out[5*idx:5*(idx+1)])
        out_norest[-5] = time

    return out_norest


def split(blocks, vocab, debug=False):
    """ split token blocks into midi and audio"""

    midi_offset = vocab['midi_offset']

    audio = torch.zeros([4,0], dtype=blocks.dtype)
    midi = torch.zeros([4,0], dtype=blocks.dtype)
    time = 0
    for i, block in enumerate(blocks.T):
        if block[0] < midi_offset:
            audio = torch.cat((audio, block.unsqueeze(1)), dim=1)
        else:
            if debug:
                print('MIDI event at sequence position', i)
                print('  MIDI sequence interrarival time is', )


            midi = torch.cat((midi, block.unsqueeze(1)), dim=1)

    return audio, midi 


if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized audio dataset')
    parser.add_argument('filename',
        help='file containing a tokenized audio dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    parser.add_argument('-v', '--vocab', default='mm',
        help='name of the audio vocabulary used in the input file {audio|mm}')
    parser.add_argument('--debug', action='store_true', help='verbose debugging outputs')
    args = parser.parse_args()

    if args.vocab == 'audio':
        from anticipation.audiovocab import vocab
    elif args.vocab == 'mm':
        from anticipation.mmvocab import vocab
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')

    #save_path = '/nlp/scr/kathli/mm_tok_out/sonify/'
    save_path = args.filename[:args.filename.rfind('/')]
    print(save_path)
 
    separator = vocab['separator']
    scale_offset = vocab['scale_offset']
    scale_res = vocab['config']['scale_resolution']
    skew = vocab['config']['skew']
    print(separator, scale_offset, scale_res, skew)
    model = EncodecModel.encodec_model_48khz()
    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            blocks = tokens

            print('Tokens:', len(tokens))

            # strip the control block
            tokens = tokens[4:]

            # find all instances of separator and print their indexes
            sep_idxs = [idx for idx, token in enumerate(tokens) if token == separator]
            print('Separator indexes:', sep_idxs)

            # strip sequence separators
            tokens = [token for token in tokens if token != separator]

            print('Tokens:', len(tokens))

            # strip control tokens
            tokens = [token for token in tokens if token > 10]

            print('Tokens:', len(tokens))

            #tokens = tokens[:6628]

            if skew:
                blocks = audio.deskew(tokens, 4)
            else:
                blocks = torch.tensor(tokens).reshape(-1, 4).T

            if args.vocab == 'mm':
                blocks, midi_blocks = split(blocks, vocab, args.debug)
                print(blocks.shape, midi_blocks.shape)
                print(midi_blocks)
                if midi_blocks.shape[1] > 0:
                    #mid = compound_to_midi(blocks, vocab) # this is compound only
                    mid = compound_to_midi(mm_to_compound(midi_blocks, vocab), vocab)

                    mid.save(f'{save_path}/{Path(args.filename).stem}-{i}.mid')
                    print('Saved midi to', f'{save_path}/{Path(args.filename).stem}-{i}.mid')

            if blocks.shape[1] == 0:
                continue

            first_seek = -1
            for seek in range(blocks.shape[1]):
                if blocks[0,seek] >= scale_offset and blocks[0,seek] < scale_offset + scale_res:
                    if first_seek == -1:
                        first_seek = seek
                    print(seek, blocks[0,seek])

            # delete the awkward bit between 906 and 1361
            #blocks = torch.cat((blocks[:,:906], blocks[:,1361:]), dim=1)

            # seek for the first complete frame
            for seek, block in enumerate(blocks.T):
                if scale_offset <= block[0] < scale_offset + scale_res:
                    break

            print('Seek to:', seek)

            blocks = blocks[:,seek:]
            if blocks.shape[1] > 0:
                frames, scales = audio.detokenize(blocks, vocab)
                print(scales)
                print(frames[-1].shape)
                # frames = frames[:-6]
                # scales = scales[:-6]
                if frames[-1].shape[2] == 1:
                    frames = frames[:-1]
                    scales = scales[:-1]
                with torch.no_grad():
                    wav = model.decode(zip(frames, [torch.tensor(s/100.).view(1) for s in scales]))[0]
                #save_path = "/nlp/scr/kathli/output/mm/pu4yo6b5"
                torchaudio.save(f'{save_path}/{Path(args.filename).stem}-{i}.wav', wav, model.sample_rate)
                print('Saved wav to', f'{save_path}/{Path(args.filename).stem}-{i}.wav')
