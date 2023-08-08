"""
Utilities for inspecting encoded music data.
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import anticipation.ops as ops
from anticipation.config import *
from anticipation.vocab import *

def visualize(tokens, output, selected=None, length=120):
    #colors = ['white', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan',
    #          'dodgerblue', 'slategray', 'navy', 'mediumpurple', 'mediumorchid', 'magenta', 'lightpink']
    colors = ['white', '#426aa0', '#b26789', '#de9283', '#eac29f', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan', 'dodgerblue', 'slategray', 'navy']

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
 
    max_time = length * TIME_RESOLUTION #ops.max_time(tokens, seconds=False)
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

        if note == REST:
            continue

        assert note < CONTROL_OFFSET

        cur_time += tm
        tm = cur_time - TIME_OFFSET
        dur = dur - DUR_OFFSET
        note = note - NOTE_OFFSET
        instr = note//2**7
        pitch = note - (2**7)*instr

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
        #if cur_time > max_time:
        #    print('time has exceeded max_time', cur_time, max_time)
        #    break

    print('cur_time', cur_time)

    plt.clf()
    plt.axis('off')
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = list(range(MAX_TRACK_INSTR)) + [16]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(np.flipud(grid.T), aspect=16, cmap=cmap, norm=norm, interpolation='none')

    patches = [matplotlib.patches.Patch(color=colors[i+1], label=f"{instruments[i]}")
               for i in range(len(instruments))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.tight_layout()
    plt.savefig(output)
