#!/bin/bash

echo "Running multi-len with mixin"
python scripts/eval-loss-multi-len.py -m fh86cy5o -n 115000

echo "Running multi-len with 2stage"
python scripts/eval-loss-multi-len.py -m w7uq05r3 -n 115000

echo "Running multi-len with long full"
python scripts/eval-loss-multi-len.py -m genial-sea-181 -n 115000

echo "Running multi-len with finetune"
python scripts/eval-loss-multi-len.py -m 1381mqhz  -n 15000

echo "Running multi-len with text"
python scripts/eval-loss-multi-len.py -m kly1r553 -n 115000

echo "Running multi-len with long medium"
python scripts/eval-loss-multi-len.py -m uk31obfe  -n 115000