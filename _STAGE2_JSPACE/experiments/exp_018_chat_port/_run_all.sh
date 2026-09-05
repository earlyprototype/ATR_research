#!/bin/sh
set -x
cd "$(dirname "$0")"
python3 run_exp018.py --stage loop --arm bare --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_bare.log 2>&1
python3 run_exp018.py --stage loop --arm chat --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_chat.log 2>&1
echo "ALL LOOPS DONE" >> output/loop_chat.log
