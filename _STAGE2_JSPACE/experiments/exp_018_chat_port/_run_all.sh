#!/bin/sh
# The commands the registered run used. `set -e` was added after the run, in
# the review fixes: the registered run used these same two commands without it.
set -eux
cd "$(dirname "$0")"
python3 run_exp018.py --stage loop --arm bare --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_bare.log 2>&1
python3 run_exp018.py --stage loop --arm chat --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_chat.log 2>&1
echo "ALL LOOPS DONE" >> output/loop_chat.log
